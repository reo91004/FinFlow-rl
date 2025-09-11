# src/core/trainer.py

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from src.core.env import PortfolioEnv
from src.core.replay import PrioritizedReplayBuffer, OfflineDataset
from src.core.iql import IQLAgent
from src.core.sac import DistributionalSAC
from src.agents.b_cell import BCell
from src.agents.t_cell import TCell
from src.agents.memory import MemoryCell
from src.agents.gating import GatingNetwork
from src.analysis.monitor import PerformanceMonitor
from src.analysis.visualization import plot_equity_curve, plot_drawdown, plot_portfolio_weights
from src.utils.monitoring import StabilityMonitor
from src.utils.logger import FinFlowLogger, get_session_directory
from src.utils.seed import set_seed, get_device_info

@dataclass
class TrainingConfig:
    """학습 설정"""
    # Environment
    env_config: Dict = field(default_factory=lambda: {
        'initial_balance': 1000000,
        'transaction_cost': 0.001,
        'max_weight': 0.2,
        'min_weight': 0.0,
        'window_size': 30,
        'max_weight_change': 0.2  # PerformanceMonitor용
    })
    
    # Data configuration
    data_config: Dict = field(default_factory=lambda: {
        'tickers': None,  # Must be provided
        'start': '2008-01-01',  # training start date
        'end': '2020-12-31',  # training end date
        'test_start': '2021-01-01',  # test start date
        'test_end': '2024-12-31',  # test end date
        'cache_dir': 'data/cache',  # cache directory
        'interval': '1d',  # daily data
        'auto_download': True,  # auto download if missing
        'use_cache': True  # use cached data if available
    })
    
    # Training configuration
    train_config: Dict = field(default_factory=lambda: {
        'offline_episodes': 100,  # 오프라인 데이터 수집 에피소드
        'offline_steps': 200000,
        'offline_batch_size': 512,
        'offline_eval_interval': 10000
    })
    
    # IQL Pretraining
    iql_epochs: int = 100
    iql_batch_size: int = 256
    iql_lr: float = 3e-4
    iql_expectile: float = 0.7
    iql_temperature: float = 3.0
    
    # SAC Fine-tuning
    sac_episodes: int = 1000
    sac_batch_size: int = 256
    sac_lr: float = 3e-4
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_alpha: float = 0.2
    sac_cql_weight: float = 1.0
    
    # Memory
    memory_capacity: int = 50000
    memory_k_neighbors: int = 5
    
    # Monitoring
    eval_interval: int = 10
    checkpoint_interval: int = 50
    log_interval: int = 1
    
    # Device & Seed
    device: str = "auto"
    seed: int = 42
    
    # Paths
    data_path: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    
    # Target metrics
    target_sharpe: float = 1.5
    target_cvar: float = -0.02
    
    # Early stopping
    patience: int = 50
    min_improvement: float = 0.01
    
    # Monitoring configuration
    monitoring_config: Optional[Dict] = None


class FinFlowTrainer:
    """
    FinFlow 통합 학습 관리자
    
    IQL 사전학습 → SAC 미세조정 파이프라인
    T-Cell, B-Cell, Memory, Gating 통합 관리
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Args:
            config: 학습 설정
        """
        self.config = config
        self.logger = FinFlowLogger("Trainer")
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Device setup
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        self.logger.info(f"디바이스 설정: {get_device_info(self.device)}")
        
        # Create directories
        self.session_dir = Path(get_session_directory())
        self.log_dir = str(self.session_dir)
        self.run_dir = self.session_dir
        self.checkpoint_dir = self.session_dir / "models"
        self.checkpoint_dir.mkdir(exist_ok=True)
        (self.run_dir / "alerts").mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Training state
        self.global_step = 0
        self.episode = 0
        self.best_sharpe = -float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.metrics_history = []
        
        self.logger.info("FinFlow Trainer 초기화 완료")
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        # 실제 데이터 로드
        from src.data.loader import DataLoader
        from src.data.features import FeatureExtractor
        import pandas as pd
        
        self.logger.info("실제 시장 데이터를 로드합니다...")
        
        # DataLoader를 사용하여 실제 데이터 로드
        data_loader = DataLoader(cache_dir="data/cache")
        
        # 설정에서 티커 가져오기 (config에서 읽기)
        tickers = self.config.data_config.get('symbols')
        
        # config에서 날짜 읽기
        config_data = self.config.data_config
        market_data = data_loader.get_market_data(
            symbols=tickers,  # 모든 티커 사용
            train_start=config_data.get('start', '2008-01-01'),
            train_end=config_data.get('end', '2020-12-31'),
            test_start=config_data.get('test_start', '2021-01-01'),
            test_end=config_data.get('test_end', '2024-12-31')
        )
        
        # 학습 데이터 선택
        price_data = market_data['train_data']
        
        if price_data.empty:
            raise ValueError("시장 데이터 로드 실패. 인터넷 연결을 확인하세요.")
        
        self.logger.info(f"데이터 로드 성공: {len(price_data)} 일, {len(price_data.columns)} 자산")
        
        # 특성 추출기 (config 전달)
        feature_config = config_data.get('features', {}) if 'features' in locals() else {}
        self.feature_extractor = FeatureExtractor(
            window=self.config.env_config.get('window_size', 30),
            feature_config=feature_config
        )
        
        # Environment 생성
        self.env = PortfolioEnv(
            price_data=price_data,
            feature_extractor=self.feature_extractor,
            initial_capital=self.config.env_config.get('initial_balance', 1000000),
            transaction_cost=self.config.env_config.get('transaction_cost', 0.001),
            max_leverage=self.config.env_config.get('max_leverage', 1.0)
        )
        
        # Get dimensions
        obs = self.env.reset()[0]
        self.state_dim = len(obs)
        self.action_dim = self.env.action_space.shape[0]
        
        self.logger.info(f"환경 초기화: state_dim={self.state_dim}, action_dim={self.action_dim}")
        
        # T-Cell (Crisis Detection) - config 기반
        feature_config = self.config.data_config.get('features', {}) if hasattr(self.config, 'data_config') else {}
        self.t_cell = TCell(
            feature_dim=None,  # config에서 자동 계산
            contamination=0.1,
            n_estimators=100,
            window_size=30,
            feature_config=feature_config
        )
        
        # Memory Cell
        self.memory_cell = MemoryCell(
            capacity=self.config.memory_capacity,
            embedding_dim=32,
            k_neighbors=self.config.memory_k_neighbors
        )
        
        # Gating Network
        self.gating_network = GatingNetwork(
            state_dim=self.state_dim,
            hidden_dim=256,
            num_experts=5
        ).to(self.device)
        
        # B-Cell (Main Agent) - 기본 전략으로 초기화
        bcell_config = {
            'actor_hidden': [256, 256],
            'critic_hidden': [256, 256],
            'actor_lr': self.config.sac_lr,
            'critic_lr': self.config.sac_lr,
            'gamma': self.config.sac_gamma,
            'tau': self.config.sac_tau,
            'alpha': self.config.sac_alpha,
            'cql_weight': self.config.sac_cql_weight,
            'n_quantiles': 32
        }
        
        # 여러 B-Cell 전략 초기화
        self.b_cells = {}
        for specialization in ['volatility', 'correlation', 'momentum', 'defensive', 'growth']:
            self.b_cells[specialization] = BCell(
                specialization=specialization,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                config=bcell_config,
                device=self.device
            )
        
        # 기본 B-Cell 선택
        self.b_cell = self.b_cells['momentum']
        
        # IQL Agent for pretraining
        self.iql_agent = IQLAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            expectile=self.config.iql_expectile,
            temperature=self.config.iql_temperature,
            device=self.device
        )
        
        # Stability Monitor 초기화
        stability_config = {
            'window_size': 100,
            'n_sigma': 3.0,
            'intervention_threshold': 3,
            'rollback_enabled': True,
            'q_value_max': 100.0,
            'q_value_min': -100.0,
            'entropy_min': 0.1,
            'gradient_max': 10.0,
            'concentration_max': 0.5,
            'turnover_max': 0.5
        }
        self.stability_monitor = StabilityMonitor(stability_config)
        self.logger.info("StabilityMonitor 초기화 완료")
        
        # Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.memory_capacity,
            alpha=0.6,
            beta=0.4
        )
        
        # Performance Monitor
        monitoring_config = self.config.monitoring_config or {}
        self.performance_monitor = PerformanceMonitor(
            log_dir=self.log_dir,
            use_wandb=monitoring_config.get('use_wandb', False),
            use_tensorboard=monitoring_config.get('use_tensorboard', True),
            wandb_config=monitoring_config
        )
        
        self.logger.info("모든 컴포넌트 초기화 완료")
    
    def train(self):
        """전체 학습 파이프라인 실행"""
        self.logger.info("=" * 50)
        self.logger.info("FinFlow 학습 시작")
        self.logger.info("=" * 50)
        
        # Phase 1: IQL Pretraining
        self.logger.info("\n[Phase 1] IQL 오프라인 사전학습")
        if not self._check_offline_data():
            self.logger.info("오프라인 데이터가 없습니다. 데이터를 생성합니다...")
            self._prepare_offline_data()
        self._pretrain_iql()
        
        # Phase 2: Online SAC Fine-tuning
        self.logger.info("\n[Phase 2] SAC 온라인 미세조정")
        self._train_sac()
        
        # Phase 3: Final Evaluation
        self.logger.info("\n[Phase 3] 최종 평가")
        final_metrics = self._evaluate()
        
        # Save final model
        self._save_checkpoint("final")
        
        # Generate report
        self._generate_report(final_metrics)
        
        self.logger.info("=" * 50)
        self.logger.info("학습 완료!")
        self.logger.info(f"최종 Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.3f}")
        self.logger.info(f"최종 CVaR(5%): {final_metrics.get('cvar_5', 0):.3f}")
        self.logger.info("=" * 50)
    
    def _check_offline_data(self) -> bool:
        """오프라인 데이터 존재 확인"""
        data_path = Path(self.config.data_path)
        if not self.config.data_config.get('use_cache', True):
            return False
        return data_path.exists() and len(list(data_path.glob("*.npz"))) > 0
    
    def _prepare_offline_data(self):
        """오프라인 데이터 준비 - OfflineDataset.collect_from_env() 사용"""
        data_path = Path(self.config.data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # 현재 환경에서 데이터 수집
        if hasattr(self, 'env') and self.env is not None:
            self.logger.info("환경에서 오프라인 데이터를 수집합니다...")
            
            # OfflineDataset 생성 및 데이터 수집
            from src.core.offline_dataset import OfflineDataset
            
            # config에서 에피소드 수 가져오기 (기본값: 100)
            n_episodes = self.config.train_config.get('offline_episodes', 100)
            self.logger.info(f"{n_episodes}개 에피소드로 오프라인 데이터 수집")
            
            dataset = OfflineDataset()
            dataset.collect_from_env(
                env=self.env,
                n_episodes=n_episodes,
                diversity_bonus=True,
                verbose=True
            )
            
            # 데이터셋 저장
            save_path = data_path / 'offline_data.npz'
            dataset.save(save_path)
            self.logger.info(f"오프라인 데이터셋 저장: {save_path}")
            return
        
        # 환경이 없으면 오류
        raise ValueError(
            "환경이 초기화되지 않았습니다. "
            "trainer를 생성할 때 환경이 설정되었는지 확인하세요."
        )
    
    
    def _pretrain_iql(self):
        """IQL 오프라인 사전학습"""
        # Prepare data if not exists
        if not self._check_offline_data():
            self.logger.info("오프라인 데이터 준비 중...")
            self._prepare_offline_data()
        
        # Load offline dataset
        from src.core.offline_dataset import OfflineDataset
        dataset = OfflineDataset(self.config.data_path)
        
        self.logger.info(f"오프라인 데이터셋 로드: {len(dataset)} samples")
        self.logger.info("=" * 50)
        self.logger.info("IQL 오프라인 사전학습 시작")
        self.logger.info("=" * 50)
        
        # Training loop with progress bar
        from tqdm import tqdm
        pbar = tqdm(range(self.config.iql_epochs), desc="IQL Pretraining", unit="epoch")
        
        for epoch in pbar:
            epoch_losses = []
            
            # Mini-batch training
            for _ in range(len(dataset) // self.config.iql_batch_size):
                batch = dataset.sample_batch(self.config.iql_batch_size)
                
                # Convert to tensors
                states = torch.FloatTensor(batch['states']).to(self.device)
                actions = torch.FloatTensor(batch['actions']).to(self.device)
                rewards = torch.FloatTensor(batch['rewards']).to(self.device)
                next_states = torch.FloatTensor(batch['next_states']).to(self.device)
                dones = torch.FloatTensor(batch['dones']).to(self.device)
                
                # IQL update
                losses = self.iql_agent.update(
                    states, actions, rewards, next_states, dones
                )
                
                epoch_losses.append(losses)
                self.global_step += 1
            
            # Calculate average losses for this epoch
            if epoch_losses:
                avg_losses = {
                    k: np.mean([l[k] for l in epoch_losses])
                    for k in epoch_losses[0].keys()
                }
                
                # Update progress bar with metrics
                pbar.set_postfix({
                    'V_Loss': f"{avg_losses.get('value_loss', 0):.4f}",
                    'Q_Loss': f"{avg_losses.get('q_loss', 0):.4f}",
                    'Actor_Loss': f"{avg_losses.get('actor_loss', 0):.4f}"
                })
                
                # Log epoch metrics
                if (epoch + 1) % self.config.log_interval == 0:
                    self.logger.info(
                        f"IQL Epoch {epoch+1}/{self.config.iql_epochs} | "
                        f"V Loss: {avg_losses['value_loss']:.6f} | "
                        f"Q Loss: {avg_losses['q_loss']:.6f} | "
                        f"Actor Loss: {avg_losses['actor_loss']:.6f}"
                    )
                    
                    # IQL 학습 진단 정보
                    self.logger.debug(f"Value gradient norm: {avg_losses.get('value_grad_norm', 0):.6f}")
                    self.logger.debug(f"Q gradient norm: {avg_losses.get('q_grad_norm', 0):.6f}")
                    self.logger.debug(f"Actor gradient norm: {avg_losses.get('actor_grad_norm', 0):.6f}")
                    
                    self.logger.log_metrics(avg_losses, self.global_step)
        
        # Transfer knowledge to B-Cell
        self._transfer_iql_to_bcell()
        self.logger.info("IQL 사전학습 완료 및 지식 전이 완료")
    
    def _transfer_iql_to_bcell(self):
        """IQL에서 B-Cell로 완전한 지식 전이"""
        
        # 1. Actor 네트워크 전이 (정책)
        self.b_cell.actor.load_state_dict(
            self.iql_agent.actor.state_dict()
        )
        self.logger.info("Actor 네트워크 전이 완료")
        
        # 2. Value network를 Critic 초기화에 활용
        with torch.no_grad():
            # IQL의 value function을 SAC의 baseline으로 사용
            if hasattr(self.iql_agent, 'value'):
                self.b_cell.value_baseline = self.iql_agent.value
                
            # Q-network 가중치를 Critic 초기화에 활용
            # 주의: IQL은 단일 Q값, SAC는 Quantile 분포 사용
            if hasattr(self.b_cell, 'critic'):
                # 호환 가능한 레이어만 복사
                self._transfer_compatible_layers(
                    source=self.iql_agent.q1,
                    target=self.b_cell.critic.q1,
                    layer_mapping={
                        'fc1': 'fc1',  # 첫 번째 레이어는 동일
                        'fc2': 'fc2',  # 두 번째 레이어도 호환
                        # fc3는 출력 차원이 다르므로 제외
                    }
                )
                self._transfer_compatible_layers(
                    source=self.iql_agent.q2,
                    target=self.b_cell.critic.q2,
                    layer_mapping={
                        'fc1': 'fc1',
                        'fc2': 'fc2',
                    }
                )
        
        # 3. IQL 학습 통계 전이
        self.b_cell.initial_stats = {
            'iql_final_value': self._compute_iql_average_value(),
            'iql_final_q': self._compute_iql_average_q(),
            'iql_training_steps': self.iql_agent.training_steps if hasattr(self.iql_agent, 'training_steps') else 0
        }
        
        # 4. Temperature (alpha) 초기화
        # IQL의 advantage 분포를 기반으로 SAC의 엔트로피 목표 설정
        advantages = self._compute_iql_advantages()
        if advantages is not None:
            initial_entropy = -np.mean(advantages) * 0.1  # 휴리스틱
            self.b_cell.target_entropy = initial_entropy
        
        # 5. 모든 B-Cell 전략에 전이
        for bcell_name, bcell in self.b_cells.items():
            if bcell != self.b_cell:  # 기본 B-Cell은 이미 전이됨
                bcell.actor.load_state_dict(self.iql_agent.actor.state_dict())
                if hasattr(bcell, 'critic'):
                    self._transfer_compatible_layers(
                        source=self.iql_agent.q1,
                        target=bcell.critic.q1,
                        layer_mapping={'fc1': 'fc1', 'fc2': 'fc2'}
                    )
                    self._transfer_compatible_layers(
                        source=self.iql_agent.q2,
                        target=bcell.critic.q2,
                        layer_mapping={'fc1': 'fc1', 'fc2': 'fc2'}
                    )
                self.logger.debug(f"B-Cell [{bcell_name}] 지식 전이 완료")
        
        self.logger.info(f"지식 전이 완료: Value baseline={self.b_cell.initial_stats.get('iql_final_value', 0):.3f}")
    
    def _transfer_compatible_layers(self, source, target, layer_mapping):
        """호환 가능한 레이어만 선택적 전이"""
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        
        for src_name, tgt_name in layer_mapping.items():
            src_key_w = f"{src_name}.weight"
            src_key_b = f"{src_name}.bias"
            tgt_key_w = f"{tgt_name}.weight"
            tgt_key_b = f"{tgt_name}.bias"
            
            if src_key_w in source_dict and tgt_key_w in target_dict:
                if source_dict[src_key_w].shape == target_dict[tgt_key_w].shape:
                    target_dict[tgt_key_w] = source_dict[src_key_w].clone()
                    target_dict[tgt_key_b] = source_dict[src_key_b].clone()
                    self.logger.debug(f"레이어 전이: {src_name} → {tgt_name}")
                else:
                    self.logger.debug(f"레이어 크기 불일치: {src_name} {source_dict[src_key_w].shape} → {tgt_name} {target_dict[tgt_key_w].shape}")
        
        target.load_state_dict(target_dict)
    
    def _compute_iql_average_value(self):
        """IQL의 평균 value 계산"""
        if not hasattr(self.iql_agent, 'value'):
            return 0.0
        
        # 샘플 상태들에 대한 평균 value 계산
        with torch.no_grad():
            if len(self.replay_buffer) > 100:
                transitions, _, _ = self.replay_buffer.sample(100)
                states = torch.FloatTensor([t.state for t in transitions]).to(self.device)
                values = self.iql_agent.value(states)
                return values.mean().item()
        return 0.0
    
    def _compute_iql_average_q(self):
        """IQL의 평균 Q값 계산"""
        if not hasattr(self.iql_agent, 'q1'):
            return 0.0
        
        with torch.no_grad():
            if len(self.replay_buffer) > 100:
                transitions, _, _ = self.replay_buffer.sample(100)
                states = torch.FloatTensor([t.state for t in transitions]).to(self.device)
                actions = torch.FloatTensor([t.action for t in transitions]).to(self.device)
                q1_values = self.iql_agent.q1(states, actions)
                q2_values = self.iql_agent.q2(states, actions)
                return torch.min(q1_values, q2_values).mean().item()
        return 0.0
    
    def _compute_iql_advantages(self):
        """IQL의 advantage 분포 계산"""
        if not hasattr(self.iql_agent, 'value') or not hasattr(self.iql_agent, 'q1'):
            return None
        
        with torch.no_grad():
            if len(self.replay_buffer) > 100:
                transitions, _, _ = self.replay_buffer.sample(100)
                states = torch.FloatTensor([t.state for t in transitions]).to(self.device)
                actions = torch.FloatTensor([t.action for t in transitions]).to(self.device)
                
                values = self.iql_agent.value(states)
                q1_values = self.iql_agent.q1(states, actions)
                q2_values = self.iql_agent.q2(states, actions)
                q_values = torch.min(q1_values, q2_values)
                
                advantages = q_values - values
                return advantages.cpu().numpy()
        return None
    
    def _train_sac(self):
        """SAC 온라인 미세조정"""
        self.logger.info("=" * 50)
        self.logger.info("SAC 온라인 미세조정 시작")
        self.logger.info("=" * 50)
        
        # 누락된 속성 초기화
        self.all_costs = []
        self.last_action = np.zeros(self.env.n_assets)
        
        episode_rewards = []
        episode_sharpes = []
        episode_cvars = []
        
        from tqdm import tqdm
        pbar = tqdm(range(self.config.sac_episodes), desc="SAC Training", unit="episode")
        
        for episode in pbar:
            self.episode = episode
            episode_reward = 0
            episode_steps = 0
            self.episode_returns = []  # 에피소드 수익률 추적
            self.episode_actions = []  # 에피소드 액션 추적
            
            # Reset environment
            state, _ = self.env.reset()
            done = False
            
            # Episode loop
            while not done:
                # Crisis detection
                crisis_info = self.t_cell.detect_crisis(self.env.get_market_data())
                crisis_level = crisis_info['overall_crisis']
                
                # Memory guidance
                memory_guidance = self.memory_cell.get_memory_guidance(
                    state, crisis_level
                )
                
                # Gating decision
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                gating_decision = self.gating_network(
                    state_tensor, memory_guidance, crisis_level
                )
                
                # Select action using B-Cell
                action = self.b_cell.select_action(
                    state_tensor,
                    bcell_type=gating_decision.selected_bcell,
                    deterministic=False
                )
                
                # Store action for tracking
                self.last_action = action.copy()
                
                # Environment step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # 에피소드 데이터 추적
                portfolio_return = info.get('portfolio_return', 0)
                if portfolio_return == 0:  # fallback
                    portfolio_return = reward / self.config.env_config['initial_balance']
                self.episode_returns.append(portfolio_return)
                self.logger.debug(f"Step {episode_steps}: portfolio_return={portfolio_return:.6f}, cumulative_return={np.prod(1 + np.array(self.episode_returns)) - 1:.4f}")
                self.episode_actions.append(action.copy())
                
                # 거래 비용 추적
                transaction_cost = info.get('transaction_cost', 0)
                self.all_costs.append(transaction_cost)
                
                # Store experience
                from src.core.replay import Transition
                transition = Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                self.replay_buffer.push(transition)
                
                # Store in memory cell
                self.memory_cell.store(
                    state, action, reward, crisis_level,
                    gating_decision.selected_bcell,
                    {'episode': episode, 'step': episode_steps}
                )
                
                # Update B-Cell
                if len(self.replay_buffer) > self.config.sac_batch_size:
                    transitions, indices, weights = self.replay_buffer.sample(self.config.sac_batch_size)
                    
                    # Convert transitions to batch format
                    states = torch.FloatTensor([t.state for t in transitions]).to(self.device)
                    actions = torch.FloatTensor([t.action for t in transitions]).to(self.device)
                    rewards = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
                    next_states = torch.FloatTensor([t.next_state for t in transitions]).to(self.device)
                    dones = torch.FloatTensor([t.done for t in transitions]).to(self.device)
                    
                    # 배치 생성 - 텐서 그대로 유지
                    batch = {
                        'states': states,  # 이미 텐서
                        'actions': actions,  # 이미 텐서
                        'rewards': rewards,  # 이미 텐서
                        'next_states': next_states,  # 이미 텐서
                        'dones': dones,  # 이미 텐서
                        'weights': torch.FloatTensor(weights).to(self.device),
                        'indices': indices
                    }
                    
                    losses = self.b_cell.update(batch)
                    
                    # Monitor stability
                    stability_metrics = {
                        'q_value': losses.get('q_value', 0.0),
                        'entropy': losses.get('entropy', 1.0),
                        'loss': losses.get('critic_loss', 0.0),
                        'reward': reward,
                        'gradient_norm': losses.get('grad_norm', 0.0),
                        'learning_rate': self.config.sac_lr,
                        'cql_alpha': losses.get('cql_alpha', 0.0),
                        'portfolio_concentration': np.max(action),
                        'turnover': np.linalg.norm(action - self.last_action)
                    }
                    
                    # 액션 업데이트
                    self.last_action = action.copy()
                    
                    # Push metrics to stability monitor
                    self.stability_monitor.push(stability_metrics)
                    
                    # Check for intervention
                    alerts = self.stability_monitor.check()
                    if alerts['severity'] in ('warning', 'critical'):
                        self.logger.warning(f"{alerts['severity'].upper()} stability alert: {alerts['issues']}")
                        
                        # 즉시 개입
                        self.stability_monitor.intervene(self)
                        
                        # 알람 스냅샷 시각화 저장
                        alert_timestamp = f"{self.global_step}"
                        
                        # Equity curve 저장 (에피소드 수익률로부터 생성)
                        if hasattr(self, 'episode_returns'):
                            equity_curve = np.cumprod(1 + np.array(self.episode_returns))
                            plot_equity_curve(
                                equity_curve,
                                save_path=self.run_dir / "alerts" / f"equity_{alert_timestamp}.png"
                            )
                            plot_drawdown(
                                equity_curve,
                                save_path=self.run_dir / "alerts" / f"dd_{alert_timestamp}.png"
                            )
                        
                        # Portfolio weights 저장
                        if hasattr(self, 'episode_actions') and len(self.episode_actions) > 0:
                            asset_names = [f"Asset_{i}" for i in range(len(action))]
                            latest_weights = self.episode_actions[-1]
                            plot_portfolio_weights(
                                latest_weights,
                                asset_names,
                                save_path=self.run_dir / "alerts" / f"weights_{alert_timestamp}.png"
                            )
                        
                        self.logger.info(f"알람 시각화 저장: {self.run_dir / 'alerts'}")
                    
                    # Update priorities
                    td_errors = losses.get('td_error', None)
                    if td_errors is not None:
                        indices = batch.get('indices', None)
                        if indices is not None:
                            self.replay_buffer.update_priorities(
                                indices, td_errors.cpu().numpy()
                            )
                
                # Update gating network performance
                self.gating_network.update_performance(
                    gating_decision.selected_bcell,
                    reward,
                    {'crisis_level': crisis_level}
                )
                
                # Accumulate
                episode_reward += reward
                episode_steps += 1
                state = next_state
                self.global_step += 1
            
            episode_rewards.append(episode_reward)
            
            # Calculate episode metrics
            if len(self.episode_returns) > 0:
                episode_sharpe = self._calculate_sharpe(self.episode_returns)
                episode_cvar = self._calculate_cvar(self.episode_returns)
                episode_calmar = self._calculate_calmar(self.episode_returns)
                episode_sortino = self._calculate_sortino(self.episode_returns)
                episode_sharpes.append(episode_sharpe)
                episode_cvars.append(episode_cvar)
                
                # Calculate portfolio metrics
                returns_array = np.array(self.episode_returns)
                portfolio_value = self.config.env_config['initial_balance'] * np.prod(1 + returns_array)
                total_return = np.prod(1 + returns_array) - 1
                volatility = np.std(returns_array) * np.sqrt(252)
                
                # 최대 낙폭 계산
                equity_curve = np.cumprod(1 + returns_array)
                running_max = np.maximum.accumulate(equity_curve)
                drawdown = (equity_curve - running_max) / running_max
                max_drawdown = np.min(drawdown)
                
                # 회전율 계산 (액션 변화량)
                if hasattr(self, 'episode_actions') and len(self.episode_actions) > 1:
                    turnovers = [np.sum(np.abs(self.episode_actions[i] - self.episode_actions[i-1])) 
                                for i in range(1, len(self.episode_actions))]
                    avg_turnover = np.mean(turnovers) if turnovers else 0
                else:
                    avg_turnover = 0
                
                # Update progress bar
                pbar.set_postfix({
                    'Return': f"{total_return:.2%}",
                    'Sharpe': f"{episode_sharpe:.2f}",
                    'Calmar': f"{episode_calmar:.2f}",
                    'Value': f"{portfolio_value/1e6:.2f}M",
                    'Steps': episode_steps
                })
                
                # 매 에피소드 종료 시 상세 성과 출력
                self.logger.info("=" * 60)
                self.logger.info(f"Episode {episode+1}/{self.config.sac_episodes} 완료")
                self.logger.info("-" * 60)
                self.logger.info(f"📊 수익률: {total_return:.6%} | 포트폴리오: ${portfolio_value:,.2f}")
                self.logger.info(f"📈 Sharpe: {episode_sharpe:.3f} | Calmar: {episode_calmar:.3f} | Sortino: {episode_sortino:.3f}")
                self.logger.info(f"📉 CVaR(5%): {episode_cvar:.6f} | MaxDD: {max_drawdown:.6%} | Vol: {volatility:.4%}")
                self.logger.info(f"🔄 Turnover: {avg_turnover:.4%} | Steps: {episode_steps} | Reward: {episode_reward:.6f}")
                
                # 디버그 정보 추가
                self.logger.debug(f"Raw portfolio value: {portfolio_value}")
                self.logger.debug(f"Transaction costs: {np.mean(self.all_costs[-episode_steps:]) if hasattr(self, 'all_costs') and len(self.all_costs) > 0 else 0:.6f}")
                self.logger.debug(f"Action std: {np.std(action) if 'action' in locals() else 0:.6f}")
                self.logger.info("=" * 60)
            
            # 10 에피소드마다 통계 요약
            if (episode + 1) % 10 == 0 and len(episode_rewards) >= 10:
                # 최근 10 에피소드 통계
                recent_returns = []
                for i in range(max(0, episode - 9), episode + 1):
                    if i < len(episode_sharpes):
                        recent_returns.append(episode_sharpes[i])
                
                self.logger.info("\n" + "="*60)
                self.logger.info("📊 최근 10 에피소드 통계:")
                self.logger.info(f"  평균 Sharpe: {np.mean(episode_sharpes[-10:]):.3f}")
                self.logger.info(f"  평균 보상: {np.mean(episode_rewards[-10:]):.4f}")
                self.logger.info(f"  최고 보상: {np.max(episode_rewards[-10:]):.4f}")
                self.logger.info(f"  최저 보상: {np.min(episode_rewards[-10:]):.4f}")
                self.logger.info(f"  평균 CVaR: {np.mean(episode_cvars[-10:]) if episode_cvars else 0:.3f}")
                self.logger.info("=" * 60 + "\n")
            
            # Evaluation
            if (episode + 1) % self.config.eval_interval == 0:
                eval_metrics = self._evaluate()
                self._check_early_stopping(eval_metrics)
            
            # Checkpoint
            if (episode + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(f"episode_{episode+1}")
    
    def _evaluate(self) -> Dict[str, float]:
        """모델 평가"""
        self.logger.info("평가 시작...")
        
        eval_rewards = []
        eval_returns = []
        eval_actions = []
        
        for _ in range(10):  # 10 episodes evaluation
            episode_reward = 0
            episode_returns = []
            
            state, _ = self.env.reset()
            done = False
            
            while not done:
                # Deterministic action
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Get crisis and memory guidance
                crisis_info = self.t_cell.detect_crisis(self.env.get_market_data())
                memory_guidance = self.memory_cell.get_memory_guidance(
                    state, crisis_info['overall_crisis']
                )
                
                # Gating decision
                gating_decision = self.gating_network(
                    state_tensor, memory_guidance, crisis_info['overall_crisis']
                )
                
                # Select action
                action = self.b_cell.select_action(
                    state_tensor,
                    bcell_type=gating_decision.selected_bcell,
                    deterministic=True
                )
                
                eval_actions.append(action)
                
                # Step
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_returns.append(reward)
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_returns.extend(episode_returns)
        
        # Calculate metrics
        returns_array = np.array(eval_returns)
        
        # Sharpe Ratio
        if len(returns_array) > 1:
            sharpe = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0
        
        # CVaR (5%)
        sorted_returns = np.sort(returns_array)
        cvar_5 = np.mean(sorted_returns[:max(1, len(sorted_returns) // 20)])
        
        # Max Drawdown
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        metrics = {
            'avg_reward': np.mean(eval_rewards),
            'sharpe_ratio': sharpe,
            'cvar_5': cvar_5,
            'max_drawdown': max_drawdown,
            'avg_return': np.mean(returns_array),
            'return_std': np.std(returns_array)
        }
        
        # Log metrics
        self.logger.info(
            f"평가 결과: Sharpe={sharpe:.3f}, CVaR={cvar_5:.3f}, "
            f"MaxDD={max_drawdown:.3f}, AvgReward={metrics['avg_reward']:.2f}"
        )
        
        self.logger.log_metrics(metrics, self.global_step)
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """샤프 비율 계산"""
        if len(returns) < 20:  # 최소 샘플 수
            return 0.0
        returns_array = np.array(returns)
        if np.std(returns_array) < 1e-8:
            return 0.0
        sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        # 디버그 로그
        self.logger.debug(f"Sharpe calculation: mean={np.mean(returns_array):.6f}, std={np.std(returns_array):.6f}, sharpe={sharpe:.3f}")
        return sharpe
    
    def _calculate_cvar(self, returns: List[float], alpha: float = 0.05) -> float:
        """CVaR 계산"""
        if len(returns) < 20:  # 최소 샘플 수
            return 0.0
        returns_array = np.array(returns)
        sorted_returns = np.sort(returns_array)
        n_tail = max(1, int(len(sorted_returns) * alpha))
        cvar = np.mean(sorted_returns[:n_tail])
        self.logger.debug(f"CVaR calculation: n_tail={n_tail}, cvar={cvar:.6f}")
        return cvar
    
    def _calculate_calmar(self, returns: List[float]) -> float:
        """칼마 비율 계산 (연간 수익률 / 최대 낙폭)"""
        if len(returns) < 20:
            return 0.0
        equity_curve = np.cumprod(1 + np.array(returns))
        
        # 연환산 수익률
        total_return = equity_curve[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # 최대 낙폭 계산
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_dd = abs(np.min(drawdown))
        
        if max_dd < 1e-8:
            return 0.0
        return annual_return / max_dd
    
    def _calculate_sortino(self, returns: List[float], target_return: float = 0.0) -> float:
        """소르티노 비율 계산 (하방 변동성만 고려)"""
        if len(returns) < 20:
            return 0.0
        returns_array = np.array(returns)
        excess_returns = returns_array - target_return
        
        # 하방 변동성
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        if downside_std < 1e-8:
            return 0.0
        
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    def _check_early_stopping(self, metrics: Dict[str, float]):
        """조기 종료 확인"""
        sharpe = metrics.get('sharpe_ratio', 0)
        
        # Check if target met
        if sharpe >= self.config.target_sharpe and \
           metrics.get('cvar_5', 0) >= self.config.target_cvar:
            self.logger.info("목표 달성! 학습 종료.")
            self._save_checkpoint("target_achieved")
            return True
        
        # Check improvement
        if sharpe > self.best_sharpe + self.config.min_improvement:
            self.best_sharpe = sharpe
            self.patience_counter = 0
            self._save_checkpoint("best")
            self.logger.info(f"새로운 최고 Sharpe: {sharpe:.3f}")
        else:
            self.patience_counter += 1
            
        # Check patience
        if self.patience_counter >= self.config.patience:
            self.logger.info(f"조기 종료 (patience={self.config.patience})")
            return True
        
        return False
    
    def _save_checkpoint(self, tag: str):
        """체크포인트 저장 (메타데이터 포함)"""
        import datetime
        
        # 메타데이터 수집
        metadata = {
            'checkpoint_type': 'full',  # 'full' or 'iql'
            'timestamp': datetime.datetime.now().isoformat(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'n_assets': self.action_dim,  # 포트폴리오 자산 수
            'framework_version': '2.0',  # BIPD 버전
            'training_mode': 'sac' if self.episode > 0 else 'iql',
            'total_steps': self.global_step,
            'episode': self.episode
        }
        
        checkpoint = {
            'episode': self.episode,
            'global_step': self.global_step,
            'b_cell': self.b_cell.state_dict(),
            'gating_network': self.gating_network.state_dict(),
            'memory_cell': {
                'memories': list(self.memory_cell.memories),
                'stats': self.memory_cell.memory_stats
            },
            't_cell': self.t_cell.get_state(),
            'metrics': self.metrics_history[-1] if self.metrics_history else {},
            'config': self.config.__dict__,
            'stability_report': self.stability_monitor.get_report(),
            'metadata': metadata  # 메타데이터 추가
        }
        
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, path)
        
        # Notify StabilityMonitor about checkpoint
        self.stability_monitor.save_checkpoint(str(path))
        
        self.logger.info(f"체크포인트 저장: {path}")
    
    def load_checkpoint(self, path: str):
        """체크포인트 로드 (호환성 검사 포함)"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 메타데이터 확인
        metadata = checkpoint.get('metadata', {})
        if metadata:
            self.logger.info("체크포인트 메타데이터:")
            self.logger.info(f"  - 타입: {metadata.get('checkpoint_type', 'unknown')}")
            self.logger.info(f"  - 타임스탬프: {metadata.get('timestamp', 'N/A')}")
            self.logger.info(f"  - State 차원: {metadata.get('state_dim', 'N/A')}")
            self.logger.info(f"  - Action 차원: {metadata.get('action_dim', 'N/A')}")
            self.logger.info(f"  - 자산 수: {metadata.get('n_assets', 'N/A')}")
            self.logger.info(f"  - 학습 모드: {metadata.get('training_mode', 'N/A')}")
            
            # 호환성 검사
            if 'state_dim' in metadata and metadata['state_dim'] != self.state_dim:
                self.logger.warning(
                    f"State 차원 불일치: 체크포인트={metadata['state_dim']}, 현재={self.state_dim}"
                )
                self.logger.warning("모델 아키텍처가 변경되었을 수 있습니다. 계속 진행합니다.")
            
            if 'action_dim' in metadata and metadata['action_dim'] != self.action_dim:
                self.logger.warning(
                    f"Action 차원 불일치: 체크포인트={metadata['action_dim']}, 현재={self.action_dim}"
                )
                self.logger.warning("자산 수가 변경되었을 수 있습니다. 로드를 중단합니다.")
                raise ValueError("체크포인트와 현재 환경의 자산 수가 일치하지 않습니다.")
        
        # IQL 체크포인트인지 full 체크포인트인지 확인
        checkpoint_type = metadata.get('checkpoint_type', None)
        if checkpoint_type == 'iql':
            is_iql_checkpoint = True
        elif checkpoint_type == 'full':
            is_iql_checkpoint = False
        else:
            # 레거시 체크포인트 (메타데이터 없음) - 휴리스틱으로 판단
            is_iql_checkpoint = 'actor' in checkpoint and 'episode' not in checkpoint
        
        if is_iql_checkpoint:
            # IQL 체크포인트 로드
            self.logger.info("IQL 체크포인트 감지 - IQL 가중치만 로드합니다")
            
            # episode와 global_step은 0으로 초기화
            self.episode = 0
            self.global_step = checkpoint.get('training_steps', 0)
            
            # 모든 B-Cell에 IQL 가중치 로드
            for bcell_name, bcell in self.b_cells.items():
                if hasattr(bcell, 'load_iql_checkpoint'):
                    bcell.load_iql_checkpoint(checkpoint)
                    self.logger.info(f"B-Cell [{bcell_name}]에 IQL 체크포인트 로드 완료")
                else:
                    # load_iql_checkpoint가 없으면 직접 actor만 로드
                    if 'actor' in checkpoint:
                        bcell.actor.load_state_dict(checkpoint['actor'])
                        self.logger.info(f"B-Cell [{bcell_name}]에 IQL actor 가중치 로드 완료")
            
            # 기본 B-Cell도 업데이트
            if hasattr(self.b_cell, 'load_iql_checkpoint'):
                self.b_cell.load_iql_checkpoint(checkpoint)
            
            self.logger.info(f"IQL 체크포인트 로드 완료: {path}")
            self.logger.info("SAC 파인튜닝을 시작할 준비가 되었습니다")
            
        else:
            # Full 체크포인트 로드
            self.episode = checkpoint['episode']
            self.global_step = checkpoint['global_step']
            
            self.b_cell.load_state_dict(checkpoint['b_cell'])
            self.gating_network.load_state_dict(checkpoint['gating_network'])
            
            # Load memory cell
            if 'memory_cell' in checkpoint:
                memory_data = checkpoint['memory_cell']
                self.memory_cell.memories = memory_data['memories']
                self.memory_cell.memory_stats = memory_data['stats']
            
            # Load T-Cell state
            if 't_cell' in checkpoint:
                self.t_cell.load_state(checkpoint['t_cell'])
            
            self.logger.info(f"Full 체크포인트 로드: {path}")
    
    def _generate_report(self, final_metrics: Dict[str, float]):
        """최종 보고서 생성"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config.__dict__,
            'final_metrics': final_metrics,
            'training_history': self.metrics_history,
            'component_stats': {
                'gating': self.gating_network.get_statistics(),
                'memory': self.memory_cell.get_statistics(),
                't_cell': self.t_cell.get_statistics(),
                'b_cell': self.b_cell.get_statistics()
            }
        }
        
        report_path = self.session_dir / "reports" / "training_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"학습 보고서 저장: {report_path}")


def main():
    """메인 실행 함수"""
    # Create config
    config = TrainingConfig()
    
    # Override with command line args if needed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iql-epochs', type=int, default=100)
    parser.add_argument('--sac-episodes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    
    config.iql_epochs = args.iql_epochs
    config.sac_episodes = args.sac_episodes
    config.seed = args.seed
    config.device = args.device
    
    # Create trainer
    trainer = FinFlowTrainer(config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()