# src/core/trainer.py

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import time
import random
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import yaml
from safetensors.torch import save_file, load_file

from src.core.env import PortfolioEnv
from src.core.replay import PrioritizedReplayBuffer
from src.core.offline_dataset import OfflineDataset
from src.core.iql import IQLAgent
# from src.core.sac import DistributionalSAC  # 사용 시 import
from src.agents.b_cell import BCell
from src.agents.t_cell import TCell
from src.agents.memory import MemoryCell
from src.agents.gating import GatingNetwork
from src.analysis.monitor import PerformanceMonitor
from src.analysis.visualization import plot_equity_curve, plot_drawdown, plot_portfolio_weights
from src.utils.monitoring import StabilityMonitor
from src.utils.logger import FinFlowLogger, get_session_directory
from src.utils.seed import set_seed, get_device_info
from src.data.loader import DataLoader
from src.data.features import FeatureExtractor
from src.core.objectives import PortfolioObjective, RewardNormalizer


@dataclass
class TrainingConfig:
    """학습 설정"""
    # Environment
    env_config: Dict = field(default_factory=lambda: {
        'initial_balance': 1000000,
        'transaction_cost': 0.001,
        'max_leverage': 1.0,
        'window_size': 20
    })

    # Data
    data_config: Dict = field(default_factory=lambda: {
        'tickers': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
        'period': '5y',
        'interval': '1d',
        'auto_download': True,
        'use_cache': True
    })

    # Data validation
    data_validation_config: Optional[Dict] = None

    # IQL Pretraining
    iql_epochs: int = 50
    iql_batch_size: int = 256
    iql_expectile: float = 0.7
    iql_temperature: float = 3.0

    # SAC Training
    sac_episodes: int = 1000
    sac_batch_size: int = 256
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_alpha: float = 0.2
    sac_cql_weight: float = 1.0

    # Memory & Replay
    memory_capacity: int = 10000
    memory_k_neighbors: int = 5
    replay_buffer_size: int = 100000

    # Monitoring & Logging
    eval_interval: int = 10
    checkpoint_interval: int = 50
    log_interval: int = 1
    use_tensorboard: bool = True
    use_wandb: bool = False

    # Target Performance
    target_sharpe: float = 1.5
    target_cvar: float = -0.02

    # Device & Seed
    device: str = 'auto'
    seed: int = 42

    # Paths
    data_path: str = 'data/processed'
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'

    # Config loading parameters (not stored as instance variables)
    config_path: Optional[str] = field(default=None, repr=False)
    override_params: Optional[Dict] = field(default=None, repr=False)

    def __post_init__(self):
        """config_path와 override_params가 제공된 경우 설정을 로드하고 오버라이드 적용"""
        if self.config_path is not None:
            self._load_from_config_file()
        if self.override_params is not None:
            self._apply_overrides()

    def _load_from_config_file(self):
        """YAML 설정 파일에서 값 로드"""
        import yaml

        config_path = Path(self.config_path)
        if not config_path.exists():
            return

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Apply config values to instance
        self._apply_config_dict(config_dict)

    def _apply_config_dict(self, config_dict: Dict):
        """설정 딕셔너리를 인스턴스 변수에 적용"""
        # Environment config
        if 'env' in config_dict:
            env = config_dict['env']
            if 'initial_capital' in env:
                self.env_config['initial_balance'] = env['initial_capital']
            if 'turnover_cost' in env:
                self.env_config['transaction_cost'] = env['turnover_cost']
            if 'max_leverage' in env:
                self.env_config['max_leverage'] = env['max_leverage']

        # Data config
        if 'data' in config_dict:
            data = config_dict['data']
            if 'symbols' in data:
                self.data_config['tickers'] = data['symbols']
            if 'interval' in data:
                self.data_config['interval'] = data['interval']

        # Training config
        if 'train' in config_dict:
            train = config_dict['train']
            if 'offline_steps' in train:
                self.iql_epochs = train['offline_steps'] // 1000
            if 'online_steps' in train:
                self.sac_episodes = train['online_steps'] // 100
            if 'offline_batch_size' in train:
                self.iql_batch_size = train['offline_batch_size']
            if 'online_batch_size' in train:
                self.sac_batch_size = train['online_batch_size']

        # BCell config
        if 'bcell' in config_dict:
            bcell = config_dict['bcell']
            if 'iql_expectile' in bcell:
                self.iql_expectile = bcell['iql_expectile']
            if 'iql_temperature' in bcell:
                self.iql_temperature = bcell['iql_temperature']
            if 'gamma' in bcell:
                self.sac_gamma = bcell['gamma']
            if 'tau' in bcell:
                self.sac_tau = bcell['tau']
            if 'alpha_init' in bcell:
                self.sac_alpha = bcell['alpha_init']
            if 'cql_alpha_start' in bcell:
                self.sac_cql_weight = bcell['cql_alpha_start']

        # Memory config
        if 'memory' in config_dict:
            memory = config_dict['memory']
            if 'capacity' in memory:
                self.memory_capacity = memory['capacity']
            if 'k_neighbors' in memory:
                self.memory_k_neighbors = memory['k_neighbors']

        # Training intervals
        if 'train' in config_dict:
            train = config_dict['train']
            if 'eval_interval' in train:
                self.eval_interval = max(1, train['eval_interval'] // 100)
            if 'save_interval' in train:
                self.checkpoint_interval = max(1, train['save_interval'] // 100)
            if 'log_interval' in train:
                self.log_interval = max(1, train['log_interval'] // 100)

        # Target metrics
        if 'objectives' in config_dict:
            obj = config_dict['objectives']
            if 'sharpe_beta' in obj:
                self.target_sharpe = obj['sharpe_beta'] * 1.5
            if 'cvar_target' in obj:
                self.target_cvar = obj['cvar_target']

        # Data validation config
        if 'data_validation' in config_dict:
            self.data_validation_config = config_dict['data_validation']

        # System config
        if 'device' in config_dict:
            self.device = config_dict['device']
        if 'seed' in config_dict:
            self.seed = config_dict['seed']

    def _apply_overrides(self):
        """override_params에서 지정된 값들을 오버라이드"""
        overrides = self.override_params

        # Direct field overrides
        for field_name in ['iql_epochs', 'sac_episodes', 'iql_batch_size', 'sac_batch_size',
                          'memory_capacity', 'target_sharpe', 'target_cvar', 'device', 'seed',
                          'data_path', 'checkpoint_dir']:
            if field_name in overrides:
                setattr(self, field_name, overrides[field_name])

        # Learning rate overrides
        if 'iql_lr' in overrides:
            # IQL learning rate은 현재 TrainingConfig에 저장되지 않음
            pass
        if 'sac_lr' in overrides:
            # SAC learning rate은 현재 TrainingConfig에 저장되지 않음
            pass

        # Nested config overrides
        if 'env_config' in overrides:
            self.env_config.update(overrides['env_config'])
        if 'data_config' in overrides:
            self.data_config.update(overrides['data_config'])


class FinFlowTrainer:
    """FinFlow 통합 학습 시스템"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = FinFlowLogger("FinFlowTrainer")

        # Set seed for reproducibility
        set_seed(config.seed)

        # Device setup
        self.device = self._setup_device()
        self.logger.info(f"Using device: {self.device}")

        # Create directories
        # get_session_directory()가 이미 "logs/timestamp" 형태로 반환하므로 직접 사용
        self.log_dir = Path(get_session_directory())
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # checkpoint는 세션별 models 디렉토리에 저장
        self.checkpoint_dir = self.log_dir / "models"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self._load_data()

        # Initialize components
        self._initialize_components()

        # Tracking
        self.episode = 0
        self.global_step = 0
        self.best_sharpe = -np.inf
        self.metrics_history = []

        self.logger.info("FinFlow Trainer 초기화 완료")

    def _setup_device(self) -> torch.device:
        """디바이스 설정"""
        if self.config.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(self.config.device)

    def _load_data(self):
        """데이터 로드 및 전처리"""
        self.logger.info("데이터 로드 중...")

        # DataLoader 초기화
        data_loader = DataLoader(
            cache_dir="data/cache",
            validation_config=self.config.data_validation_config
        )

        # 데이터 다운로드
        if self.config.data_config.get('auto_download', True):
            tickers = self.config.data_config['tickers']

            # 기간 설정 (YAML 설정에서 start/end 사용)
            if 'start' in self.config.data_config and 'end' in self.config.data_config:
                start_date = self.config.data_config['start']
                end_date = self.config.data_config['end']
            else:
                # period를 날짜로 변환 (기본값)
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime('%Y-%m-%d')
                period = self.config.data_config.get('period', '5y')
                if period == '5y':
                    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
                elif period == '2y':
                    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
                else:
                    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

            self.price_data = data_loader.download_data(
                symbols=tickers,
                start_date=start_date,
                end_date=end_date,
                use_cache=self.config.data_config.get('use_cache', True)
            )
        else:
            # 저장된 데이터 로드
            data_path = Path(self.config.data_path)
            if (data_path / 'prices.csv').exists():
                self.price_data = pd.read_csv(data_path / 'prices.csv', index_col=0, parse_dates=True)
            else:
                raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path / 'prices.csv'}")

        # Train/Test split (80/20)
        split_idx = int(len(self.price_data) * 0.8)
        self.train_data = self.price_data.iloc[:split_idx]
        self.test_data = self.price_data.iloc[split_idx:]

        self.logger.info(f"데이터 로드 완료: {len(self.price_data)} 일, {len(self.price_data.columns)} 자산")
        self.logger.info(f"학습 데이터: {len(self.train_data)} 일, 테스트 데이터: {len(self.test_data)} 일")

        # Feature Extractor 초기화
        self.feature_extractor = FeatureExtractor(
            window=self.config.env_config.get('window_size', 20)
        )

    def _initialize_components(self):
        """모든 컴포넌트 초기화"""
        # Dimensions
        n_assets = len(self.price_data.columns)

        # feature_extractor에서 차원 정보 가져오기
        if hasattr(self.feature_extractor, 'total_dim'):
            feature_dim = self.feature_extractor.total_dim
        else:
            # 기본값 사용
            feature_dim = 12  # default from config

        self.state_dim = feature_dim + n_assets + 1  # features + weights + crisis
        self.action_dim = n_assets

        self.logger.info(f"State dimension: {self.state_dim}, Action dimension: {self.action_dim}")

        # Environment
        self.env = PortfolioEnv(
            price_data=self.train_data,
            feature_extractor=self.feature_extractor,
            initial_capital=self.config.env_config.get('initial_balance', 1000000),
            transaction_cost=self.config.env_config.get('transaction_cost', 0.001),
            max_leverage=self.config.env_config.get('max_leverage', 1.0)
        )

        self.test_env = PortfolioEnv(
            price_data=self.test_data,
            feature_extractor=self.feature_extractor,
            initial_capital=self.config.env_config.get('initial_balance', 1000000),
            transaction_cost=self.config.env_config.get('transaction_cost', 0.001),
            max_leverage=self.config.env_config.get('max_leverage', 1.0)
        )

        # Agents
        self._initialize_agents()

        # Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.replay_buffer_size,
            alpha=0.6,
            beta=0.4
        )

        # Performance Monitor
        self.performance_monitor = PerformanceMonitor(
            log_dir=str(self.log_dir),
            use_tensorboard=self.config.use_tensorboard,
            use_wandb=self.config.use_wandb
        )

        # Stability Monitor
        stability_config = {
            'window_size': 100,
            'n_sigma': 3.0,
            'intervention_threshold': 5,
            'rollback_enabled': True,
            'max_weight_change': 0.3,
            'min_effective_assets': 3,
            'log_interval': 100
        }
        self.stability_monitor = StabilityMonitor(stability_config)

        # Objective Function
        objective_config = {
            'sharpe_ema_alpha': 0.99,
            'sharpe_epsilon': 1e-8,
            'cvar_alpha': 0.05,
            'cvar_target': -0.02,
            'lambda_cvar': 1.0,
            'lambda_turn': 0.1,
            'lambda_dd': 0.0,
            'r_clip': 5.0
        }
        self.objective = PortfolioObjective(objective_config)

        # Reward Normalizer
        self.reward_normalizer = RewardNormalizer()

    def _initialize_agents(self):
        """에이전트들 초기화"""
        # T-Cell (Crisis Detection)
        feature_config = {
            'dimensions': {
                'returns': 3,
                'technical': 4,
                'structure': 3,
                'momentum': 2
            },
            'total_dim': 12
        }

        self.t_cell = TCell(
            feature_dim=self.state_dim - self.action_dim - 1,  # features only
            contamination=0.1,
            n_estimators=100,
            feature_config=feature_config
        )

        # B-Cell configuration
        bcell_config = {
            'gamma': self.config.sac_gamma,
            'tau': self.config.sac_tau,
            'alpha_init': self.config.sac_alpha,
            'cql_alpha_start': self.config.sac_cql_weight,
            'actor_hidden': [256, 256],
            'critic_hidden': [256, 256],
            'n_quantiles': 32,
            'actor_lr': 3e-4,
            'critic_lr': 3e-4,
            'alpha_lr': 3e-4
        }

        # Main B-Cell
        self.b_cell = BCell(
            specialization='defensive',  # Default specialization
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=bcell_config,
            device=self.device
        )

        # Specialized B-Cells
        self.b_cells = {}
        for specialization in ['volatility', 'correlation', 'momentum', 'defensive', 'growth']:
            self.b_cells[specialization] = BCell(
                specialization=specialization,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                config=bcell_config,
                device=self.device
            )

        # Memory Cell
        self.memory_cell = MemoryCell(
            capacity=self.config.memory_capacity,
            k_neighbors=self.config.memory_k_neighbors
        )

        # Gating Network
        self.gating_network = GatingNetwork(
            state_dim=self.state_dim,
            num_experts=len(self.b_cells),
            hidden_dim=128,
            temperature=1.0
        ).to(self.device)

        # GatingNetwork Optimizer (추가: expert 선택 학습)
        self.gating_optimizer = torch.optim.Adam(
            self.gating_network.parameters(),
            lr=3e-4
        )

    def train(self):
        """전체 학습 파이프라인"""
        self.logger.info("=" * 80)
        self.logger.info("FinFlow 학습 시작")
        self.logger.info("=" * 80)

        # Phase 1: Offline Pretraining with IQL
        if self.config.iql_epochs > 0:
            self.logger.info("\n[Phase 1] IQL 오프라인 사전학습")
            self._pretrain_iql()

        # Phase 2: Online Fine-tuning with SAC
        if self.config.sac_episodes > 0:
            self.logger.info("\n[Phase 2] SAC 온라인 파인튜닝")
            self._train_sac()

        # Final evaluation
        self._final_evaluation()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("학습 완료!")
        self.logger.info(f"최고 샤프 비율: {self.best_sharpe:.4f}")
        self.logger.info("=" * 80)

    def _pretrain_iql(self):
        """IQL 오프라인 사전학습"""
        self.logger.info("오프라인 데이터 수집 중...")

        # Collect offline dataset
        offline_dataset = OfflineDataset()

        # Use the built-in collect_from_env method
        n_episodes = min(50, self.config.iql_epochs * 10)  # Collect more episodes but with fewer epochs
        offline_dataset.collect_from_env(
            env=self.env,
            n_episodes=n_episodes,
            diversity_bonus=True,
            verbose=True
        )

        self.logger.info(f"오프라인 데이터셋 크기: {len(offline_dataset)}")

        # Initialize IQL agent
        iql_agent = IQLAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            expectile=self.config.iql_expectile,
            temperature=self.config.iql_temperature,
            discount=self.config.sac_gamma,
            tau=self.config.sac_tau,
            learning_rate=3e-4,
            device=self.device
        )

        # Train IQL
        self.logger.info("IQL 학습 시작...")
        for epoch in range(self.config.iql_epochs):
            epoch_losses = []

            for _ in range(1000):  # Steps per epoch
                batch = offline_dataset.get_batch(self.config.iql_batch_size, device=self.device)
                losses = iql_agent.update(**batch)
                epoch_losses.append(losses)

            # Compute average losses
            avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}

            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}/{self.config.iql_epochs} - "
                    f"Value Loss: {avg_losses['value_loss']:.4f}, "
                    f"Q Loss: {avg_losses['q_loss']:.4f}, "
                    f"Actor Loss: {avg_losses['actor_loss']:.4f}"
                )

        # Transfer learned weights to B-Cells
        self.logger.info("IQL 가중치를 B-Cell로 전송...")
        for bcell in self.b_cells.values():
            bcell.actor.load_state_dict(iql_agent.actor.state_dict())

        # Save IQL checkpoint
        iql_checkpoint_path = self.checkpoint_dir / "iql_pretrained.pt"
        iql_agent.save(str(iql_checkpoint_path))
        self.logger.info(f"IQL 체크포인트 저장: {iql_checkpoint_path}")

    def _train_sac(self):
        """SAC 온라인 학습"""
        self.logger.info("SAC 온라인 학습 시작...")

        # T-Cell training with normal data
        self._train_tcell()

        # Main training loop
        for episode in range(self.config.sac_episodes):
            self.episode = episode
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            # Episode metrics
            episode_metrics = {
                'returns': [],
                'actions': [],
                'q_values': [],
                'entropies': []
            }

            done = False
            prev_weights = np.ones(self.action_dim) / self.action_dim  # Initialize with equal weights
            while not done:
                # Get crisis level from T-Cell
                market_data = self.env.get_market_data()
                crisis_info = self.t_cell.detect_crisis(market_data)
                self.env.set_crisis_level(crisis_info['overall_crisis'])

                # Get memory guidance
                memory_guidance = self.memory_cell.get_memory_guidance(
                    state=state,
                    crisis_level=crisis_info['overall_crisis']
                )

                # Select B-Cell using gating network
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                gating_decision = self.gating_network(
                    state_tensor,
                    memory_guidance=memory_guidance,
                    crisis_level=crisis_info['overall_crisis']
                )
                selected_bcell = gating_decision.selected_bcell

                # Get action from selected B-Cell
                action, action_info = self.b_cells[selected_bcell].get_action(state)

                # Environment step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Calculate shaped reward using the objective function
                returns_tensor = torch.tensor([reward], dtype=torch.float32)
                weights_tensor = torch.tensor([action], dtype=torch.float32)
                prev_weights_tensor = torch.tensor([prev_weights], dtype=torch.float32)

                shaped_reward_tensor, objective_metrics = self.objective(
                    returns=returns_tensor,
                    weights=weights_tensor,
                    prev_weights=prev_weights_tensor
                )
                shaped_reward = shaped_reward_tensor.item()

                # Update previous weights for next iteration
                prev_weights = action.copy()

                # Normalize reward
                normalized_reward = self.reward_normalizer.normalize(shaped_reward)

                # Store experience (Soft MoE: 모든 B-Cell에 가중치 비례 저장)
                for bcell_name, bcell in self.b_cells.items():
                    # Soft MoE: 모든 expert가 가중치에 비례하여 학습
                    # 선택된 B-Cell은 더 높은 가중치로 학습
                    weight = gating_decision.weights.get(bcell_name, 0.0)
                    if weight > 0.01:  # 너무 작은 가중치는 무시 (효율성)
                        # 가중치를 보상에 반영하여 저장
                        weighted_reward = normalized_reward * weight
                        bcell.store_experience(state, action, weighted_reward, next_state, done)

                # Store in replay buffer
                from src.core.replay import Transition
                transition = Transition(
                    state=state,
                    action=action,
                    reward=normalized_reward,
                    next_state=next_state,
                    done=done,
                    info=info
                )
                self.replay_buffer.push(transition)

                # Store in memory cell
                self.memory_cell.store(
                    state=state,
                    action=action,
                    reward=normalized_reward,
                    crisis_level=crisis_info['overall_crisis'],
                    bcell_type=selected_bcell,
                    additional_info=info
                )

                # Update metrics
                episode_reward += reward
                episode_steps += 1
                self.global_step += 1

                episode_metrics['returns'].append(reward)
                episode_metrics['actions'].append(action)
                episode_metrics['q_values'].append(action_info.get('q_value', 0))
                episode_metrics['entropies'].append(action_info.get('entropy', 0))

                # Update B-Cells
                if len(self.replay_buffer) >= self.config.sac_batch_size:
                    for bcell_name, bcell in self.b_cells.items():
                        update_losses = bcell.update(batch_size=self.config.sac_batch_size)

                        # Stability check - commented out for now (method doesn't exist)
                        # if update_losses:
                        #     self.stability_monitor.check_training_stability(
                        #         q_values=update_losses.get('td_error', 0),
                        #         actor_loss=update_losses.get('actor_loss', 0),
                        #         critic_loss=update_losses.get('critic_loss', 0),
                        #         step=self.global_step
                        #     )

                # Update gating network (간단한 버전: 성과 기반 업데이트)
                # TODO: gradient-based 업데이트를 위해 GatingNetwork.forward() 수정 필요
                self.gating_network.update_performance(
                    bcell_type=selected_bcell,
                    reward=normalized_reward
                )

                # Performance tracking update
                if self.global_step % 100 == 0:
                    self._update_gating_network()

                state = next_state

            # Episode complete
            episode_sharpe = self._compute_episode_metrics(episode_metrics)

            # Update best
            if episode_sharpe > self.best_sharpe:
                self.best_sharpe = episode_sharpe
                self._save_checkpoint("best")

            # Logging
            if episode % self.config.log_interval == 0:
                self.logger.info(
                    f"Episode {episode}/{self.config.sac_episodes} - "
                    f"Reward: {episode_reward:.4f}, "
                    f"Steps: {episode_steps}, "
                    f"Sharpe: {episode_sharpe:.4f}, "
                    f"Best Sharpe: {self.best_sharpe:.4f}"
                )

            # Evaluation
            if episode % self.config.eval_interval == 0:
                eval_metrics = self._evaluate()
                self.logger.info(f"Evaluation: {eval_metrics}")

            # Checkpoint
            if episode % self.config.checkpoint_interval == 0:
                self._save_checkpoint(f"episode_{episode}")

            # Track metrics
            self.metrics_history.append({
                'episode': episode,
                'reward': episode_reward,
                'sharpe': episode_sharpe,
                'steps': episode_steps,
                **episode_metrics
            })

    def _train_tcell(self):
        """T-Cell 학습"""
        self.logger.info("T-Cell 학습 중...")

        # Collect normal market features
        features = []
        for i in range(self.feature_extractor.window, len(self.train_data)):
            feature = self.feature_extractor.extract_features(
                self.train_data,
                current_idx=i
            )
            features.append(feature)

        features_array = np.array(features)
        self.t_cell.fit(features_array)

        # T-Cell 학습 데이터 캐싱 (체크포인트 저장용)
        self.t_cell_training_data = features_array

        self.logger.info("T-Cell 학습 완료")

    def _update_gating_network(self):
        """Gating network 업데이트"""
        # 각 B-Cell의 성과를 기반으로 gating network 업데이트
        for bcell_name, bcell in self.b_cells.items():
            performance = bcell.performance_score
            self.gating_network.update_performance(bcell_name, performance)

    def _compute_episode_metrics(self, metrics: Dict) -> float:
        """에피소드 메트릭 계산"""
        returns = np.array(metrics['returns'])
        if len(returns) > 1:
            sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)
        else:
            sharpe = 0.0
        return sharpe

    def _evaluate(self) -> Dict:
        """테스트 환경에서 평가"""
        state, _ = self.test_env.reset()
        total_reward = 0
        returns = []

        done = False
        while not done:
            # Get crisis level
            market_data = self.test_env.get_market_data()
            crisis_info = self.t_cell.detect_crisis(market_data)
            self.test_env.set_crisis_level(crisis_info['overall_crisis'])

            # Select B-Cell
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Memory guidance (평가에서도 실제 메모리 활용)
            memory_guidance = self.memory_cell.get_memory_guidance(
                state=state,
                crisis_level=crisis_info['overall_crisis']
            )

            gating_decision = self.gating_network(
                state_tensor,
                memory_guidance=memory_guidance,
                crisis_level=crisis_info['overall_crisis']
            )
            selected_bcell = gating_decision.selected_bcell

            # Get action (deterministic)
            action, _ = self.b_cells[selected_bcell].get_action(state, deterministic=True)

            # Step
            next_state, reward, terminated, truncated, info = self.test_env.step(action)
            done = terminated or truncated

            total_reward += reward
            returns.append(reward)
            state = next_state

        # Compute metrics
        returns_array = np.array(returns)
        sharpe = np.sqrt(252) * returns_array.mean() / (returns_array.std() + 1e-8) if len(returns) > 1 else 0

        return {
            'total_return': total_reward,
            'sharpe_ratio': sharpe,
            'max_drawdown': self._compute_max_drawdown(returns_array)
        }

    def _compute_max_drawdown(self, returns: np.ndarray) -> float:
        """최대 낙폭 계산"""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown) if len(drawdown) > 0 else 0

    def _convert_numpy_types(self, obj):
        """numpy 타입을 Python 네이티브 타입으로 재귀적 변환"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _save_checkpoint(self, tag: str):
        """체크포인트 저장 (SafeTensors 사용)"""
        from datetime import datetime

        # 체크포인트 디렉토리 생성
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{tag}"
        checkpoint_path.mkdir(exist_ok=True)

        # 1. 모델 가중치를 safetensors로 저장
        model_tensors = {}

        # B-Cell state_dict
        b_cell_dict = self.b_cell.state_dict()
        for key, value in b_cell_dict.items():
            if key in ['actor', 'critic_q1', 'critic_q2']:
                # 중첩된 state_dict 처리
                for param_name, param_value in value.items():
                    if isinstance(param_value, torch.Tensor):
                        model_tensors[f"b_cell.{key}.{param_name}"] = param_value
            elif isinstance(value, torch.Tensor):
                model_tensors[f"b_cell.{key}"] = value

        # Gating Network
        for key, value in self.gating_network.state_dict().items():
            if isinstance(value, torch.Tensor):
                model_tensors[f"gating_network.{key}"] = value

        # Specialized B-Cells
        for bcell_name, bcell in self.b_cells.items():
            bcell_dict = bcell.state_dict()
            for key, value in bcell_dict.items():
                if key in ['actor', 'critic_q1', 'critic_q2']:
                    for param_name, param_value in value.items():
                        if isinstance(param_value, torch.Tensor):
                            model_tensors[f"b_cells.{bcell_name}.{key}.{param_name}"] = param_value
                elif isinstance(value, torch.Tensor):
                    model_tensors[f"b_cells.{bcell_name}.{key}"] = value

        # 모델 가중치 저장
        save_file(model_tensors, checkpoint_path / "model.safetensors")

        # 2. Replay Buffer를 numpy 배열로 저장
        if hasattr(self, 'replay_buffer') and len(self.replay_buffer) > 0:
            max_samples = min(10000, len(self.replay_buffer))

            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

            for i in range(max_samples):
                transition = self.replay_buffer.buffer[i]
                states.append(transition.state if isinstance(transition.state, np.ndarray) else np.array(transition.state))
                actions.append(transition.action if isinstance(transition.action, np.ndarray) else np.array(transition.action))
                rewards.append(transition.reward)
                next_states.append(transition.next_state if isinstance(transition.next_state, np.ndarray) else np.array(transition.next_state))
                dones.append(transition.done)

            np.savez_compressed(
                checkpoint_path / "replay_buffer.npz",
                states=np.array(states),
                actions=np.array(actions),
                rewards=np.array(rewards),
                next_states=np.array(next_states),
                dones=np.array(dones),
                size=len(self.replay_buffer),
                ptr=getattr(self.replay_buffer, 'ptr', 0)
            )

        # 3. 메타데이터를 JSON으로 저장
        metadata = {
            'version': '3.0',  # SafeTensors 버전
            'timestamp': datetime.now().isoformat(),
            'episode': int(self.episode),
            'global_step': int(self.global_step),
            'best_sharpe': float(self.best_sharpe),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'device': str(self.device),
            'checkpoint_type': 'full'
        }

        # B-Cell 메타데이터
        b_cell_meta = {}
        for key, value in self.b_cell.state_dict().items():
            if key not in ['actor', 'critic_q1', 'critic_q2'] and not isinstance(value, torch.Tensor):
                b_cell_meta[key] = value
        metadata['b_cell_meta'] = b_cell_meta

        # Specialized B-Cells 메타데이터
        specialized_meta = {}
        for bcell_name, bcell in self.b_cells.items():
            bcell_meta = {}
            for key, value in bcell.state_dict().items():
                if key not in ['actor', 'critic_q1', 'critic_q2'] and not isinstance(value, torch.Tensor):
                    bcell_meta[key] = value
            specialized_meta[bcell_name] = bcell_meta
        metadata['specialized_b_cells_meta'] = specialized_meta

        # T-Cell 상태
        metadata['t_cell'] = self.t_cell.get_state()

        # T-Cell 학습 데이터 샘플 저장 (최대 1000개)
        if hasattr(self, 't_cell_training_data') and self.t_cell_training_data is not None:
            max_samples = min(1000, len(self.t_cell_training_data))
            t_cell_samples = self.t_cell_training_data[:max_samples]
            np.savez_compressed(
                checkpoint_path / "t_cell_training_data.npz",
                features=t_cell_samples
            )
            metadata['t_cell']['has_training_data'] = True
            self.logger.info(f"T-Cell 학습 데이터 저장: {max_samples} 샘플")
        else:
            metadata['t_cell']['has_training_data'] = False

        # Memory Cell 통계
        memory_stats = {}
        if hasattr(self.memory_cell, 'memories') and self.memory_cell.memories:
            memory_stats['size'] = len(self.memory_cell.memories)
            memory_stats['capacity'] = self.memory_cell.capacity
        metadata['memory_stats'] = memory_stats

        # 최근 메트릭 (numpy 타입을 Python 타입으로 변환)
        if self.metrics_history:
            recent_metrics = self.metrics_history[-1]
            # numpy 타입을 Python 타입으로 변환
            converted_metrics = {}
            for k, v in recent_metrics.items():
                if isinstance(v, (np.floating, np.integer)):
                    converted_metrics[k] = float(v)
                elif isinstance(v, np.ndarray):
                    converted_metrics[k] = v.tolist()
                elif isinstance(v, list) and v and isinstance(v[0], (np.floating, np.integer, np.ndarray)):
                    # 리스트 내부의 numpy 타입 변환
                    converted_metrics[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x.tolist() if isinstance(x, np.ndarray) else x for x in v]
                else:
                    converted_metrics[k] = v
            metadata['recent_metrics'] = converted_metrics

        # Config (직렬화 가능한 형태로)
        config_dict = {}
        for key in dir(self.config):
            if not key.startswith('_'):
                value = getattr(self.config, key)
                if not callable(value):
                    # 직접 저장 (필요시 _convert_numpy_types가 처리)
                    config_dict[key] = value
        metadata['config'] = config_dict

        # numpy 타입을 Python 타입으로 변환
        metadata = self._convert_numpy_types(metadata)

        # JSON으로 저장
        with open(checkpoint_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Notify StabilityMonitor
        self.stability_monitor.save_checkpoint(str(checkpoint_path))

        self.logger.info(f"체크포인트 저장 완료 (SafeTensors): {checkpoint_path}")

    def load_checkpoint(self, path: str):
        """체크포인트 로드 (SafeTensors와 기존 형식 모두 지원)"""
        checkpoint_path = Path(path)

        # SafeTensors 형식 체크
        if checkpoint_path.is_dir() and (checkpoint_path / "model.safetensors").exists():
            self._load_safetensors_checkpoint(checkpoint_path)
        else:
            # SafeTensors가 아닌 경우 오류
            raise ValueError(f"SafeTensors 형식이 아닙니다: {path}")

    def _load_safetensors_checkpoint(self, checkpoint_path: Path):
        """SafeTensors 형식 체크포인트 로드"""
        self.logger.info(f"SafeTensors 체크포인트 로드: {checkpoint_path}")

        # 1. 메타데이터 로드
        with open(checkpoint_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        self.episode = metadata['episode']
        self.global_step = metadata['global_step']
        self.best_sharpe = metadata['best_sharpe']

        # 호환성 체크
        if metadata['state_dim'] != self.state_dim:
            self.logger.warning(f"State dimension mismatch: {metadata['state_dim']} vs {self.state_dim}")
        if metadata['action_dim'] != self.action_dim:
            self.logger.warning(f"Action dimension mismatch: {metadata['action_dim']} vs {self.action_dim}")

        # 2. 모델 가중치 로드
        model_tensors = load_file(checkpoint_path / "model.safetensors")

        # B-Cell 로드
        b_cell_state = {'actor': {}, 'critic_q1': {}, 'critic_q2': {}}
        for key, value in model_tensors.items():
            if key.startswith("b_cell."):
                parts = key.replace("b_cell.", "").split(".", 1)
                if len(parts) == 2 and parts[0] in ['actor', 'critic_q1', 'critic_q2']:
                    b_cell_state[parts[0]][parts[1]] = value

        # 메타데이터 추가
        if 'b_cell_meta' in metadata:
            b_cell_state.update(metadata['b_cell_meta'])

        self.b_cell.load_state_dict(b_cell_state)

        # Gating Network 로드
        gating_state = {}
        for key, value in model_tensors.items():
            if key.startswith("gating_network."):
                param_name = key.replace("gating_network.", "")
                gating_state[param_name] = value
        self.gating_network.load_state_dict(gating_state)

        # Specialized B-Cells 로드
        for bcell_name in self.b_cells:
            bcell_state = {'actor': {}, 'critic_q1': {}, 'critic_q2': {}}
            for key, value in model_tensors.items():
                if key.startswith(f"b_cells.{bcell_name}."):
                    parts = key.replace(f"b_cells.{bcell_name}.", "").split(".", 1)
                    if len(parts) == 2 and parts[0] in ['actor', 'critic_q1', 'critic_q2']:
                        bcell_state[parts[0]][parts[1]] = value

            # 메타데이터 추가
            if 'specialized_b_cells_meta' in metadata and bcell_name in metadata['specialized_b_cells_meta']:
                bcell_state.update(metadata['specialized_b_cells_meta'][bcell_name])

            if bcell_state['actor']:  # 데이터가 있을 때만 로드
                self.b_cells[bcell_name].load_state_dict(bcell_state)

        # 3. Replay Buffer 로드
        if (checkpoint_path / "replay_buffer.npz").exists():
            buffer_data = np.load(checkpoint_path / "replay_buffer.npz")

            # Replay buffer 재구성
            from src.core.replay import Transition
            for i in range(len(buffer_data['states'])):
                transition = Transition(
                    state=buffer_data['states'][i],
                    action=buffer_data['actions'][i],
                    reward=buffer_data['rewards'][i],
                    next_state=buffer_data['next_states'][i],
                    done=buffer_data['dones'][i]
                )
                self.replay_buffer.push(transition)

            self.logger.info(f"Replay buffer 로드: {len(buffer_data['states'])} transitions")

        # 4. T-Cell 상태 로드 (학습 데이터 포함)
        if 't_cell' in metadata:
            # T-Cell 학습 데이터 로드
            t_cell_training_data = None
            t_cell_data_path = checkpoint_path / "t_cell_training_data.npz"
            if t_cell_data_path.exists():
                t_cell_data = np.load(t_cell_data_path)
                t_cell_training_data = t_cell_data['features']
                self.logger.info(f"T-Cell 학습 데이터 로드: {t_cell_training_data.shape}")

            # T-Cell 상태 로드 (학습 데이터와 함께)
            if t_cell_training_data is not None:
                self.t_cell.load_state(metadata['t_cell'], training_data=t_cell_training_data)
            else:
                self.logger.warning("T-Cell 학습 데이터 없음 - 재학습 필요")
                self.t_cell.load_state(metadata['t_cell'])

        self.logger.info("SafeTensors 체크포인트 로드 완료")

    def _final_evaluation(self):
        """최종 평가 및 시각화"""
        self.logger.info("\n최종 평가 중...")

        # Test environment evaluation
        final_metrics = self._evaluate()

        self.logger.info("\n최종 성과:")
        self.logger.info(f"  - 총 수익률: {final_metrics['total_return']:.2%}")
        self.logger.info(f"  - 샤프 비율: {final_metrics['sharpe_ratio']:.4f}")
        self.logger.info(f"  - 최대 낙폭: {final_metrics['max_drawdown']:.2%}")

        # Save final checkpoint
        self._save_checkpoint("final")

        # Generate visualizations
        if self.metrics_history:
            self._generate_visualizations()

    def _generate_visualizations(self):
        """학습 결과 시각화"""
        self.logger.info("시각화 생성 중...")

        # Extract data
        episodes = [m['episode'] for m in self.metrics_history]
        rewards = [m['reward'] for m in self.metrics_history]
        sharpes = [m['sharpe'] for m in self.metrics_history]

        # Plot training curves
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Reward curve
        axes[0].plot(episodes, rewards)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Episode Reward')
        axes[0].set_title('Training Reward Progress')
        axes[0].grid(True)

        # Sharpe ratio curve
        axes[1].plot(episodes, sharpes)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Sharpe Ratio')
        axes[1].set_title('Sharpe Ratio Progress')
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_progress.png')
        plt.close()

        self.logger.info(f"시각화 저장: {self.log_dir / 'training_progress.png'}")