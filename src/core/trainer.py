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

@dataclass
class TrainingConfig:
    """학습 설정 - YAML 파일 기반"""
    
    # 필수 파라미터
    config_path: str = "configs/default.yaml"
    override_params: Optional[Dict] = None
    
    # YAML에서 로드될 속성들 (초기값은 None)
    env_config: Dict = field(default_factory=dict)
    data_config: Dict = field(default_factory=dict)
    train_config: Dict = field(default_factory=dict)
    
    # 개별 속성들 (YAML에서 로드됨)
    offline_episodes: int = 500
    offline_steps: int = 200000
    offline_batch_size: int = 512
    offline_eval_interval: int = 10000
    
    iql_epochs: int = 100
    iql_batch_size: int = 256
    iql_lr: float = 3e-4
    iql_expectile: float = 0.7
    iql_temperature: float = 3.0
    
    sac_episodes: int = 1000
    sac_batch_size: int = 256
    sac_lr: float = 3e-4
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_alpha: float = 0.2
    sac_cql_weight: float = 1.0
    
    memory_capacity: int = 50000
    memory_k_neighbors: int = 5
    
    eval_interval: int = 10
    checkpoint_interval: int = 50
    log_interval: int = 1
    
    device: str = "auto"
    seed: int = 42
    
    data_path: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    
    target_sharpe: float = 1.5
    target_cvar: float = -0.02
    
    patience: int = 50
    min_improvement: float = 0.01
    
    monitoring_config: Optional[Dict] = None
    
    def __post_init__(self):
        """YAML 파일에서 설정 자동 로드"""
        # YAML 파일 로드
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # 설정 매핑
            self._load_from_yaml(yaml_config)
        
        # CLI 오버라이드 적용
        if self.override_params:
            self._apply_overrides(self.override_params)
            
        # 로거 생성 (설정 로드 후)
        self.logger = FinFlowLogger("TrainingConfig")
        self.logger.info(f"설정 로드 완료: {self.config_path}")
        self.logger.info(f"  오프라인 에피소드: {self.offline_episodes}")
        self.logger.info(f"  IQL 에폭: {self.iql_epochs}")
        self.logger.info(f"  SAC 에피소드: {self.sac_episodes}")
    
    def _load_from_yaml(self, config: Dict):
        """YAML 설정을 속성으로 변환"""
        # 환경 설정
        env = config.get('env', {})
        self.env_config = {
            'initial_balance': env.get('initial_capital', 1000000),
            'turnover_cost': env.get('turnover_cost', 0.001),
            'slip_coeff': env.get('slip_coeff', 0.0005),
            'max_weight': env.get('max_weight', 0.2),
            'min_weight': env.get('min_weight', 0.0),
            'window_size': config.get('features', {}).get('window', 30),
            'max_weight_change': env.get('max_turnover', 0.5)
        }
        
        # 데이터 설정
        data = config.get('data', {})
        self.data_config = {
            'tickers': data.get('symbols'),
            'symbols': data.get('symbols'),  # 호환성을 위해 둘 다 저장
            'start': data.get('start', '2008-01-01'),
            'end': data.get('end', '2020-12-31'),
            'test_start': data.get('test_start', '2021-01-01'),
            'test_end': data.get('test_end', '2024-12-31'),
            'cache_dir': data.get('cache_dir', 'data/cache'),
            'interval': data.get('interval', '1d'),
            'auto_download': True,
            'use_cache': True
        }
        
        # 학습 설정
        train = config.get('train', {})
        self.offline_episodes = train.get('offline_episodes', 500)
        self.offline_steps = train.get('offline_steps', 200000)
        self.offline_batch_size = train.get('offline_batch_size', 512)
        self.offline_eval_interval = train.get('offline_eval_interval', 10000)
        
        self.train_config = {
            'offline_episodes': self.offline_episodes,
            'offline_steps': self.offline_steps,
            'offline_batch_size': self.offline_batch_size,
            'offline_eval_interval': self.offline_eval_interval
        }
        
        # IQL 설정
        bcell = config.get('bcell', {})
        self.iql_expectile = bcell.get('iql_expectile', 0.7)
        self.iql_temperature = bcell.get('iql_temperature', 3.0)
        self.iql_lr = bcell.get('critic_lr', 3e-4)
        self.iql_batch_size = train.get('offline_batch_size', 256)
        self.iql_epochs = train.get('offline_steps', 200000) // 1000  # steps를 epochs로 변환
        
        # SAC 설정
        self.sac_lr = bcell.get('actor_lr', 3e-4)
        self.sac_gamma = bcell.get('gamma', 0.99)
        self.sac_tau = bcell.get('tau', 0.005)
        self.sac_alpha = bcell.get('alpha_init', 0.2)
        self.sac_cql_weight = bcell.get('cql_alpha_start', 0.01)
        self.sac_batch_size = train.get('online_batch_size', 256)
        self.sac_episodes = train.get('online_steps', 300000) // 300  # steps를 episodes로 변환
        
        # Memory 설정
        memory = config.get('memory', {})
        self.memory_capacity = train.get('buffer_size', 100000)
        self.memory_k_neighbors = memory.get('k_neighbors', 5)
        
        # 평가 설정
        self.eval_interval = train.get('eval_interval', 5000) // 500  # steps를 episodes로 변환
        self.checkpoint_interval = train.get('save_interval', 20000) // 400
        self.log_interval = train.get('log_interval', 100) // 100
        
        # 시스템 설정
        system = config.get('system', {})
        self.device = system.get('device', config.get('device', 'auto'))
        self.seed = system.get('seed', config.get('seed', 42))
        self.data_path = system.get('data_path', 'data/processed')
        self.checkpoint_dir = system.get('checkpoint_dir', 'checkpoints')
        
        # 목표 지표
        objectives = config.get('objectives', {})
        self.target_sharpe = objectives.get('sharpe_target', 1.5)
        self.target_cvar = objectives.get('cvar_target', -0.02)
        
        # 조기 종료
        self.patience = train.get('early_stop_patience', 50000) // 1000
        self.min_improvement = train.get('early_stop_min_delta', 0.001)
        
        # 모니터링 설정
        self.monitoring_config = config.get('monitoring', {})
        
        # 전체 config 저장
        self._raw_config = config
    
    def _apply_overrides(self, overrides: Dict):
        """CLI 인자로 설정 오버라이드"""
        for key, value in overrides.items():
            if value is not None:
                if hasattr(self, key):
                    setattr(self, key, value)
                    if hasattr(self, 'logger'):
                        self.logger.info(f"설정 오버라이드: {key} = {value}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 가져오기 (dict-like 인터페이스)"""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict:
        """설정을 딕셔너리로 변환"""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and k != 'logger'}


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

        # Training state (초기화 순서 중요: _initialize_components 호출 전에 설정)
        self.global_step = 0
        self.episode = 0
        self.best_sharpe = -float('inf')
        self.patience_counter = 0

        # 무체결 감지를 위한 카운터
        self.zero_return_count = 0
        self.zero_return_threshold = 50  # K=50 연속 무체결 시 경고

        # Metrics tracking
        self.metrics_history = []

        # 알람 시각화 쿨다운 설정
        self.last_visualization_step = 0
        self.visualization_cooldown = 1000  # 최소 1000 step 간격

        # 턴오버 단계적 완화 설정 (초기화 순서 중요: get_adaptive_turnover_limit 호출 전에 설정)
        self.turnover_schedule = {
            'warmup_episodes': 20,
            'stabilization_episodes': 50,
            'exploration_episodes': 100
        }

        # 안전 체크포인트 관련 속성 초기화
        self.last_safe_checkpoint = None
        self.failure_count = 0
        self.max_failures = 3

        # 설정에서 safety_stages 로드 (있는 경우)
        self.safety_stages = config.raw_config.get('safety_stages', {
            'warmup': {'max_turnover': 0.3, 'cql_weight': 10.0, 'lr_scale': 0.5, 'min_buffer_size': 256},
            'stabilization': {'max_turnover': 0.5, 'cql_weight': 5.0, 'lr_scale': 0.75, 'min_buffer_size': 512},
            'exploration': {'max_turnover': 0.9, 'cql_weight': 1.0, 'lr_scale': 1.0, 'min_buffer_size': 1000}
        })

        # Initialize components (모든 필수 속성 정의 후에 호출)
        self._initialize_components()

        # 모니터링 설정 완화 (초기 학습 안정화)
        if hasattr(self, 'monitoring'):
            self.monitoring.rollback_on_divergence = False  # 초기에는 rollback 비활성화
            self.monitoring.intervention_threshold = 6.0  # 기준 완화

        self.logger.info("FinFlow Trainer 초기화 완료")

    def get_adaptive_turnover_limit(self, episode: int) -> float:
        """
        Get adaptive turnover limit based on episode number

        Args:
            episode: Current episode number

        Returns:
            Maximum allowed turnover for this episode
        """
        if episode < self.turnover_schedule['warmup_episodes']:
            return 0.3  # Conservative during warmup
        elif episode < self.turnover_schedule['stabilization_episodes']:
            return 0.5  # Moderate during stabilization
        elif episode < self.turnover_schedule['exploration_episodes']:
            return 0.7  # Progressive increase
        else:
            return 0.9  # Final exploration phase

    def get_safety_config(self, episode: int) -> dict:
        """
        Get safety configuration based on episode number

        Args:
            episode: Current episode number

        Returns:
            Safety configuration dictionary
        """
        if episode < self.turnover_schedule['warmup_episodes']:
            return self.safety_stages.get('warmup', {
                'max_turnover': 0.3,
                'cql_weight': 10.0,
                'lr_scale': 0.5,
                'min_buffer_size': 256
            })
        elif episode < self.turnover_schedule['stabilization_episodes']:
            return self.safety_stages.get('stabilization', {
                'max_turnover': 0.5,
                'cql_weight': 5.0,
                'lr_scale': 0.75,
                'min_buffer_size': 512
            })
        else:
            return self.safety_stages.get('exploration', {
                'max_turnover': 0.9,
                'cql_weight': 1.0,
                'lr_scale': 1.0,
                'min_buffer_size': 1000
            })

    def check_network_health(self) -> bool:
        """
        Check network parameters for NaN/Inf values

        Returns:
            True if healthy, False otherwise
        """
        # Return True if networks not initialized yet
        if not hasattr(self, 'b_cell') or self.b_cell is None:
            return True

        # Check if b_cell has required attributes
        if not hasattr(self.b_cell, 'actor') or not hasattr(self.b_cell, 'critic'):
            return True

        import torch
        try:
            # Check actor network
            if hasattr(self.b_cell.actor, 'named_parameters'):
                for name, param in self.b_cell.actor.named_parameters():
                    if param is not None and (torch.isnan(param).any() or torch.isinf(param).any()):
                        self.logger.error(f"NaN/Inf detected in actor {name}")
                        return False

            # Check critic network
            if hasattr(self.b_cell.critic, 'named_parameters'):
                for name, param in self.b_cell.critic.named_parameters():
                    if param is not None and (torch.isnan(param).any() or torch.isinf(param).any()):
                        self.logger.error(f"NaN/Inf detected in critic {name}")
                        return False

            # Check log_alpha if it exists
            if hasattr(self.b_cell, 'log_alpha') and self.b_cell.log_alpha is not None:
                if torch.isnan(self.b_cell.log_alpha).any() or torch.isinf(self.b_cell.log_alpha).any():
                    self.logger.error("NaN/Inf detected in log_alpha")
                    return False

        except Exception as e:
            self.logger.warning(f"Error during network health check: {e}")
            return True  # Assume healthy if check fails

        return True

    def _initialize_components(self):
        """컴포넌트 초기화"""
        # 실제 데이터 로드
        from src.data.loader import DataLoader
        from src.data.features import FeatureExtractor
        # import pandas as pd  # 사용 시 import
        
        self.logger.info("실제 시장 데이터를 로드합니다...")
        
        # DataLoader를 사용하여 실제 데이터 로드
        data_loader = DataLoader(cache_dir="data/cache")
        
        # 설정에서 티커 가져오기 (config에서 읽기)
        tickers = self.config.data_config.get('tickers')
        
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
        
        # Environment 생성 with adaptive turnover
        initial_max_turnover = self.get_adaptive_turnover_limit(0)
        self.env = PortfolioEnv(
            price_data=price_data,
            feature_extractor=self.feature_extractor,
            initial_capital=self.config.env_config.get('initial_capital', 1000000),
            turnover_cost=self.config.env_config.get('turnover_cost', 0.001),
            slip_coeff=self.config.env_config.get('slip_coeff', 0.0005),
            no_trade_band=self.config.env_config.get('no_trade_band', 0.002),
            max_leverage=self.config.env_config.get('max_leverage', 1.0),
            max_turnover=initial_max_turnover
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
            'intervention_threshold': 6.0,  # 가이드 요구값으로 상향
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
        # 초기에는 PER off (uniform sampling)로 시작
        self.use_per = False  # 초기 PER off
        self.per_activation_step = 10000  # 10k 스텝 후 PER 활성화
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.memory_capacity,
            alpha=0.0 if not self.use_per else 0.6,  # 초기 alpha=0 (uniform)
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
            # config에서 에피소드 수 가져오기 (YAML 설정 사용)
            n_episodes = self.config.offline_episodes
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
        dataset = OfflineDataset(self.config.data_path)
        
        self.logger.info(f"오프라인 데이터셋 로드: {len(dataset)} samples")
        self.logger.info("=" * 50)
        self.logger.info("IQL 오프라인 사전학습 시작")
        self.logger.info("=" * 50)
        
        # Training loop with progress bar
        from tqdm import tqdm
        # YAML 설정 사용
        pbar = tqdm(range(self.config.iql_epochs), desc="IQL Pretraining", unit="epoch")
        
        for epoch in pbar:
            epoch_losses = []
            
            # Mini-batch training
            for _ in range(len(dataset) // self.config.iql_batch_size):
                # YAML 설정 사용
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
        
        # T-Cell prefit 강제 수행
        if hasattr(self, 't_cell') and not self.t_cell.is_fitted:
            # 초기 특성 윈도우 준비
            initial_features = []
            for i in range(self.t_cell.window_size):
                feat = self.feature_extractor.extract_features(
                    self.env.price_data,
                    current_idx=self.feature_extractor.window + i
                )
                initial_features.append(feat)
            initial_features = np.array(initial_features)
            self.t_cell.prefit(initial_features)
            self.logger.info("T-Cell prefit 완료")

        from tqdm import tqdm
        pbar = tqdm(range(self.config.sac_episodes), desc="SAC Training", unit="episode")

        for episode in pbar:
            self.episode = episode
            episode_reward = 0
            episode_steps = 0
            self.episode_returns = []  # 에피소드 수익률 추적
            self.episode_actions = []  # 에피소드 액션 추적

            # Update environment with adaptive turnover limit
            adaptive_turnover = self.get_adaptive_turnover_limit(episode)
            self.env.max_turnover = adaptive_turnover

            # Reset environment
            state, _ = self.env.reset()
            done = False
            
            # Episode loop
            max_episode_steps = min(self.env.max_steps, len(self.env.price_data) - 2)
            last_progress_checkpoint = 0

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

                # 에피소드 10% 진행 시점마다 통계 출력 (과도한 로그 방지)
                progress_percentage = (episode_steps / max_episode_steps) * 100 if max_episode_steps > 0 else 0
                current_checkpoint = int(progress_percentage // 10) * 10  # 10, 20, 30, ...

                if current_checkpoint > last_progress_checkpoint and current_checkpoint > 0:
                    last_progress_checkpoint = current_checkpoint

                    # 최근 수익률 통계 계산
                    window_size = max(10, episode_steps // 10)  # 최소 10 스텝, 아니면 전체의 10%
                    recent_returns = self.episode_returns[-window_size:] if len(self.episode_returns) >= window_size else self.episode_returns
                    cumulative_return = np.prod(1 + np.array(self.episode_returns)) - 1

                    self.logger.debug(
                        f"진행률 {current_checkpoint}% (Step {episode_steps}/{max_episode_steps}) | "
                        f"최근 {len(recent_returns)}스텝 통계: "
                        f"평균={np.mean(recent_returns)*100:.2f}%, "
                        f"표준편차={np.std(recent_returns)*100:.2f}%, "
                        f"최대={np.max(recent_returns)*100:.2f}%, "
                        f"최소={np.min(recent_returns)*100:.2f}% | "
                        f"누적수익률={(cumulative_return)*100:.2f}%"
                    )

                self.episode_actions.append(action.copy())
                
                # 거래 비용 추적
                transaction_cost = info.get('transaction_cost', 0)
                self.all_costs.append(transaction_cost)
                
                # CVaR 페널티 적용 (최근 수익률 기반)
                if len(self.episode_returns) >= 20:
                    recent_returns = np.array(self.episode_returns[-20:])
                    cvar_alpha = 0.95  # 하위 5%
                    var_idx = int(len(recent_returns) * (1 - cvar_alpha))
                    if var_idx > 0:
                        sorted_returns = np.sort(recent_returns)
                        cvar = np.mean(sorted_returns[:var_idx])
                        cvar_target = -0.02  # -2% 목표
                        cvar_penalty = max(0, cvar_target - cvar) * 10.0  # 강한 페널티
                        reward = reward - cvar_penalty * 0.1  # 보상에 CVaR 페널티 반영

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
                
                # 무체결 감지 로직
                if abs(portfolio_return) < 1e-12:
                    self.zero_return_count += 1
                else:
                    self.zero_return_count = 0

                if self.zero_return_count >= self.zero_return_threshold:
                    self.logger.error(f"{self.zero_return_threshold}회 연속 무거래 감지!")
                    self._diagnose_no_trade()
                    assert False, f"무거래 루프 감지: {self.zero_return_threshold}회 연속 0 수익"

                # PER 활성화 체크
                if not self.use_per and self.global_step >= self.per_activation_step:
                    self.use_per = True
                    self.replay_buffer.alpha = 0.6  # PER 활성화
                    self.logger.info(f"PER 활성화 (step {self.global_step})")

                # Update B-Cell (adaptive min buffer size from safety config)
                safety_config = self.get_safety_config(episode)
                min_buffer_size = safety_config.get('min_buffer_size', 256)
                if len(self.replay_buffer) > min_buffer_size:
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
                    
                    # Check network health before update
                    if not self.check_network_health():
                        self.logger.error("Network unhealthy, restoring checkpoint")
                        if hasattr(self, 'last_safe_checkpoint'):
                            self.load_checkpoint(self.last_safe_checkpoint)
                        break  # Exit episode

                    losses = self.b_cell.update(batch)

                    # Check for NaN in losses
                    if any(not torch.isfinite(torch.tensor(v)) if isinstance(v, float) else False
                           for v in losses.values()):
                        self.logger.error("NaN detected in losses, restoring checkpoint")
                        if hasattr(self, 'last_safe_checkpoint'):
                            self.load_checkpoint(self.last_safe_checkpoint)
                        break  # Exit episode

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
                        
                        # 알람 스냅샷 시각화 저장 (critical만 + 쿨다운 체크)
                        should_visualize = (
                            alerts['severity'] == 'critical' and  # critical만
                            self.global_step - self.last_visualization_step >= self.visualization_cooldown
                        )
                        
                        if should_visualize:
                            self.last_visualization_step = self.global_step
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
                            
                            self.logger.info(f"Critical 알람 시각화 저장: {self.run_dir / 'alerts'}")
                    
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

            # Save checkpoint periodically if network is healthy
            if episode > 0 and episode % 10 == 0 and self.check_network_health():
                self.last_safe_checkpoint = self.save_checkpoint(
                    f"safe_checkpoint_ep{episode}.pt"
                )
                self.logger.info(f"Saved safe checkpoint at episode {episode}")

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
    
    def _diagnose_no_trade(self):
        """무거래 상황 진단"""
        self.logger.error("\n" + "="*60)
        self.logger.error("🔍 무거래 진단 스냅샷")
        self.logger.error("="*60)

        # 현재 포트폴리오 상태
        self.logger.error(f"Portfolio value: {self.env.portfolio_value}")
        self.logger.error(f"Cash: {self.env.cash}")
        self.logger.error(f"Weights: {self.env.weights}")
        self.logger.error(f"Holdings: {getattr(self.env, 'holdings', 'N/A')}")

        # 거래 제약
        self.logger.error(f"No-trade band: {self.env.no_trade_band}")
        self.logger.error(f"Max turnover: {self.env.max_turnover}")
        self.logger.error(f"Min trade size: {getattr(self.env, 'min_trade_size', 1)}")

        # 최근 액션
        if hasattr(self, 'last_action'):
            self.logger.error(f"Last action: {self.last_action}")
            self.logger.error(f"Action L1 norm: {np.sum(np.abs(self.last_action))}")
            self.logger.error(f"Action change from weights: {np.sum(np.abs(self.last_action - self.env.weights))}")

        # 버퍼 상태
        self.logger.error(f"Replay buffer size: {len(self.replay_buffer)}")
        self.logger.error(f"Zero return count: {self.zero_return_count}")
        self.logger.error("="*60 + "\n")

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
    
    def save_checkpoint(self, filename: Optional[str] = None) -> str:
        """
        Save checkpoint with optional filename

        Args:
            filename: Optional checkpoint filename. If None, auto-generates based on episode

        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_ep{self.episode}_step{self.global_step}.pt"

        # Use _save_checkpoint internally with filename as tag
        tag = filename.replace('.pt', '').replace('checkpoint_', '')
        self._save_checkpoint(tag)

        # Return the full path
        return str(self.checkpoint_dir / f"checkpoint_{tag}.pt")

    def _save_checkpoint(self, tag: str):
        """체크포인트 저장 (internal method)"""
        import numpy as np
        import copy
        from datetime import datetime

        # memory_cell 데이터를 완전히 텐서로 변환
        memory_data = []
        for m in list(self.memory_cell.memories):
            memory_item = {}
            for key, value in m.items():
                if isinstance(value, np.ndarray):
                    memory_item[key] = torch.tensor(value, dtype=torch.float32)
                elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], (int, float)):
                    memory_item[key] = torch.tensor(value, dtype=torch.float32)
                elif isinstance(value, (int, float, bool)):
                    memory_item[key] = value
                elif isinstance(value, str):
                    memory_item[key] = value
                elif value is None:
                    memory_item[key] = value
                else:
                    # 다른 타입은 텐서로 변환 시도
                    try:
                        memory_item[key] = torch.tensor(value, dtype=torch.float32)
                    except:
                        memory_item[key] = value
            memory_data.append(memory_item)

        # memory_stats도 안전하게 변환
        memory_stats = {}
        if hasattr(self.memory_cell, 'memory_stats'):
            for key, value in self.memory_cell.memory_stats.items():
                if isinstance(value, np.ndarray):
                    memory_stats[key] = torch.tensor(value, dtype=torch.float32)
                elif isinstance(value, (np.integer, np.floating)):
                    memory_stats[key] = float(value)
                else:
                    memory_stats[key] = value

        # 실제 device 문자열 저장
        device_str = str(self.device)
        if device_str == 'auto' or 'auto' in str(device_str):
            if torch.cuda.is_available():
                device_str = f'cuda:{torch.cuda.current_device()}'
            else:
                device_str = 'cpu'
            self.logger.info(f"device='auto'를 '{device_str}'로 변환하여 저장")

        # config를 직렬화 가능한 형태로 변환
        def make_serializable(obj):
            """객체를 직렬화 가능한 형태로 변환"""
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, (int, float, bool, str)) or obj is None:
                return obj
            else:
                return str(obj)

        # config 전체를 직렬화 가능한 형태로 변환
        config_dict = {}
        for key in dir(self.config):
            if not key.startswith('_') and key != 'logger':
                value = getattr(self.config, key)
                if not callable(value):
                    config_dict[key] = value
        serializable_config = make_serializable(config_dict)

        # metrics도 안전하게 변환
        safe_metrics = {}
        if self.metrics_history:
            for key, value in self.metrics_history[-1].items():
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    safe_metrics[key] = float(value.item() if hasattr(value, 'item') else value)
                elif isinstance(value, (np.integer, np.floating)):
                    safe_metrics[key] = float(value)
                else:
                    safe_metrics[key] = value

        # B-Cell state_dict를 안전하게 변환
        b_cell_state = self.b_cell.state_dict()
        safe_b_cell = {}

        # state_dict들은 그대로 유지
        for key in ['actor', 'critic_q1', 'critic_q2']:
            if key in b_cell_state:
                safe_b_cell[key] = b_cell_state[key]

        # log_alpha numpy array를 텐서로 변환
        if 'log_alpha' in b_cell_state:
            if isinstance(b_cell_state['log_alpha'], np.ndarray):
                safe_b_cell['log_alpha'] = torch.tensor(b_cell_state['log_alpha'], dtype=torch.float32)
            else:
                safe_b_cell['log_alpha'] = b_cell_state['log_alpha']

        # 메타데이터는 그대로 유지
        for key in ['specialization', 'training_step', 'performance_score']:
            if key in b_cell_state:
                safe_b_cell[key] = b_cell_state[key]

        # Optimizer states 저장
        optimizer_states = {
            'actor_optimizer': self.b_cell.actor_optimizer.state_dict(),
            'critic_optimizer': self.b_cell.critic_optimizer.state_dict(),
            'alpha_optimizer': self.b_cell.alpha_optimizer.state_dict()
        }

        # Specialized B-Cells 저장 (있는 경우)
        specialized_b_cells = {}
        if hasattr(self, 'b_cells'):
            for name, bcell in self.b_cells.items():
                bcell_state = bcell.state_dict()
                safe_bcell = {}
                for key in ['actor', 'critic_q1', 'critic_q2']:
                    if key in bcell_state:
                        safe_bcell[key] = bcell_state[key]
                if 'log_alpha' in bcell_state:
                    if isinstance(bcell_state['log_alpha'], np.ndarray):
                        safe_bcell['log_alpha'] = torch.tensor(bcell_state['log_alpha'], dtype=torch.float32)
                    else:
                        safe_bcell['log_alpha'] = bcell_state['log_alpha']
                for key in ['specialization', 'training_step', 'performance_score']:
                    if key in bcell_state:
                        safe_bcell[key] = bcell_state[key]
                specialized_b_cells[name] = safe_bcell

        # Replay Buffer 저장 (크기 제한)
        replay_buffer_data = None
        if hasattr(self, 'replay_buffer') and len(self.replay_buffer) > 0:
            # 최근 10000개 샘플만 저장 (메모리 고려)
            max_samples = min(10000, len(self.replay_buffer))
            buffer_samples = []
            for i in range(max_samples):
                buffer_samples.append(self.replay_buffer.buffer[i])
            replay_buffer_data = {
                'buffer': buffer_samples,
                'priorities': self.replay_buffer.priorities[:max_samples].copy() if hasattr(self.replay_buffer, 'priorities') else None,
                'size': len(self.replay_buffer),
                'ptr': self.replay_buffer.ptr if hasattr(self.replay_buffer, 'ptr') else 0
            }

        # Random states 저장
        random_states = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate() if 'random' in globals() else None
        }

        # Gating Network performance history 저장
        gating_history = None
        if hasattr(self.gating_network, 'performance_history'):
            gating_history = {
                'performance_history': dict(self.gating_network.performance_history),
                'selection_count': dict(self.gating_network.selection_count) if hasattr(self.gating_network, 'selection_count') else {}
            }

        # T-Cell crisis history 저장
        tcell_history = None
        if hasattr(self.t_cell, 'crisis_history'):
            tcell_history = {
                'crisis_history': list(self.t_cell.crisis_history) if hasattr(self.t_cell, 'crisis_history') else [],
                'detection_stats': self.t_cell.detection_stats if hasattr(self.t_cell, 'detection_stats') else {}
            }

        # StabilityMonitor state 저장
        stability_state = None
        if hasattr(self, 'stability_monitor'):
            stability_state = {
                'alert_counts': self.stability_monitor.alert_counts if hasattr(self.stability_monitor, 'alert_counts') else {},
                'interventions': self.stability_monitor.interventions if hasattr(self.stability_monitor, 'interventions') else [],
                'report': self.stability_monitor.get_report()
            }

        checkpoint = {
            'episode': int(self.episode),
            'global_step': int(self.global_step),
            'b_cell': safe_b_cell,
            'gating_network': self.gating_network.state_dict(),
            'memory_cell': {
                'memories': memory_data,
                'stats': memory_stats
            },
            't_cell': self.t_cell.get_state(),
            'metrics': safe_metrics,
            'config': serializable_config,
            'device': device_str,
            'best_sharpe': float(self.best_sharpe),
            'timestamp': datetime.now().isoformat(),
            'stability_report': self.stability_monitor.get_report() if hasattr(self, 'stability_monitor') else None,
            # 새로 추가된 항목들
            'optimizer_states': optimizer_states,
            'specialized_b_cells': specialized_b_cells if specialized_b_cells else None,
            'replay_buffer': replay_buffer_data,
            'random_states': random_states,
            'gating_history': gating_history,
            'tcell_history': tcell_history,
            'stability_state': stability_state,
            'version': '2.0'  # 버전 정보 추가
        }

        # 체크포인트 저장
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, path)

        # Notify StabilityMonitor about checkpoint
        self.stability_monitor.save_checkpoint(str(path))

        self.logger.info(f"체크포인트 저장: {path}")
    
    def load_checkpoint(self, path: str):
        """체크포인트 로드 (호환성 검사 포함)"""
        # Convert to Path object and handle relative paths
        path_obj = Path(path)
        if not path_obj.is_absolute():
            # If relative path, check in checkpoint_dir first
            checkpoint_path = self.checkpoint_dir / path
            if checkpoint_path.exists():
                path = str(checkpoint_path)
            elif path_obj.exists():
                path = str(path_obj)
            else:
                raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
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

            # Load optimizer states (새로 추가)
            if 'optimizer_states' in checkpoint:
                self.b_cell.actor_optimizer.load_state_dict(checkpoint['optimizer_states']['actor_optimizer'])
                self.b_cell.critic_optimizer.load_state_dict(checkpoint['optimizer_states']['critic_optimizer'])
                self.b_cell.alpha_optimizer.load_state_dict(checkpoint['optimizer_states']['alpha_optimizer'])
                self.logger.info("Optimizer states 로드 완료")

            # Load specialized B-Cells (새로 추가)
            if 'specialized_b_cells' in checkpoint and checkpoint['specialized_b_cells']:
                for name, bcell_state in checkpoint['specialized_b_cells'].items():
                    if name in self.b_cells:
                        self.b_cells[name].load_state_dict(bcell_state)
                        self.logger.info(f"Specialized B-Cell [{name}] 로드 완료")

            # Load replay buffer (새로 추가)
            if 'replay_buffer' in checkpoint and checkpoint['replay_buffer']:
                buffer_data = checkpoint['replay_buffer']
                self.replay_buffer.buffer = buffer_data['buffer']
                if 'priorities' in buffer_data and buffer_data['priorities'] is not None:
                    self.replay_buffer.priorities = buffer_data['priorities']
                self.replay_buffer.ptr = buffer_data.get('ptr', 0)
                self.logger.info(f"Replay buffer 로드 완료 (size: {len(buffer_data['buffer'])})")

            # Load random states (새로 추가)
            if 'random_states' in checkpoint:
                torch.set_rng_state(checkpoint['random_states']['torch'])
                np.random.set_state(checkpoint['random_states']['numpy'])
                if checkpoint['random_states']['python']:
                    random.setstate(checkpoint['random_states']['python'])
                self.logger.info("Random states 복원 완료")

            # Load Gating Network history (새로 추가)
            if 'gating_history' in checkpoint and checkpoint['gating_history']:
                if hasattr(self.gating_network, 'performance_history'):
                    self.gating_network.performance_history = defaultdict(list, checkpoint['gating_history']['performance_history'])
                    self.gating_network.selection_count = defaultdict(int, checkpoint['gating_history']['selection_count'])
                    self.logger.info("Gating Network history 로드 완료")

            # Load T-Cell history (새로 추가)
            if 'tcell_history' in checkpoint and checkpoint['tcell_history']:
                if hasattr(self.t_cell, 'crisis_history'):
                    self.t_cell.crisis_history = deque(checkpoint['tcell_history']['crisis_history'], maxlen=100)
                    self.t_cell.detection_stats = checkpoint['tcell_history'].get('detection_stats', {})
                    self.logger.info("T-Cell history 로드 완료")

            # Load StabilityMonitor state (새로 추가)
            if 'stability_state' in checkpoint and checkpoint['stability_state']:
                if hasattr(self, 'stability_monitor'):
                    self.stability_monitor.alert_counts = checkpoint['stability_state'].get('alert_counts', {})
                    self.stability_monitor.interventions = checkpoint['stability_state'].get('interventions', [])
                    self.logger.info("StabilityMonitor state 로드 완료")

            # Load memory cell
            if 'memory_cell' in checkpoint:
                memory_data = checkpoint['memory_cell']
                self.memory_cell.memories = memory_data['memories']
                self.memory_cell.memory_stats = memory_data['stats']

            # Load T-Cell state
            if 't_cell' in checkpoint:
                self.t_cell.load_state(checkpoint['t_cell'])

            # 버전 정보 확인
            version = checkpoint.get('version', '1.0')
            self.logger.info(f"Full 체크포인트 로드 완료: {path} (version: {version})")
    
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