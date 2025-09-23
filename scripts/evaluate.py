# scripts/evaluate.py

import os
# 평가 스크립트에서는 새 로그 디렉토리 생성 방지
os.environ['FINFLOW_NO_FILE_LOG'] = '1'

import numpy as np
import pandas as pd
import torch
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

# Safe globals 등록 (마이그레이션된 체크포인트용)
import torch.serialization
# 기본 타입들은 이미 안전하므로, 필요한 경우에만 추가
# torch.serialization.add_safe_globals([dict, list, tuple])  # 이미 기본으로 허용됨

from src.environments.portfolio_env import PortfolioEnv
from src.algorithms.online.b_cell import BCell
from src.algorithms.online.t_cell import TCell
from src.algorithms.online.memory import MemoryCell
# from src.agents.gating import GatingNetwork  # Not used anymore
from src.evaluation.explainer import XAIExplainer
from src.evaluation.metrics import calculate_sharpe_ratio, calculate_cvar, calculate_max_drawdown
from src.evaluation.visualizer import plot_portfolio_weights, plot_equity_curve, plot_drawdown
from src.evaluation.backtester import RealisticBacktester  # 백테스터 통합
from src.utils.device_manager import set_seed
from src.utils.logger import FinFlowLogger, get_session_directory, set_session_directory
import logging

class FinFlowEvaluator:
    """
    FinFlow 시스템 평가기
    
    백테스팅, 벤치마크 비교, 성능 분석
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 data_path: str,
                 config_path: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Args:
            checkpoint_path: 체크포인트 경로
            data_path: 평가 데이터 경로
            config_path: 설정 파일 경로
            device: 디바이스
        """
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.device = device
        
        # 표준 logging 사용 (새 세션 디렉토리 생성 방지)
        self.logger = logging.getLogger("Evaluator")
        self.logger.setLevel(logging.INFO)

        # 콘솔 핸들러만 추가 (파일 로깅 제외)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
                datefmt="%H:%M:%S"
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.info("평가기 초기화")

        # Set session directory to checkpoint's parent directory for evaluate mode
        # This prevents creating new timestamp directories
        checkpoint_parent = Path(self.checkpoint_path).parent.parent  # logs/20250923_194512/
        set_session_directory(str(checkpoint_parent))

        # Load configuration - YAML 우선, JSON 폴백
        if config_path is None:
            config_path = 'configs/default.yaml'

        if Path(config_path).exists():
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        # Initialize components
        self._load_models()
        self._load_data()

        # Initialize RealisticBacktester
        backtest_config = self.config.get('backtest', None)
        self.backtester = RealisticBacktester(config=backtest_config)

        # Results storage
        self.results = {}
        # 평가 결과를 checkpoint의 상위 logs 폴더 내에 저장
        checkpoint_parent = Path(self.checkpoint_path).parent.parent  # logs/20250923_194512/
        self.session_dir = checkpoint_parent / "evaluation"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.session_dir
        (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)
        self.viz_dir = self.run_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_default_config(self) -> Dict:
        """기본 설정"""
        return {
            'env': {
                'initial_capital': 1000000,
                'turnover_cost': 0.001,
                'max_weight': 0.2,
                'min_weight': 0.0,
                'window_size': 30
            },
            'evaluation': {
                'n_episodes': 10,
                'benchmarks': ['equal_weight', 'market_cap', 'momentum'],
                'metrics': ['sharpe', 'cvar', 'max_drawdown', 'turnover']
            },
            'bcell': {
                'actor_hidden': [256, 256],
                'critic_hidden': [256, 256],
                'n_quantiles': 32,
                'actor_lr': 3e-4,
                'critic_lr': 3e-4
            },
            'gating': {
                'hidden_dim': 128,
                'temperature': 1.0
            },
            'train': {
                'n_experts': 5,
                'feature_window': 20,
                'momentum_lookback': 20,
                'gating_hidden_dim': 128,
                'gating_temperature': 1.0,
                'gating_performance_maxlen': 100
            }
        }
    
    def _load_models(self) -> None:
        """모델과 메모리 셀 로드"""
        checkpoint_path = Path(self.checkpoint_path)

        # 체크포인트 경로 결정
        if checkpoint_path.is_dir():
            # 디렉토리가 주어지면 best.pt를 우선 확인
            best_path = checkpoint_path / 'best.pt'
            if best_path.exists():
                checkpoint_path = best_path
                self.logger.info(f"Best 체크포인트 로드: {checkpoint_path}")
            else:
                # best가 없으면 가장 최근 step_*.pt 파일
                step_files = sorted(checkpoint_path.glob('step_*.pt'),
                                   key=lambda x: int(x.stem.split('_')[1]),
                                   reverse=True)
                if step_files:
                    checkpoint_path = step_files[0]
                    self.logger.info(f"최근 체크포인트 로드: {checkpoint_path}")
                else:
                    # 기존 episode_*.pt 파일 확인 (backward compatibility)
                    episode_files = sorted(checkpoint_path.glob('episode_*.pt'),
                                          key=lambda x: int(x.stem.split('_')[1]),
                                          reverse=True)
                    if episode_files:
                        checkpoint_path = episode_files[0]
                        self.logger.info(f"최근 체크포인트 로드: {checkpoint_path}")
                    else:
                        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        else:
            self.logger.info(f"체크포인트 로드: {checkpoint_path}")

        # SafeTensors 형식 체크
        if checkpoint_path.is_dir() and (checkpoint_path / "model.safetensors").exists():
            checkpoint = self._load_safetensors_checkpoint(checkpoint_path)
        else:
            # 기존 .pt 형식
            checkpoint = self._load_legacy_checkpoint(self.checkpoint_path)

        # 체크포인트 검증
        # 현재 체크포인트는 B-Cell 자체의 state dict
        required_keys = ['actor', 'critics']  # B-Cell의 필수 구성 요소
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            # 이전 형식의 체크포인트일 수 있음
            if 'b_cell' in checkpoint:
                checkpoint = checkpoint['b_cell']
                missing_keys = [k for k in required_keys if k not in checkpoint]
                if missing_keys:
                    raise KeyError(f"체크포인트에 필수 키가 없습니다: {missing_keys}")
            else:
                raise KeyError(f"체크포인트에 필수 키가 없습니다: {missing_keys}")

        # 차원 정보를 config에서 가져오기
        # 체크포인트에는 차원 정보가 없으므로 데이터 설정에서 추론
        data_config = self.config.get('data', {})
        symbols = data_config.get('symbols', data_config.get('tickers', []))

        if symbols:
            n_assets = len(symbols)
            # state_dim = features(12) + weights(n_assets) + crisis(1)
            state_dim = 12 + n_assets + 1
            action_dim = n_assets
            self.logger.info(f"자산 수({n_assets})에서 차원 추론: state_dim={state_dim}, action_dim={action_dim}")
        else:
            # 최종 fallback
            state_dim = 43  # 12 + 30 + 1
            action_dim = 30
            self.logger.warning(f"기본값 사용: state_dim={state_dim}, action_dim={action_dim}")

        self.logger.info(f"체크포인트 차원: state_dim={state_dim}, action_dim={action_dim}")

        # Store action_dim for later validation
        self.expected_action_dim = action_dim

        # Initialize components
        # 현재 체크포인트는 B-Cell의 state dict
        b_cell_data = checkpoint

        # BCell 초기화에 필요한 config (YAML config에서 가져오기)
        bcell_config = self.config.get('bcell', {})

        # REDQ/TQC 통합 설정
        bcell_config_final = {
            'algorithm': bcell_config.get('algorithm', 'REDQ'),
            'gamma': bcell_config.get('gamma', 0.99),
            'tau': bcell_config.get('tau', 0.005),
            'alpha': bcell_config.get('alpha', 0.2),
            'hidden_dims': bcell_config.get('hidden_dims', [256, 256]),
            'actor_lr': float(bcell_config.get('actor_lr', 3e-4)),
            'critic_lr': float(bcell_config.get('critic_lr', 3e-4)),
            'alpha_lr': float(bcell_config.get('alpha_lr', 3e-4)),
            'batch_size': bcell_config.get('batch_size', 256),
            'buffer_size': bcell_config.get('buffer_size', 100000),
            'n_critics': bcell_config.get('n_critics', 10 if bcell_config.get('algorithm', 'REDQ') == 'REDQ' else 2),
            'm_sample': bcell_config.get('m_sample', 2),
            'utd_ratio': bcell_config.get('utd_ratio', 20 if bcell_config.get('algorithm', 'REDQ') == 'REDQ' else 1),
            'n_quantiles': bcell_config.get('n_quantiles', 25),
            'quantile_embedding_dim': bcell_config.get('quantile_embedding_dim', 64),
            'top_quantiles_to_drop_per_net': bcell_config.get('top_quantiles_to_drop_per_net', 2),
            'crisis_threshold': bcell_config.get('crisis_threshold', 0.7)
        }

        self.b_cell = BCell(
            state_dim=state_dim,
            action_dim=action_dim,
            config=bcell_config_final,
            device=self.device
        )
        # B-Cell state 로드
        # B-Cell의 load 메서드를 직접 사용
        self.b_cell.load(str(self.checkpoint_path))

        # GatingNetwork는 더 이상 사용하지 않음
        self.gating_network = None

        self.t_cell = TCell()
        if 't_cell' in checkpoint:
            # T-Cell 학습 데이터 로드 (재학습용)
            t_cell_training_data = None
            if checkpoint['t_cell'].get('has_training_data', False):
                t_cell_data_path = Path(self.checkpoint_path) / "t_cell_training_data.npz"
                if t_cell_data_path.exists():
                    t_cell_data = np.load(t_cell_data_path)
                    t_cell_training_data = t_cell_data['features']
                    self.logger.info(f"T-Cell 학습 데이터 로드: {t_cell_training_data.shape}")
                else:
                    self.logger.warning("T-Cell 학습 데이터 파일이 없음")

            # T-Cell 상태 로드 및 재학습
            if t_cell_training_data is not None:
                self.t_cell.load_state(checkpoint['t_cell'], training_data=t_cell_training_data)
            else:
                # 학습 데이터가 없으면, 평가 데이터를 사용하여 자동 학습 예정
                self.logger.warning("T-Cell 학습 데이터 없음 - 평가 시 자동 학습 필요")
                self.t_cell.load_state(checkpoint['t_cell'])
                # 평가 시작 시 데이터를 모아서 학습할 필요가 있음을 표시
                self.t_cell_needs_fitting = True
        else:
            self.t_cell_needs_fitting = True

        # 메모리 셀 복원
        self.memory_cell = MemoryCell()

        # 메모리 데이터 복원
        if 'memory_cell' in checkpoint:
            memories = checkpoint['memory_cell'].get('memories', [])

            for m in memories:
                memory_item = {}
                for key, value in m.items():
                    if isinstance(value, torch.Tensor):
                        # 텐서를 numpy로 변환 (메모리셀은 numpy 사용)
                        memory_item[key] = value.cpu().numpy()
                    else:
                        memory_item[key] = value

                # 필수 키 검증
                required_memory_keys = ['state', 'action', 'reward']
                if all(k in memory_item for k in required_memory_keys):
                    self.memory_cell.memories.append(memory_item)
                else:
                    self.logger.warning(f"메모리 항목에 필수 키가 누락됨: {set(required_memory_keys) - set(memory_item.keys())}")

            # stats 복원
            if 'stats' in checkpoint['memory_cell']:
                stats = checkpoint['memory_cell']['stats']
                for key, value in stats.items():
                    if isinstance(value, torch.Tensor):
                        stats[key] = value.cpu().numpy()
                self.memory_cell.memory_stats = stats

        # Initialize XAI
        feature_names = self._get_feature_names()

        self.explainer = XAIExplainer(
            model=self.b_cell.actor,
            feature_names=feature_names,
            memory_cell=self.memory_cell
        )

        # 체크포인트 메트릭 저장
        self.checkpoint_metrics = checkpoint.get('metrics', {})
        self.checkpoint_epoch = checkpoint.get('episode', 0)

        self.logger.info(f"모델 로드 완료 (Episode: {self.checkpoint_epoch})")
        self.logger.info(f"메모리 셀 크기: {len(self.memory_cell.memories)}")


    def _load_safetensors_checkpoint(self, checkpoint_path: Path) -> Dict:
        """"""
        from safetensors.torch import load_file
        import json

        self.logger.info(f"SafeTensors 체크포인트 로드: {checkpoint_path}")

        # 메타데이터 로드
        with open(checkpoint_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        # 모델 가중치 로드
        model_tensors = load_file(checkpoint_path / "model.safetensors")

        # 체크포인트 형식으로 변환
        checkpoint = {
            'device': metadata.get('device', 'cpu'),
            'episode': metadata.get('episode', 0),
            'global_step': metadata.get('global_step', 0),
            'state_dim': metadata.get('state_dim'),
            'action_dim': metadata.get('action_dim'),
            'config': metadata.get('config', {}),
            'metrics': metadata.get('recent_metrics', {})
        }

        # B-Cell state 복원
        b_cell_state = {'actor': {}, 'critic_q1': {}, 'critic_q2': {}}
        for key, value in model_tensors.items():
            if key.startswith("b_cell."):
                parts = key.replace("b_cell.", "").split(".", 1)
                if len(parts) == 2 and parts[0] in ['actor', 'critic_q1', 'critic_q2']:
                    b_cell_state[parts[0]][parts[1]] = value
                elif parts[0] == 'log_alpha':
                    b_cell_state['log_alpha'] = value

        # 메타데이터 추가
        if 'b_cell_meta' in metadata:
            b_cell_state.update(metadata['b_cell_meta'])

        checkpoint['b_cell'] = b_cell_state

        # Gating Network state 복원
        gating_state = {}
        for key, value in model_tensors.items():
            if key.startswith("gating_network."):
                param_name = key.replace("gating_network.", "")
                gating_state[param_name] = value
        checkpoint['gating_network'] = gating_state

        # T-Cell state
        if 't_cell' in metadata:
            checkpoint['t_cell'] = metadata['t_cell']

        # Memory cell stats
        if 'memory_stats' in metadata:
            checkpoint['memory_cell'] = {'stats': metadata['memory_stats']}

        return checkpoint

    def _load_legacy_checkpoint(self, checkpoint_path: str) -> Dict:
        """Legacy .pt 체크포인트 로드"""
        # device mapping 함수 정의
        def device_mapper(storage, loc):
            """디바이스 매핑 처리"""
            if isinstance(loc, str):
                if loc == 'auto':
                    # auto는 처리하지 않음 - 오류 발생하도록
                    raise ValueError(f"Invalid device location: {loc}. Run migration script first.")
                elif loc == 'cpu':
                    return storage
                elif loc.startswith('cuda'):
                    # cuda:0 형태 처리
                    if torch.cuda.is_available():
                        return storage.to(loc)  # torch.device 래핑 제거
                    else:
                        return storage
            return storage

        # 체크포인트 로드 (weights_only=True로 안전하게)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device_mapper,
            weights_only=True  # 보안을 위해 True 사용
        )

        return checkpoint
    
    def _load_data(self):
        """데이터 로드"""
        # 데이터 경로가 None이면 동적 로드
        if self.data_path is None:
            self.logger.info("데이터 경로가 지정되지 않음 - DataLoader로 동적 로드")
            self._load_data_dynamically()
            return

        self.logger.info(f"데이터 로드: {self.data_path}")

        # Load price data
        data_file = Path(self.data_path) / "test_data.npz"
        if data_file.exists():
            data = np.load(data_file)
            self.prices = data['prices']
            self.returns = data['returns']
            self.features = data.get('features', None)
        else:
            self.logger.warning(f"테스트 데이터를 찾을 수 없습니다: {data_file}")
            self.logger.info("DataLoader로 동적 로드 시도")
            self._load_data_dynamically()
    
    def _get_feature_names(self) -> List[str]:
        """특징 이름 생성"""
        feature_names = []
        # 체크포인트에서 가져온 action_dim 사용
        n_assets = self.expected_action_dim if hasattr(self, 'expected_action_dim') else 30

        # 실제 FeatureExtractor의 차원과 정확히 일치
        # FeatureExtractor: returns=3, technical=4, structure=3, momentum=2 = 12차원

        # Returns features (3)
        feature_names.extend(['return_mean', 'return_std', 'return_skew'])

        # Technical features (4)
        feature_names.extend(['rsi', 'macd', 'bollinger_upper', 'bollinger_lower'])

        # Structure features (3)
        feature_names.extend(['correlation', 'beta', 'max_drawdown'])

        # Momentum features (2)
        feature_names.extend(['momentum_short', 'momentum_long'])

        # Portfolio weights (n_assets)
        for i in range(n_assets):
            feature_names.append(f'weight_{i}')

        # Crisis level (1)
        feature_names.append('crisis_level')

        # 실제 state dimension에 맞게 반환
        total_features = 12 + n_assets + 1
        return feature_names[:total_features]

    def _load_data_dynamically(self):
        """체크포인트 설정을 사용하여 데이터 동적 로드"""
        from src.data.market_loader import DataLoader
        import pandas as pd

        # YAML 설정에서 심볼과 날짜 정보 가져오기
        data_config = self.config.get('data', {})
        symbols = data_config.get('symbols', [])

        if not symbols:
            raise ValueError(
                "심볼 리스트가 비어있습니다. configs/default.yaml의 data.symbols를 확인하세요."
            )

        # 테스트 기간 설정
        test_start = data_config.get('test_start', '2021-01-01')
        test_end = data_config.get('test_end', '2024-12-31')

        self.logger.info(f"동적 데이터 로드: {len(symbols)}개 자산 (첫 5개: {symbols[:5]}), period={test_start} to {test_end}")

        # DataLoader로 데이터 로드
        validation_config = self.config.get('data_validation', None)
        loader = DataLoader(
            cache_dir='data/cache',
            validation_config=validation_config
        )

        # 데이터 다운로드
        price_data = loader.download_data(
            symbols=symbols,
            start_date=test_start,
            end_date=test_end,
            use_cache=True
        )

        if price_data.empty:
            raise ValueError("데이터 로드 실패: 가격 데이터가 비어있습니다")

        # 가격과 수익률 계산
        self.prices = price_data.values  # (T, N) 형태

        # 수익률 계산
        returns = price_data.pct_change().fillna(0)
        self.returns = returns.values

        # FeatureExtractor를 사용한 적절한 특징 계산
        from src.data.feature_extractor import FeatureExtractor
        feature_extractor = FeatureExtractor(window=20)

        T, N = self.prices.shape
        self.features = []

        for t in range(T):
            if t < 20:
                # 초기 윈도우는 0 패딩
                feat = np.zeros(feature_extractor.total_dim)
            else:
                # 가격 윈도우 생성
                price_window = pd.DataFrame(
                    self.prices[t-20:t+1],
                    columns=price_data.columns
                )
                # 특징 추출
                feat = feature_extractor.extract_features(price_window)
            self.features.append(feat)

        self.features = np.array(self.features)

        self.logger.info(f"데이터 로드 완료: prices shape={self.prices.shape}, "
                        f"returns shape={self.returns.shape}, features shape={self.features.shape}")

    def evaluate(self):
        """전체 평가 실행"""
        # SHAP explainer 초기화 (데이터 로드 후)
        if hasattr(self, 'features') and self.features is not None:
            # 완전한 상태 구성 (43차원)
            n_samples = min(100, len(self.features))
            n_assets = self.prices.shape[1]  # 30

            background_states = []
            for i in range(n_samples):
                # features (12차원)
                features = self.features[i]

                # portfolio weights (30차원) - 초기값은 균등 배분
                weights = np.ones(n_assets) / n_assets

                # crisis level (1차원) - 초기값 0
                crisis_level = 0.0

                # 완전한 상태 구성 (43차원)
                full_state = np.concatenate([features, weights, [crisis_level]])
                background_states.append(full_state)

            background_data = np.array(background_states)
            self.explainer.initialize_shap(background_data)
            self.logger.info(f"SHAP explainer 초기화 완료 (background shape: {background_data.shape})")

        self.logger.info("=" * 50)
        self.logger.info("평가 시작")
        self.logger.info("=" * 50)

        # 1. Backtest FinFlow
        self.logger.info("\n1. FinFlow 백테스팅")
        finflow_results = self._backtest_finflow()
        self.results['finflow'] = finflow_results
        
        # 2. Benchmark comparisons
        self.logger.info("\n2. 벤치마크 비교")
        benchmark_results = self._evaluate_benchmarks()
        self.results['benchmarks'] = benchmark_results
        
        # 3. Stability analysis
        self.logger.info("\n3. 안정성 분석")
        stability_results = self._analyze_stability()
        self.results['stability'] = stability_results
        
        # 4. XAI analysis
        self.logger.info("\n4. XAI 분석")
        xai_results = self._analyze_explainability()
        self.results['xai'] = xai_results
        
        # 5. Generate visualizations
        self.logger.info("\n5. 시각화 생성")
        self._create_visualizations()
        
        # 6. Generate report
        self.logger.info("\n6. 보고서 생성")
        self._generate_report()
        
        self.logger.info("=" * 50)
        self.logger.info("평가 완료")
        self._print_summary()
        self.logger.info("=" * 50)
    
    def _backtest_finflow(self) -> Dict:
        """FinFlow 백테스팅"""
        # 가격 데이터를 DataFrame으로 변환
        price_df = pd.DataFrame(
            self.prices,
            columns=[f'Asset_{i}' for i in range(self.prices.shape[1])]
        )

        # FeatureExtractor 초기화 (config에서 window 가져오기)
        from src.data.feature_extractor import FeatureExtractor
        train_config = self.config.get('train', {})
        feature_window = train_config.get('feature_window', 20)
        feature_extractor = FeatureExtractor(window=feature_window)

        # T-Cell이 학습되지 않았다면 평가 데이터로 학습
        if hasattr(self, 't_cell_needs_fitting') and self.t_cell_needs_fitting:
            self.logger.info("T-Cell을 평가 데이터로 학습 중...")

            # 평가 데이터의 처음 100개 샘플 사용
            if hasattr(self, 'features') and self.features is not None:
                # 기존에 추출된 features 사용
                training_features = self.features[:min(100, len(self.features))]
            else:
                # features가 없으면 새로 추출
                training_features = []
                for i in range(20, min(120, len(price_df))):
                    feature = feature_extractor.extract_features(price_df, current_idx=i)
                    training_features.append(feature)
                training_features = np.array(training_features)

            # T-Cell 학습
            self.t_cell.fit(training_features)
            self.logger.info(f"T-Cell 학습 완료: {len(training_features)} 샘플 사용")
            self.t_cell_needs_fitting = False

        # env 섹션 가져오기 (YAML에는 env 섹션이 있음)
        env_config = self.config.get('env', {})

        # Reports 디렉토리 생성
        reports_dir = self.run_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        # 환경 초기화 (price_data 필수 인자)
        env = PortfolioEnv(
            price_data=price_df,
            feature_extractor=feature_extractor,
            initial_capital=env_config.get('initial_capital', 1000000),
            transaction_cost=env_config.get('turnover_cost', 0.001),
            slippage=env_config.get('slip_coeff', 0.0005),
            no_trade_band=env_config.get('no_trade_band', 0.002),
            max_leverage=env_config.get('max_leverage', 1.0),
            max_turnover=env_config.get('max_turnover', 0.5)
        )

        # 자산 수 검증
        if env.n_assets != self.expected_action_dim:
            raise ValueError(
                f"자산 수 불일치: 데이터={env.n_assets}, "
                f"모델={self.expected_action_dim}. "
                f"체크포인트와 일치하는 {self.expected_action_dim}개 자산의 데이터를 준비하세요."
            )

        episode_returns = []
        episode_actions = []
        episode_rewards = []
        episode_equity_curves = []
        xai_reports = []
        
        # eval 또는 evaluation 섹션 지원
        eval_config = self.config.get('eval', self.config.get('evaluation', {}))
        n_episodes = eval_config.get('episodes', eval_config.get('n_episodes', 10))

        for episode in tqdm(range(n_episodes), desc="백테스팅"):
            state, _ = env.reset()
            done = False
            
            returns = []
            actions = []
            rewards = []
            equity_curve = [1.0]
            step_count = 0
            
            while not done:
                # Get action from FinFlow
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Crisis detection
                market_data = env.get_market_data()
                crisis_level, crisis_explanation = self.t_cell.detect_crisis(market_data['features'])

                # Memory recall
                memory_action = self.memory_cell.recall(
                    state, crisis_level
                )

                # Select action
                action = self.b_cell.select_action(
                    state.squeeze() if state.ndim > 1 else state,  # \uc2a4\ud0ec\uce58\ub41c numpy \ubc30\uc5f4 \uc804\ub2ec
                    crisis_level=crisis_level,
                    deterministic=True
                )

                # \uc561\uc158 \ucc28\uc6d0 \ud655\uc778 \ubc0f \uc870\uc815
                if len(action.shape) == 0:
                    action = np.array([action])
                elif action.shape[0] == 1:
                    action = action.squeeze()
                
                # XAI Analysis - 마지막 스텝 또는 100스텝마다
                if step_count % 100 == 0 or done:
                    # XAI 3함수 호출
                    local_attr = self.explainer.local_attribution(state, action)
                    cf_report = self.explainer.counterfactual(state, action, deltas={"volatility": -0.2})

                    # SHAP top-k 특성 계산
                    shap_topk_features = list(local_attr.items())[:5]  # 상위 5개 특성

                    # Memory recall를 similar cases로 변환
                    similar_cases = None
                    if memory_action is not None:
                        similar_cases = [memory_action]  # 메모리에서 리콜된 액션

                    # 위기 정보 구성
                    crisis_info_dict = {
                        'crisis_level': crisis_level,
                        'explanation': crisis_explanation
                    }

                    reg_report = self.explainer.regime_report(
                        crisis_info_dict,
                        shap_topk=shap_topk_features,
                        similar_cases=similar_cases
                    )
                    
                    # Decision card 생성 및 저장
                    decision_card = {
                        "timestamp": datetime.now().isoformat(),
                        "episode": episode,
                        "step": step_count,
                        "action": list(map(float, action)),
                        "local_attribution": local_attr,
                        "counterfactual": cf_report,
                        "regime_report": reg_report,
                        "crisis_info": crisis_info_dict
                    }
                    
                    # JSON 저장 (numpy 타입 변환)
                    card_path = self.run_dir / "reports" / f"decision_card_ep{episode}_step{step_count}.json"

                    # numpy 타입을 Python native 타입으로 변환
                    def convert_numpy(obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return obj

                    with open(card_path, 'w') as f:
                        json.dump(decision_card, f, indent=2, default=convert_numpy)
                    
                    xai_reports.append(decision_card)
                
                # Step
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                portfolio_return = info.get('portfolio_return', reward)
                returns.append(portfolio_return)
                actions.append(action)
                rewards.append(reward)
                equity_curve.append(equity_curve[-1] * (1 + portfolio_return))
                
                state = next_state
                step_count += 1
            
            episode_returns.append(returns)
            episode_actions.append(actions)
            episode_rewards.append(rewards)
            episode_equity_curves.append(equity_curve)
        
        # Calculate metrics
        all_returns = np.concatenate(episode_returns)
        all_actions = np.vstack(np.concatenate(episode_actions))
        all_equity = np.concatenate(episode_equity_curves)
        
        # 지표 계산
        sharpe = calculate_sharpe_ratio(all_returns, risk_free_rate=0.0)
        cvar_95 = calculate_cvar(all_returns, alpha=0.05)
        mdd = calculate_max_drawdown(all_equity)
        
        # 기본 메트릭과 새 메트릭 통합
        metrics = self._calculate_metrics(all_returns, all_actions)
        metrics.update({
            'sharpe_ratio_new': float(sharpe),
            'cvar_95_new': float(cvar_95),
            'max_drawdown_new': float(mdd)
        })
        
        # 메트릭 리포트 저장
        metrics_report = {
            "sharpe": float(sharpe),
            "cvar_95": float(cvar_95),
            "max_drawdown": float(mdd),
            "total_return": float(metrics['total_return']),
            "annual_return": float(metrics['annual_return']),
            "volatility": float(metrics['volatility'])
        }
        (self.run_dir / "reports" / "metrics.json").write_text(json.dumps(metrics_report, indent=2))
        
        # 시각화 생성 및 저장
        # Equity curve
        plot_equity_curve(all_equity, save_path=self.viz_dir / "finflow_equity_curve.png")

        # Drawdown
        plot_drawdown(all_equity, save_path=self.viz_dir / "finflow_drawdown.png")
        
        # Portfolio weights
        asset_names = [f"Asset_{i}" for i in range(all_actions.shape[1])]
        # 최근 가중치 사용 (시계열 대신 마지막 스텝의 스냅샷)
        latest_weights = all_actions[-1] if len(all_actions) > 0 else all_actions[0]
        plot_portfolio_weights(
            latest_weights,
            asset_names,
            save_path=self.viz_dir / "finflow_weights.png"
        )
        
        self.logger.info(f"XAI 리포트 {len(xai_reports)}개 생성 완료")
        self.logger.info(f"메트릭 및 시각화 저장 완료: {self.run_dir / 'reports'}")
        
        return {
            'metrics': metrics,
            'returns': all_returns,
            'actions': all_actions,
            'rewards': np.concatenate(episode_rewards),
            'equity_curve': all_equity,
            'xai_reports': xai_reports
        }

    def evaluate_with_backtest(self) -> Dict:
        """현실적인 백테스트를 포함한 평가"""
        self.logger.info("현실적인 백테스트 시작...")

        # Strategy wrapper for backtester
        def finflow_strategy(data: pd.DataFrame, positions: np.ndarray, timestamp: int) -> np.ndarray:
            """FinFlow 전략을 백테스터에 맞게 래핑"""
            # 현재 시점 features 추출
            if timestamp < self.config.get('features', {}).get('window', 20):
                # 초기에는 균등 배분
                n_assets = len(positions)
                return np.ones(n_assets) / n_assets

            # 상태 구성
            from src.data.feature_extractor import FeatureExtractor
            extractor = FeatureExtractor(
                window=self.config.get('features', {}).get('window', 20),
                feature_config=self.config.get('features', {})
            )
            features = extractor.extract_features(data, current_idx=timestamp)

            # 위기 감지
            crisis_level, crisis_explanation = self.t_cell.detect_crisis(features)

            # 전체 상태 구성
            state = np.concatenate([features, positions, [crisis_level]])
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # 메모리 리콜
            memory_action = self.memory_cell.recall(
                state, crisis_level
            )

            # Select action
            action = self.b_cell.select_action(
                state_tensor,
                deterministic=True
            )

            return action

        # 백테스트 실행
        backtest_results = self.backtester.backtest(
            strategy=finflow_strategy,
            data=pd.DataFrame(self.prices),
            initial_capital=self.config.get('env', {}).get('initial_capital', 1000000),
            verbose=True
        )

        # 결과 저장
        backtest_report_path = self.run_dir / "reports" / "backtest_results.json"
        with open(backtest_report_path, 'w') as f:
            # numpy array를 리스트로 변환하여 JSON serializable하게 만듦
            serializable_results = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in backtest_results.items()
            }
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"백테스트 결과 저장: {backtest_report_path}")

        return backtest_results

    def _evaluate_benchmarks(self) -> Dict:
        """벤치마크 전략 평가"""
        benchmarks = {}

        # eval.benchmark 사용 (YAML 구조에 맞춤)
        eval_config = self.config.get('eval', {})
        benchmark_strategy = eval_config.get('benchmark', 'equal_weight')

        # 단일 벤치마크를 리스트로 처리
        benchmark_strategies = [benchmark_strategy] if isinstance(benchmark_strategy, str) else benchmark_strategy

        for strategy in benchmark_strategies:
            self.logger.info(f"벤치마크 평가: {strategy}")
            
            if strategy == 'equal_weight':
                returns, actions = self._equal_weight_strategy()
            elif strategy == 'market_cap':
                returns, actions = self._market_cap_strategy()
            elif strategy == 'momentum':
                returns, actions = self._momentum_strategy()
            else:
                continue
            
            metrics = self._calculate_metrics(returns, actions)
            benchmarks[strategy] = {
                'metrics': metrics,
                'returns': returns,
                'actions': actions
            }
        
        return benchmarks
    
    def _equal_weight_strategy(self) -> Tuple[np.ndarray, np.ndarray]:
        """균등 가중 전략"""
        n_assets = self.returns.shape[1]
        weights = np.ones(n_assets) / n_assets
        
        returns = []
        actions = []
        
        for t in range(len(self.returns)):
            portfolio_return = np.dot(weights, self.returns[t])
            returns.append(portfolio_return)
            actions.append(weights.copy())
        
        return np.array(returns), np.array(actions)
    
    def _market_cap_strategy(self) -> Tuple[np.ndarray, np.ndarray]:
        """시가총액 가중 전략 (시뮬레이션)"""
        # Simulate market cap weights
        market_caps = np.random.lognormal(10, 2, self.returns.shape[1])
        weights = market_caps / market_caps.sum()
        
        returns = []
        actions = []
        
        for t in range(len(self.returns)):
            portfolio_return = np.dot(weights, self.returns[t])
            returns.append(portfolio_return)
            actions.append(weights.copy())
            
            # Update weights based on returns
            weights = weights * (1 + self.returns[t])
            weights = weights / weights.sum()
        
        return np.array(returns), np.array(actions)
    
    def _momentum_strategy(self) -> Tuple[np.ndarray, np.ndarray]:
        """모멘텀 전략"""
        train_config = self.config.get('train', {})
        lookback = train_config.get('momentum_lookback', 20)
        returns_list = []
        actions_list = []
        
        for t in range(lookback, len(self.returns)):
            # Calculate momentum
            past_returns = self.returns[t-lookback:t].mean(axis=0)
            
            # Rank and select top assets
            ranks = np.argsort(past_returns)[::-1]
            weights = np.zeros(len(ranks))
            weights[ranks[:3]] = 1/3  # Top 3 assets
            
            portfolio_return = np.dot(weights, self.returns[t])
            returns_list.append(portfolio_return)
            actions_list.append(weights)
        
        return np.array(returns_list), np.array(actions_list)
    
    def _calculate_metrics(self, returns: np.ndarray, actions: np.ndarray) -> Dict:
        """성능 메트릭 계산"""
        metrics = {}
        
        # Sharpe Ratio
        if len(returns) > 1:
            metrics['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        # CVaR (5%)
        sorted_returns = np.sort(returns)
        n_tail = max(1, len(sorted_returns) // 20)
        metrics['cvar_5'] = np.mean(sorted_returns[:n_tail])
        
        # Max Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = np.min(drawdown)
        
        # Turnover
        if len(actions) > 1:
            turnovers = [np.sum(np.abs(actions[i] - actions[i-1])) / 2 
                        for i in range(1, len(actions))]
            metrics['avg_turnover'] = np.mean(turnovers)
        else:
            metrics['avg_turnover'] = 0
        
        # Additional metrics
        metrics['total_return'] = np.prod(1 + returns) - 1
        metrics['annual_return'] = (1 + metrics['total_return']) ** (252 / len(returns)) - 1
        metrics['volatility'] = np.std(returns) * np.sqrt(252)
        metrics['win_rate'] = np.mean(returns > 0)
        
        return metrics
    
    def _analyze_stability(self) -> Dict:
        """안정성 분석"""
        finflow_actions = self.results['finflow']['actions']
        
        # Weight stability
        weight_changes = np.diff(finflow_actions, axis=0)
        weight_stability = 1 / (1 + np.std(weight_changes))
        
        # Concentration
        concentrations = np.sum(finflow_actions ** 2, axis=1)
        avg_concentration = np.mean(concentrations)
        
        # Effective assets
        effective_assets = np.sum(finflow_actions > 0.01, axis=1)
        avg_effective = np.mean(effective_assets)
        
        return {
            'weight_stability': weight_stability,
            'avg_concentration': avg_concentration,
            'avg_effective_assets': avg_effective,
            'max_single_weight': np.max(finflow_actions),
            'min_single_weight': np.min(finflow_actions[finflow_actions > 0])
        }
    
    def _analyze_explainability(self) -> Dict:
        """XAI 분석"""
        # Sample some decisions for explanation
        sample_states = []
        sample_actions = []

        # price_df 생성 (self.prices를 DataFrame으로 변환)
        price_df = pd.DataFrame(
            self.prices,
            columns=[f'Asset_{i}' for i in range(self.prices.shape[1])]
        )

        # FeatureExtractor 초기화 (config에서 window 가져오기)
        from src.data.feature_extractor import FeatureExtractor
        train_config = self.config.get('train', {})
        feature_window = train_config.get('feature_window', 20)
        feature_extractor = FeatureExtractor(window=feature_window)

        # env config 가져오기
        env_config = self.config.get('env', {})

        # PortfolioEnv 초기화 (올바른 파라미터 매핑)
        env = PortfolioEnv(
            price_data=price_df,
            feature_extractor=feature_extractor,
            initial_capital=env_config.get('initial_capital', 1000000),
            transaction_cost=env_config.get('turnover_cost', 0.001),
            slippage=env_config.get('slip_coeff', 0.0005),
            no_trade_band=env_config.get('no_trade_band', 0.002),
            max_leverage=env_config.get('max_leverage', 1.0),
            max_turnover=env_config.get('max_turnover', 0.5)
        )

        state, _ = env.reset()
        
        for _ in range(5):  # Explain 5 decisions
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action
            action = self.b_cell.select_action(
                state.squeeze() if state.ndim > 1 else state,  # numpy array \uc804\ub2ec
                deterministic=True
            )
            
            sample_states.append(state)
            sample_actions.append(action)
            
            # Step (for next state)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        
        # Generate explanations
        explanations = []
        for state, action in zip(sample_states, sample_actions):
            explanation = self.explainer.explain_decision(state, action)
            explanations.append(explanation)
        
        # Aggregate feature importance
        all_importance = {}
        for exp in explanations:
            for feature, importance in exp.feature_importance.items():
                if feature not in all_importance:
                    all_importance[feature] = []
                all_importance[feature].append(importance)
        
        avg_importance = {k: np.mean(v) for k, v in all_importance.items()}
        
        return {
            'avg_feature_importance': avg_importance,
            'n_explanations': len(explanations),
            'avg_confidence': np.mean([exp.confidence for exp in explanations])
        }
    
    def _create_visualizations(self):
        """시각화 생성"""
        # 1. Performance comparison
        self._plot_performance_comparison()
        
        # 2. Returns distribution
        self._plot_returns_distribution()
        
        # 3. Portfolio weights over time
        self._plot_portfolio_weights()
        
        # 4. Drawdown analysis
        self._plot_drawdown()
    
    def _plot_performance_comparison(self):
        """성능 비교 플롯"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Collect metrics
        strategies = ['FinFlow'] + list(self.results['benchmarks'].keys())
        metrics_data = {
            'Sharpe Ratio': [],
            'CVaR (5%)': [],
            'Max Drawdown': [],
            'Turnover': []
        }
        
        for strategy in strategies:
            if strategy == 'FinFlow':
                m = self.results['finflow']['metrics']
            else:
                m = self.results['benchmarks'][strategy]['metrics']
            
            metrics_data['Sharpe Ratio'].append(m['sharpe_ratio'])
            metrics_data['CVaR (5%)'].append(m['cvar_5'])
            metrics_data['Max Drawdown'].append(m['max_drawdown'])
            metrics_data['Turnover'].append(m['avg_turnover'])
        
        # Plot each metric
        for ax, (metric_name, values) in zip(axes.flat, metrics_data.items()):
            ax.bar(strategies, values)
            ax.set_title(metric_name)
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "performance_comparison.png")
        plt.close()
    
    def _plot_returns_distribution(self):
        """수익률 분포 플롯"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot returns distribution
        finflow_returns = self.results['finflow']['returns']
        
        ax.hist(finflow_returns, bins=50, alpha=0.7, label='FinFlow', density=True)
        
        for name, data in self.results['benchmarks'].items():
            ax.hist(data['returns'], bins=50, alpha=0.5, label=name, density=True)
        
        ax.set_xlabel('Returns')
        ax.set_ylabel('Density')
        ax.set_title('Returns Distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "returns_distribution.png")
        plt.close()
    
    def _plot_portfolio_weights(self):
        """포트폴리오 가중치 플롯"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        actions = self.results['finflow']['actions']
        
        # Plot stacked area chart
        x = np.arange(len(actions))
        
        # Select top 5 assets by average weight
        avg_weights = actions.mean(axis=0)
        top_assets = np.argsort(avg_weights)[::-1][:5]
        
        bottom = np.zeros(len(actions))
        for i in top_assets:
            ax.fill_between(x, bottom, bottom + actions[:, i], 
                           label=f'Asset {i}', alpha=0.7)
            bottom += actions[:, i]
        
        # Others
        others = actions[:, [i for i in range(actions.shape[1]) if i not in top_assets]].sum(axis=1)
        ax.fill_between(x, bottom, bottom + others, label='Others', alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Portfolio Weight')
        ax.set_title('Portfolio Weights Over Time')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "portfolio_weights.png")
        plt.close()
    
    def _plot_drawdown(self):
        """드로다운 플롯"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate drawdown for FinFlow
        returns = self.results['finflow']['returns']
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        ax.fill_between(np.arange(len(drawdown)), 0, drawdown, 
                        color='red', alpha=0.3, label='FinFlow')
        ax.plot(drawdown, color='red', linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Drawdown')
        ax.set_title('Drawdown Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "drawdown.png")
        plt.close()
    
    def _generate_report(self):
        """평가 보고서 생성"""
        from pathlib import Path
        report_path = self.run_dir / "reports" / "evaluation_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        # Save detailed results
        with open(report_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"평가 보고서 저장: {report_path}")
    
    def _make_serializable(self, obj):
        """객체를 JSON 직렬화 가능하게 변환"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        else:
            return obj
    
    def _print_summary(self):
        """요약 출력"""
        print("\n" + "=" * 50)
        print("평가 요약")
        print("=" * 50)
        
        # FinFlow metrics
        finflow_metrics = self.results['finflow']['metrics']
        print(f"\nFinFlow 성능:")
        print(f"  Sharpe Ratio: {finflow_metrics['sharpe_ratio']:.3f}")
        print(f"  CVaR (5%): {finflow_metrics['cvar_5']:.3f}")
        print(f"  Max Drawdown: {finflow_metrics['max_drawdown']:.3f}")
        print(f"  Annual Return: {finflow_metrics['annual_return']*100:.1f}%")
        print(f"  Volatility: {finflow_metrics['volatility']*100:.1f}%")
        
        # Benchmark comparison
        print(f"\n벤치마크 대비:")
        for name, data in self.results['benchmarks'].items():
            metrics = data['metrics']
            sharpe_diff = finflow_metrics['sharpe_ratio'] - metrics['sharpe_ratio']
            print(f"  vs {name}: Sharpe +{sharpe_diff:.3f}" if sharpe_diff > 0 
                  else f"  vs {name}: Sharpe {sharpe_diff:.3f}")
        
        # Stability
        stability = self.results['stability']
        print(f"\n안정성 지표:")
        print(f"  Weight Stability: {stability['weight_stability']:.3f}")
        print(f"  Avg Concentration: {stability['avg_concentration']:.3f}")
        print(f"  Avg Effective Assets: {stability['avg_effective_assets']:.1f}")
        
        # XAI
        xai = self.results['xai']
        print(f"\nXAI 분석:")
        print(f"  Avg Confidence: {xai['avg_confidence']:.3f}")
        print(f"  Top Features:")
        
        top_features = sorted(xai['avg_feature_importance'].items(), 
                             key=lambda x: x[1], reverse=True)[:3]
        for feature, importance in top_features:
            print(f"    - {feature}: {importance*100:.1f}%")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='FinFlow Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data', type=str, default='data/test',
                       help='Path to test data')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--with-backtest', action='store_true',
                       help='Run realistic backtest')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create evaluator
    evaluator = FinFlowEvaluator(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        config_path=args.config,
        device=args.device
    )

    # Run evaluation
    evaluator.evaluate()

    # Run realistic backtest if requested
    if args.with_backtest:
        print("\n" + "="*60)
        print("Running Realistic Backtest...")
        print("="*60)
        backtest_results = evaluator.evaluate_with_backtest()
        print(f"\nBacktest completed. Results saved to: {evaluator.run_dir / 'reports' / 'backtest_results.json'}")


if __name__ == "__main__":
    main()