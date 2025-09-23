# src/core/trainer.py

"""
메인 학습 파이프라인 조정자

목적: 오프라인 사전학습 → 온라인 미세조정 통합
의존: 모든 core/agents/data 모듈
사용처: main.py, scripts/train.py
역할: 전체 학습 프로세스 관리

파이프라인:
- Phase 1: OfflineDataset → OfflineTrainer → IQL/TD3BC
- Phase 2: BCell + TCell + Memory → 온라인 학습
- Phase 3: 평가 및 결과 저장

절제 연구(ablation) 지원
"""

import numpy as np
import torch
from typing import Dict, Optional, Any
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json

from src.core.env import PortfolioEnv
from src.core.offline_trainer import OfflineTrainer
from src.agents.b_cell import BCell
from src.agents.t_cell import TCell
from src.agents.memory import MemoryCell
from src.analysis.monitor import PerformanceMonitor
from src.analysis.explainer import XAIExplainer
from src.utils.logger import FinFlowLogger
from src.data.loader import DataLoader
from src.data.features import FeatureExtractor
from src.core.objectives import PortfolioObjective
from src.core.replay import Transition

class FinFlowTrainer:
    """
    FinFlow 통합 학습 파이프라인
    Phase 1: 오프라인 사전학습 (IQL 또는 TD3+BC)
    Phase 2: 온라인 미세조정 (REDQ)
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 학습 설정
        """
        self.config = config
        self.logger = FinFlowLogger("Trainer")
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Ablation study 설정
        self.ablation_config = config.get('ablation', {
            'use_tcell': True,
            'use_memory': True,
            'use_xai': True,
            'offline_method': 'iql'  # 'iql' or 'td3bc'
        })

        self.logger.info("="*60)
        self.logger.info("FinFlow Trainer 초기화")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Ablation 설정: {self.ablation_config}")
        self.logger.info("="*60)

        # 데이터 로드
        self._load_data()

        # 컴포넌트 초기화
        self._initialize_components()

        # 모니터링
        self.monitor = PerformanceMonitor()

    def _load_data(self):
        """데이터 로드 및 분할"""
        self.logger.info("데이터 로드 중...")

        loader = DataLoader(self.config.get('data', {}))
        self.price_data = loader.load()

        # 학습/검증/테스트 분할
        n = len(self.price_data)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        self.train_data = self.price_data[:train_end]
        self.val_data = self.price_data[train_end:val_end]
        self.test_data = self.price_data[val_end:]

        # 특성 추출기
        self.feature_extractor = FeatureExtractor()

        self.logger.info(f"데이터 분할: Train={len(self.train_data)}, Val={len(self.val_data)}, Test={len(self.test_data)}")

    def _initialize_components(self):
        """컴포넌트 초기화"""
        # 차원 계산
        n_assets = len(self.price_data.columns)
        feature_dim = self.config.get('feature_dim', 12)
        state_dim = feature_dim + n_assets + 1  # features + weights + crisis

        self.n_assets = n_assets
        self.state_dim = state_dim
        self.action_dim = n_assets

        self.logger.info(f"차원: state={state_dim}, action={n_assets}")

        # 환경
        self.train_env = PortfolioEnv(
            price_data=self.train_data,
            feature_extractor=self.feature_extractor,
            config=self.config.get('env', {})
        )

        self.val_env = PortfolioEnv(
            price_data=self.val_data,
            feature_extractor=self.feature_extractor,
            config=self.config.get('env', {})
        )

        # T-Cell (조건부)
        if self.ablation_config['use_tcell']:
            self.t_cell = TCell(
                feature_dim=feature_dim,
                contamination=self.config.get('tcell', {}).get('contamination', 0.1)
            )
            # 과거 데이터로 학습
            self._train_tcell()
        else:
            self.t_cell = None
            self.logger.info("T-Cell 비활성화 (Ablation)")

        # B-Cell
        self.b_cell = BCell(
            state_dim=state_dim,
            action_dim=n_assets,
            config=self.config.get('bcell', {}),
            device=self.device
        )

        # Memory Cell (조건부)
        if self.ablation_config['use_memory']:
            self.memory_cell = MemoryCell(
                capacity=self.config.get('memory', {}).get('capacity', 1000)
            )
        else:
            self.memory_cell = None
            self.logger.info("Memory Cell 비활성화 (Ablation)")

        # XAI (조건부)
        if self.ablation_config['use_xai']:
            self.explainer = XAIExplainer(
                model=self.b_cell,
                feature_names=self.feature_extractor.get_feature_names()
            )
        else:
            self.explainer = None
            self.logger.info("XAI 비활성화 (Ablation)")

    def _train_tcell(self):
        """T-Cell 학습"""
        self.logger.info("T-Cell 학습 중...")

        # 학습 데이터에서 특성 추출
        features_list = []
        for i in range(100, min(1000, len(self.train_data))):
            window_data = self.train_data.iloc[i-100:i]
            features = self.feature_extractor.extract(window_data)
            if features is not None:
                features_list.append(features[:12])  # 시장 특성만

        if features_list:
            historical_features = np.array(features_list)
            self.t_cell.fit(historical_features)
            self.logger.info(f"T-Cell 학습 완료: {len(historical_features)} 샘플")

    def train(self):
        """전체 학습 파이프라인"""
        self.logger.info("="*60)
        self.logger.info("FinFlow 학습 시작")
        self.logger.info("="*60)

        # Phase 1: 오프라인 사전학습
        if self.config.get('skip_offline', False):
            self.logger.info("오프라인 학습 스킵")
        else:
            self.logger.info("\n[Phase 1] 오프라인 사전학습")
            self._offline_pretrain()

        # Phase 2: 온라인 미세조정
        self.logger.info("\n[Phase 2] 온라인 미세조정")
        best_model = self._online_finetune()

        # 최종 평가
        self.logger.info("\n[Phase 3] 최종 평가")
        test_metrics = self._evaluate(self.test_data, "테스트")

        # 결과 저장
        self._save_results(test_metrics)

        return best_model

    def _offline_pretrain(self):
        """오프라인 사전학습"""
        from src.core.offline_dataset import OfflineDataset

        # 오프라인 데이터 수집/로드
        offline_data_path = Path('data/offline_data.npz')

        if not offline_data_path.exists():
            self.logger.info("오프라인 데이터 수집 중...")
            # OfflineDataset으로 수집
            dataset = OfflineDataset()
            dataset.collect_from_env(
                self.train_env,
                n_episodes=100,
                diversity_bonus=True,
                verbose=True
            )
            dataset.save(offline_data_path)
            offline_data = dataset.get_all_data()
        else:
            self.logger.info("기존 오프라인 데이터 로드")
            # NPZ 파일 로드
            dataset = OfflineDataset(offline_data_path)
            offline_data = dataset.get_all_data()

        # 오프라인 학습
        offline_trainer = OfflineTrainer(
            method=self.ablation_config['offline_method'],
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=self.config.get('offline', {}),
            device=self.device
        )

        offline_agent = offline_trainer.train(offline_data)

        # B-Cell에 가중치 전이
        self.b_cell.load_from_offline(offline_agent)
        self.logger.info("오프라인 가중치 전이 완료")


    def _online_finetune(self) -> BCell:
        """온라인 미세조정"""
        best_sharpe = -np.inf
        best_model = None
        patience_counter = 0
        patience = self.config.get('early_stopping_patience', 20)

        n_episodes = self.config.get('online_episodes', 200)

        for episode in tqdm(range(n_episodes), desc="온라인 학습"):
            # 학습
            train_metrics = self._train_episode(episode)

            # 주기적 검증
            if episode % 10 == 0:
                val_metrics = self._evaluate(self.val_data, "검증", max_episodes=5)

                # 최고 성능 갱신
                if val_metrics['sharpe'] > best_sharpe:
                    best_sharpe = val_metrics['sharpe']
                    best_model = self._save_checkpoint(episode)
                    patience_counter = 0
                    self.logger.info(f"최고 성능 갱신: Sharpe={best_sharpe:.3f}")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at episode {episode}")
                    break

                # 로깅
                self.logger.info(
                    f"Episode {episode}: "
                    f"Train Return={train_metrics['episode_return']:.4f}, "
                    f"Val Sharpe={val_metrics['sharpe']:.3f}"
                )

        return best_model or self.b_cell

    def _train_episode(self, episode: int) -> Dict:
        """단일 에피소드 학습"""
        state, _ = self.train_env.reset()
        episode_return = 0
        episode_length = 0
        losses = []

        done = False
        while not done:
            # 위기 감지
            crisis_level = 0.0
            if self.t_cell:
                features = state[:12]  # 첫 12차원이 특성
                crisis_level, _ = self.t_cell.detect_crisis(features)

            # 메모리 검색
            memory_action = None
            if self.memory_cell:
                memory_action = self.memory_cell.recall(state, crisis_level)

            # 행동 선택
            if memory_action is not None and crisis_level > 0.7:
                # 위기시 메모리 활용
                action = 0.5 * memory_action + 0.5 * self.b_cell.select_action(state, crisis_level)
            else:
                action = self.b_cell.select_action(state, crisis_level)

            # 환경 스텝
            next_state, reward, done, truncated, _ = self.train_env.step(action)

            # 경험 저장
            self.b_cell.replay_buffer.push(
                Transition(state, action, reward, next_state, done or truncated)
            )

            # 메모리 업데이트
            if self.memory_cell and reward > 0:
                self.memory_cell.store(state, action, reward, crisis_level)

            # B-Cell 학습
            if len(self.b_cell.replay_buffer) > 1000:
                loss = self.b_cell.train(batch_size=256)
                if loss:
                    losses.append(loss)

            state = next_state
            episode_return += reward
            episode_length += 1

            if done or truncated:
                break

        # 에피소드 통계
        metrics = {
            'episode_return': episode_return,
            'episode_length': episode_length,
        }

        if losses:
            avg_losses = {}
            for key in losses[0].keys():
                avg_losses[key] = np.mean([l[key] for l in losses if key in l])
            metrics.update(avg_losses)

        return metrics

    def _evaluate(self, data: pd.DataFrame, phase: str, max_episodes: int = 10) -> Dict:
        """정책 평가"""
        env = PortfolioEnv(
            price_data=data,
            feature_extractor=self.feature_extractor,
            config=self.config.get('env', {})
        )

        episode_returns = []
        episode_lengths = []

        for episode in range(min(max_episodes, 10)):
            state, _ = env.reset()
            episode_return = 0
            episode_length = 0
            returns = []

            done = False
            while not done:
                # 위기 감지
                crisis_level = 0.0
                if self.t_cell:
                    features = state[:12]
                    crisis_level, _ = self.t_cell.detect_crisis(features)

                # 행동 선택 (결정적)
                action = self.b_cell.select_action(state, crisis_level, deterministic=True)

                # 환경 스텝
                state, reward, done, truncated, _ = env.step(action)

                episode_return += reward
                episode_length += 1
                returns.append(reward)

                if done or truncated:
                    break

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

        # 메트릭 계산
        returns = np.array(episode_returns)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

        metrics = {
            'sharpe': sharpe,
            'returns': np.mean(returns),
            'std': np.std(returns),
            'max_return': np.max(returns),
            'min_return': np.min(returns),
            'avg_length': np.mean(episode_lengths),
        }

        self.logger.info(f"{phase} 평가: Sharpe={sharpe:.3f}, Return={np.mean(returns):.4f}")

        return metrics

    def _save_checkpoint(self, episode: int) -> BCell:
        """체크포인트 저장"""
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f'episode_{episode}.pt'
        self.b_cell.save(str(checkpoint_path))

        # T-Cell, Memory 저장
        if self.t_cell:
            self.t_cell.save(str(checkpoint_dir / f't_cell_{episode}.pkl'))
        if self.memory_cell:
            self.memory_cell.save(str(checkpoint_dir / f'memory_{episode}.pkl'))

        return self.b_cell

    def _save_results(self, metrics: Dict):
        """결과 저장"""
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)

        # 결과 파일
        results = {
            'config': self.config,
            'ablation': self.ablation_config,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        with open(results_dir / 'final_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"결과 저장 완료: {results_dir / 'final_results.json'}")