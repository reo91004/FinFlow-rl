# scripts/evaluate_irt.py

"""
IRT 평가 스크립트

사용법:
python scripts/evaluate_irt.py --checkpoint logs/YYYYMMDD_HHMMSS/checkpoints/best_model.pth --config configs/default_irt.yaml
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from src.agents.bcell_irt import BCellIRTActor
from src.algorithms.critics.redq import REDQCritic
from src.environments.portfolio_env import PortfolioEnv
from src.data.market_loader import DataLoader
from src.data.feature_extractor import FeatureExtractor
from src.evaluation.metrics import MetricsCalculator
from src.utils.logger import FinFlowLogger

class IRTEvaluator:
    """IRT 모델 평가기"""

    def __init__(self, config: Dict, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.logger = FinFlowLogger("IRTEvaluator")
        self.metrics_calc = MetricsCalculator()

        # 데이터 로드
        self._load_data()

        # 모델 로드
        self._load_model()

    def _load_data(self):
        """테스트 데이터 로드"""
        data_config = self.config['data']

        loader = DataLoader(cache_dir='data/cache')
        self.price_data = loader.download_data(
            symbols=data_config['symbols'],
            start_date=data_config['test_start'],
            end_date=data_config['test_end'],
            use_cache=data_config.get('cache', True)
        )

        self.logger.info(f"테스트 데이터 로드 완료: {len(self.price_data)}일")

    def _load_model(self):
        """체크포인트에서 모델 로드"""
        # 차원 계산
        n_assets = len(self.price_data.columns)
        feature_dim = self.config.get('feature_dim', 12)
        state_dim = feature_dim + n_assets + 1

        # IRT Actor
        irt_config = self.config.get('irt', {})
        self.actor = BCellIRTActor(
            state_dim=state_dim,
            action_dim=n_assets,
            emb_dim=irt_config.get('emb_dim', 128),
            m_tokens=irt_config.get('m_tokens', 6),
            M_proto=irt_config.get('M_proto', 8),
            alpha=irt_config.get('alpha', 0.3)
        ).to(self.device)

        # 체크포인트 로드
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()

        self.logger.info(f"모델 로드 완료: {self.checkpoint_path}")

    def evaluate(self):
        """평가 실행"""
        # 환경 생성
        env_config = self.config['env']
        objective_config = self.config.get('objectives')

        env = PortfolioEnv(
            price_data=self.price_data,
            feature_extractor=FeatureExtractor(window=20),
            initial_capital=env_config.get('initial_balance', 1000000),
            transaction_cost=env_config.get('transaction_cost', 0.001),
            max_leverage=env_config.get('max_leverage', 1.0),
            objective_config=objective_config,
            use_advanced_reward=(objective_config is not None)
        )

        # 에피소드 실행
        state, _ = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        episode_return = 0
        all_returns = []
        all_weights = []
        crisis_levels = []
        prototype_weights = []

        done = False
        truncated = False

        while not (done or truncated):
            with torch.no_grad():
                action, info = self.actor(state_tensor, deterministic=True)

            action_np = action.cpu().numpy()[0]

            # 환경 스텝
            next_state, reward, done, truncated, _ = env.step(action_np)

            # 기록
            episode_return += reward
            all_returns.append(reward)
            all_weights.append(action_np)
            crisis_levels.append(info['crisis_level'].item())
            prototype_weights.append(info['w'].cpu().numpy()[0])

            state = next_state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 메트릭 계산
        returns_array = np.array(all_returns)
        metrics = self.metrics_calc.calculate_all_metrics(returns_array)
        metrics['total_return'] = episode_return
        metrics['avg_crisis_level'] = np.mean(crisis_levels)

        # 결과 출력
        self.logger.info("="*60)
        self.logger.info("평가 결과")
        self.logger.info("="*60)
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value:.4f}")

        # 결과 저장
        results = {
            'metrics': metrics,
            'returns': all_returns,
            'weights': all_weights,
            'crisis_levels': crisis_levels,
            'prototype_weights': prototype_weights
        }

        save_dir = self.checkpoint_path.parent.parent / 'evaluation'
        save_dir.mkdir(exist_ok=True)

        # NumPy 배열을 리스트로 변환
        import json
        with open(save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        self.logger.info(f"결과 저장 완료: {save_dir}")

        return metrics

def main():
    parser = argparse.ArgumentParser(description='IRT Evaluation Script')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='configs/default_irt.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 평가 실행
    evaluator = IRTEvaluator(config, args.checkpoint)
    metrics = evaluator.evaluate()

if __name__ == '__main__':
    main()