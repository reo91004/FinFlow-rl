# src/experiments/ablation.py

"""
절제 연구 (Ablation Study) 실행 모듈

목적: FinFlow-RL 시스템의 각 컴포넌트 기여도 측정
의존: trainer.py, logger.py
사용처: 연구 및 논문 작성시 컴포넌트별 성능 비교
역할: T-Cell, Memory Cell, XAI 등 각 모듈의 영향력 분석

구현 내용:
- 전체 시스템 vs 개별 컴포넌트 제거 실험
- IQL vs TD3+BC 오프라인 방법 비교
- 성능 메트릭 수집 및 비교표 생성
- 통계적 유의성 검증
"""

import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import copy

from src.training.trainer import FinFlowTrainer
from src.utils.logger import FinFlowLogger


class AblationStudy:
    """
    Ablation Study 실행기
    각 컴포넌트의 기여도 측정
    """

    def __init__(self, base_config: Dict):
        """
        Args:
            base_config: 기본 설정
        """
        self.base_config = base_config
        self.results = {}
        self.logger = FinFlowLogger("AblationStudy")

    def run_experiments(self):
        """모든 ablation 실험 실행"""

        configurations = {
            'full': {
                'description': '전체 시스템 (모든 컴포넌트 활성화)',
                'ablation': {
                    'use_tcell': True,
                    'use_memory': True,
                    'use_xai': True,
                    'offline_method': 'iql'
                }
            },
            'no_tcell': {
                'description': 'T-Cell 제거',
                'ablation': {
                    'use_tcell': False,
                    'use_memory': True,
                    'use_xai': True,
                    'offline_method': 'iql'
                }
            },
            'no_memory': {
                'description': 'Memory Cell 제거',
                'ablation': {
                    'use_tcell': True,
                    'use_memory': False,
                    'use_xai': True,
                    'offline_method': 'iql'
                }
            },
            'no_offline': {
                'description': '오프라인 사전학습 제거',
                'ablation': {
                    'use_tcell': True,
                    'use_memory': True,
                    'use_xai': True,
                    'offline_method': 'iql'
                },
                'skip_offline': True
            },
            'td3bc_offline': {
                'description': 'TD3+BC 오프라인 학습',
                'ablation': {
                    'use_tcell': True,
                    'use_memory': True,
                    'use_xai': True,
                    'offline_method': 'td3bc'
                }
            },
            'minimal': {
                'description': '최소 구성 (B-Cell만)',
                'ablation': {
                    'use_tcell': False,
                    'use_memory': False,
                    'use_xai': False,
                    'offline_method': 'iql'
                },
                'skip_offline': True
            },
            'baseline_equal': {
                'description': '균등 가중치 베이스라인',
                'method': 'equal_weight'
            },
            'baseline_sac': {
                'description': '표준 SAC (면역 메타포 없음)',
                'method': 'standard_sac'
            }
        }

        self.logger.info("="*60)
        self.logger.info("Ablation Study 시작")
        self.logger.info(f"총 {len(configurations)}개 실험")
        self.logger.info("="*60)

        for name, config in configurations.items():
            self.logger.info(f"\n실험: {name}")
            self.logger.info(f"설명: {config['description']}")
            self.logger.info("="*60)

            # 설정 업데이트
            experiment_config = copy.deepcopy(self.base_config)

            if 'ablation' in config:
                experiment_config['ablation'] = config['ablation']
            if 'skip_offline' in config:
                experiment_config['skip_offline'] = config['skip_offline']

            # 실험 실행
            if 'method' in config:
                # 베이스라인 실행
                metrics = self._run_baseline(config['method'])
            else:
                # FinFlow 변형 실행
                trainer = FinFlowTrainer(experiment_config)
                trainer.train()
                metrics = self._extract_metrics(trainer)

            # 결과 저장
            self.results[name] = {
                'config': config,
                'metrics': metrics
            }

            self.logger.info(f"결과: Sharpe={metrics['sharpe']:.3f}, "
                           f"Returns={metrics['returns']:.1%}, "
                           f"MDD={metrics.get('mdd', 0):.1%}")

        # 결과 저장
        self._save_results()

        # 분석 리포트 생성
        self._generate_report()

    def _run_baseline(self, method: str) -> Dict:
        """베이스라인 실행"""
        if method == 'equal_weight':
            from src.baselines.equal_weight import EqualWeightStrategy
            strategy = EqualWeightStrategy()
        elif method == 'standard_sac':
            from src.baselines.standard_sac import StandardSAC
            strategy = StandardSAC(self.base_config)
        else:
            raise ValueError(f"Unknown baseline method: {method}")

        return strategy.backtest(self.base_config)

    def _extract_metrics(self, trainer: FinFlowTrainer) -> Dict:
        """트레이너에서 메트릭 추출"""
        # 테스트 데이터로 최종 평가
        test_metrics = trainer._evaluate(trainer.test_data, "최종평가", max_episodes=20)

        # 추가 메트릭 계산
        if trainer.monitor:
            portfolio_values = trainer.monitor.get_portfolio_values()
            if len(portfolio_values) > 0:
                # Maximum Drawdown
                cumulative = np.array(portfolio_values)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                test_metrics['mdd'] = np.min(drawdown)

                # Calmar Ratio
                annual_return = test_metrics['returns'] * 252
                calmar = annual_return / abs(test_metrics['mdd']) if test_metrics['mdd'] != 0 else 0
                test_metrics['calmar'] = calmar

        return test_metrics

    def _save_results(self):
        """결과 저장"""
        output_dir = Path('results/ablation')
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON 저장
        with open(output_dir / 'ablation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # DataFrame으로 변환
        df_data = []
        for name, data in self.results.items():
            row = {'experiment': name}
            row.update(data['metrics'])
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv(output_dir / 'ablation_results.csv', index=False)

        self.logger.info(f"결과 저장 완료: {output_dir}")

    def _generate_report(self):
        """분석 리포트 생성"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Ablation Study 결과 요약")
        self.logger.info("="*60)

        # 기준선 성능
        if 'baseline_equal' in self.results:
            baseline_sharpe = self.results['baseline_equal']['metrics']['sharpe']
        else:
            baseline_sharpe = 0

        if 'full' in self.results:
            full_sharpe = self.results['full']['metrics']['sharpe']
            full_returns = self.results['full']['metrics']['returns']
        else:
            full_sharpe = 0
            full_returns = 0

        # 개선도
        if baseline_sharpe != 0:
            improvement = (full_sharpe / baseline_sharpe - 1) * 100
            self.logger.info(f"\n균등 가중치 대비 개선: {improvement:.1f}%")

        # 각 컴포넌트 기여도
        self.logger.info("\n컴포넌트별 기여도 (Sharpe 감소):")

        contributions = {}
        if 'full' in self.results and 'no_tcell' in self.results:
            contributions['T-Cell'] = full_sharpe - self.results['no_tcell']['metrics']['sharpe']
        if 'full' in self.results and 'no_memory' in self.results:
            contributions['Memory'] = full_sharpe - self.results['no_memory']['metrics']['sharpe']
        if 'full' in self.results and 'no_offline' in self.results:
            contributions['Offline'] = full_sharpe - self.results['no_offline']['metrics']['sharpe']

        for component, contribution in contributions.items():
            percentage = (contribution / full_sharpe) * 100 if full_sharpe != 0 else 0
            self.logger.info(f"  {component}: {contribution:.3f} ({percentage:.1f}%)")

        # 오프라인 방법 비교
        if 'full' in self.results and 'td3bc_offline' in self.results:
            iql_sharpe = self.results['full']['metrics']['sharpe']
            td3bc_sharpe = self.results['td3bc_offline']['metrics']['sharpe']
            self.logger.info(f"\n오프라인 방법 비교:")
            self.logger.info(f"  IQL: {iql_sharpe:.3f}")
            self.logger.info(f"  TD3+BC: {td3bc_sharpe:.3f}")

        # 전체 결과 테이블
        self.logger.info("\n전체 실험 결과:")
        self.logger.info("-"*60)
        self.logger.info(f"{'실험':<20} {'Sharpe':>10} {'Returns':>10} {'MDD':>10}")
        self.logger.info("-"*60)

        for name, data in self.results.items():
            metrics = data['metrics']
            self.logger.info(
                f"{name:<20} "
                f"{metrics['sharpe']:>10.3f} "
                f"{metrics['returns']*100:>9.1f}% "
                f"{metrics.get('mdd', 0)*100:>9.1f}%"
            )

    def run_statistical_tests(self):
        """통계적 유의성 검정"""
        # 각 실험을 여러 번 실행하여 통계적 유의성 검정
        # (시간이 오래 걸려서 선택적으로 실행)
        pass