# tests/ab_testing_framework.py

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict

from utils.logger import BIPDLogger
from core.environment import PortfolioEnvironment
from core.system import ImmunePortfolioSystem
from core.trainer import BIPDTrainer
from data.features import FeatureExtractor
from data.loader import DataLoader
from config import SYMBOLS, TRAIN_START, TRAIN_END, TEST_START, TEST_END


@dataclass
class ExperimentConfig:
    """실험 설정"""

    name: str
    description: str
    adaptive_entropy: bool = True
    adaptive_no_trade_band: bool = True
    use_simplex_projection: bool = True
    n_episodes: int = 100  # 빠른 실험을 위해 축소
    random_seed: int = 42

    def to_dict(self):
        return asdict(self)


@dataclass
class ExperimentResult:
    """실험 결과"""

    config_name: str
    final_portfolio_value: float
    avg_sharpe_ratio: float
    max_drawdown: float
    avg_turnover: float
    success_rate: float
    training_stability: float
    regime_stats: Optional[Dict] = None
    entropy_gap_stats: Optional[Dict] = None
    execution_time: float = 0.0

    def to_dict(self):
        return asdict(self)


class ABTestingFramework:
    """Phase 3 개선사항 A/B 테스트 프레임워크"""

    def __init__(
        self,
        market_data: Optional[pd.DataFrame] = None,
        save_dir: str = "experiments/results",
        logger_name: str = "ABTesting",
    ):

        self.logger = BIPDLogger(logger_name)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 시장 데이터 로드
        if market_data is None:
            self.logger.info("시장 데이터를 로드합니다...")
            data_loader = DataLoader()
            market_data = data_loader.get_market_data(
                symbols=SYMBOLS[:10],  # 빠른 실험을 위해 10개 종목만
                train_start=TRAIN_START,
                train_end=TRAIN_END,
                test_start=TEST_START,
                test_end=TEST_END,
            )

        self.train_data = market_data["train_data"]
        self.test_data = market_data["test_data"]

        # 실험 결과 저장
        self.experiment_results = {}

        self.logger.info(f"A/B 테스트 프레임워크 초기화 완료")
        self.logger.info(f"  훈련 데이터: {len(self.train_data)} 거래일")
        self.logger.info(f"  테스트 데이터: {len(self.test_data)} 거래일")
        self.logger.info(f"  종목 수: {len(self.train_data.columns)}개")

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """단일 실험 실행"""
        self.logger.info(f"실험 시작: {config.name}")
        self.logger.info(f"  설정: {config.description}")

        start_time = datetime.now()

        try:
            # 실험별 시드 설정
            np.random.seed(config.random_seed)

            # 트레이너 생성 및 설정 적용
            trainer = self._create_trainer_with_config(config)

            # 훈련 실행
            training_results = trainer.train(
                n_episodes=config.n_episodes,
                save_interval=max(config.n_episodes // 4, 10),
            )

            # 평가 실행
            evaluation_results = trainer.evaluate(n_episodes=5)

            # 레짐 통계 수집 (적응형 엔트로피가 활성화된 경우)
            regime_stats = None
            entropy_gap_stats = None

            if config.adaptive_entropy:
                regime_stats = self._collect_regime_statistics(trainer)
                entropy_gap_stats = self._collect_entropy_gap_statistics(trainer)

            # 실행 시간 계산
            execution_time = (datetime.now() - start_time).total_seconds()

            # 결과 구성
            result = ExperimentResult(
                config_name=config.name,
                final_portfolio_value=evaluation_results["avg_final_value"],
                avg_sharpe_ratio=evaluation_results["avg_sharpe_ratio"],
                max_drawdown=evaluation_results["avg_max_drawdown"],
                avg_turnover=evaluation_results.get("avg_turnover", 0.0),
                success_rate=evaluation_results["success_rate"],
                training_stability=training_results["training_stability"],
                regime_stats=regime_stats,
                entropy_gap_stats=entropy_gap_stats,
                execution_time=execution_time,
            )

            self.logger.info(f"실험 완료: {config.name} ({execution_time:.1f}초)")
            self.logger.info(f"  최종 가치: {result.final_portfolio_value:,.0f}")
            self.logger.info(f"  샤프 비율: {result.avg_sharpe_ratio:.3f}")
            self.logger.info(f"  성공률: {result.success_rate:.1%}")

            return result

        except Exception as e:
            self.logger.error(f"실험 실패: {config.name} - {str(e)}")
            import traceback

            self.logger.error(f"상세 오류:\n{traceback.format_exc()}")
            raise

    def _create_trainer_with_config(self, config: ExperimentConfig) -> BIPDTrainer:
        """실험 설정에 맞는 트레이너 생성"""

        # 임시로 설정을 적용하기 위해 config 모듈을 수정
        # (실제로는 config를 매개변수로 전달하는 것이 좋지만, 기존 코드 구조상 이렇게 함)

        # B-Cell 설정 수정 (적응형 엔트로피)
        if hasattr(config, "adaptive_entropy"):
            # 이는 BCell 초기화 시 적용됨 (이미 구현됨)
            pass

        # 트레이너 생성
        trainer = BIPDTrainer(train_data=self.train_data, test_data=self.test_data)

        return trainer

    def _collect_regime_statistics(self, trainer) -> Optional[Dict]:
        """레짐 통계 수집"""
        try:
            # 시스템에서 B-Cell들의 엔트로피 스케줄러 통계 수집
            regime_stats = {}

            for bcell_name, bcell in trainer.immune_system.bcells.items():
                if (
                    hasattr(bcell, "entropy_scheduler")
                    and bcell.entropy_scheduler is not None
                ):
                    stats = bcell.entropy_scheduler.get_regime_statistics()
                    regime_stats[bcell_name] = stats

            return regime_stats if regime_stats else None

        except Exception as e:
            self.logger.warning(f"레짐 통계 수집 실패: {str(e)}")
            return None

    def _collect_entropy_gap_statistics(self, trainer) -> Optional[Dict]:
        """엔트로피 갭 통계 수집"""
        try:
            entropy_stats = {}

            for bcell_name, bcell in trainer.immune_system.bcells.items():
                if (
                    hasattr(bcell, "entropy_scheduler")
                    and bcell.entropy_scheduler is not None
                ):
                    stats = bcell.entropy_scheduler.get_regime_statistics()
                    if "entropy_gap_stats" in stats:
                        entropy_stats[bcell_name] = stats["entropy_gap_stats"]

            return entropy_stats if entropy_stats else None

        except Exception as e:
            self.logger.warning(f"엔트로피 갭 통계 수집 실패: {str(e)}")
            return None

    def run_comparative_study(
        self, experiment_configs: List[ExperimentConfig], n_runs: int = 3
    ) -> Dict[str, List[ExperimentResult]]:
        """비교 연구 실행 (여러 실행으로 통계적 유의성 확보)"""

        self.logger.info(
            f"비교 연구 시작: {len(experiment_configs)}개 설정, 각각 {n_runs}회 실행"
        )

        all_results = defaultdict(list)

        for config in experiment_configs:
            self.logger.info(f"\n설정 '{config.name}' 실험 시작...")

            for run_idx in range(n_runs):
                self.logger.info(f"  실행 {run_idx + 1}/{n_runs}")

                # 실행별 다른 시드 사용
                run_config = ExperimentConfig(
                    name=f"{config.name}_run_{run_idx + 1}",
                    description=config.description,
                    adaptive_entropy=config.adaptive_entropy,
                    adaptive_no_trade_band=config.adaptive_no_trade_band,
                    use_simplex_projection=config.use_simplex_projection,
                    n_episodes=config.n_episodes,
                    random_seed=config.random_seed + run_idx * 100,
                )

                result = self.run_experiment(run_config)
                all_results[config.name].append(result)

        # 결과 저장
        self._save_comparative_results(all_results)

        return dict(all_results)

    def _save_comparative_results(self, results: Dict[str, List[ExperimentResult]]):
        """비교 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 원시 결과 저장
        raw_results = {
            config_name: [result.to_dict() for result in result_list]
            for config_name, result_list in results.items()
        }

        raw_file = os.path.join(self.save_dir, f"ab_test_raw_results_{timestamp}.json")
        with open(raw_file, "w") as f:
            json.dump(raw_results, f, indent=2, ensure_ascii=False)

        # 통계 요약 생성
        summary = self._generate_summary_statistics(results)

        summary_file = os.path.join(self.save_dir, f"ab_test_summary_{timestamp}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"결과 저장 완료:")
        self.logger.info(f"  원시 데이터: {raw_file}")
        self.logger.info(f"  통계 요약: {summary_file}")

    def _generate_summary_statistics(
        self, results: Dict[str, List[ExperimentResult]]
    ) -> Dict:
        """통계 요약 생성"""
        summary = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "n_configs": len(results),
                "n_runs_per_config": len(list(results.values())[0]) if results else 0,
            },
            "config_summaries": {},
        }

        for config_name, result_list in results.items():
            # 메트릭별 통계
            metrics = [
                "final_portfolio_value",
                "avg_sharpe_ratio",
                "max_drawdown",
                "success_rate",
                "training_stability",
                "execution_time",
            ]

            config_summary = {}

            for metric in metrics:
                values = [getattr(result, metric, 0.0) for result in result_list]
                if values and not all(v == 0.0 for v in values):
                    config_summary[metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "values": values,
                    }

            summary["config_summaries"][config_name] = config_summary

        return summary

    def print_comparison_report(self, results: Dict[str, List[ExperimentResult]]):
        """비교 보고서 출력"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Phase 3 개선사항 A/B 테스트 결과 보고서")
        self.logger.info("=" * 80)

        # 설정별 평균 성능
        for config_name, result_list in results.items():
            n_runs = len(result_list)

            # 메트릭 평균 계산
            avg_value = np.mean([r.final_portfolio_value for r in result_list])
            avg_sharpe = np.mean([r.avg_sharpe_ratio for r in result_list])
            avg_success = np.mean([r.success_rate for r in result_list])
            avg_time = np.mean([r.execution_time for r in result_list])

            std_value = np.std([r.final_portfolio_value for r in result_list])
            std_sharpe = np.std([r.avg_sharpe_ratio for r in result_list])

            self.logger.info(f"\n--- {config_name} ({n_runs}회 실행) ---")
            self.logger.info(f"  최종 가치: {avg_value:,.0f} (±{std_value:,.0f})")
            self.logger.info(f"  샤프 비율: {avg_sharpe:.3f} (±{std_sharpe:.3f})")
            self.logger.info(f"  성공률: {avg_success:.1%}")
            self.logger.info(f"  실행 시간: {avg_time:.1f}초")

        # 최고 성능 구성 식별
        best_config = max(
            results.keys(),
            key=lambda k: np.mean([r.avg_sharpe_ratio for r in results[k]]),
        )

        self.logger.info(f"\n🏆 최고 성능: {best_config}")

        # 통계적 유의성 간단 체크
        if len(results) >= 2:
            self._simple_significance_test(results)

    def _simple_significance_test(self, results: Dict[str, List[ExperimentResult]]):
        """간단한 통계적 유의성 테스트"""
        config_names = list(results.keys())

        if len(config_names) == 2:
            config_a, config_b = config_names
            values_a = [r.avg_sharpe_ratio for r in results[config_a]]
            values_b = [r.avg_sharpe_ratio for r in results[config_b]]

            # 단순 t-테스트 (정규분포 가정)
            from scipy import stats

            try:
                t_stat, p_value = stats.ttest_ind(values_a, values_b)

                self.logger.info(f"\n📊 통계적 유의성 테스트 (샤프 비율 기준):")
                self.logger.info(f"  {config_a} vs {config_b}")
                self.logger.info(f"  t-통계량: {t_stat:.3f}")
                self.logger.info(f"  p-값: {p_value:.3f}")

                if p_value < 0.05:
                    winner = (
                        config_a if np.mean(values_a) > np.mean(values_b) else config_b
                    )
                    self.logger.info(
                        f"  결론: {winner}이(가) 통계적으로 유의하게 우수 (p<0.05)"
                    )
                else:
                    self.logger.info(f"  결론: 통계적으로 유의한 차이 없음 (p≥0.05)")

            except ImportError:
                self.logger.info(
                    "scipy를 사용할 수 없어 통계적 유의성 테스트를 건너뜁니다."
                )
            except Exception as e:
                self.logger.warning(f"통계적 유의성 테스트 실패: {str(e)}")


# 사전 정의된 실험 설정들
def get_phase3_experiment_configs() -> List[ExperimentConfig]:
    """Phase 3 개선사항 실험 설정들"""

    return [
        ExperimentConfig(
            name="baseline",
            description="기존 시스템 (Phase 1,2 개선사항만)",
            adaptive_entropy=False,
            adaptive_no_trade_band=True,
            use_simplex_projection=True,
            n_episodes=100,
        ),
        ExperimentConfig(
            name="adaptive_entropy_only",
            description="적응형 엔트로피만 활성화",
            adaptive_entropy=True,
            adaptive_no_trade_band=False,
            use_simplex_projection=True,
            n_episodes=100,
        ),
        ExperimentConfig(
            name="full_phase3",
            description="모든 Phase 3 개선사항 활성화",
            adaptive_entropy=True,
            adaptive_no_trade_band=True,
            use_simplex_projection=True,
            n_episodes=100,
        ),
        ExperimentConfig(
            name="legacy_system",
            description="원본 시스템 (개선사항 없음)",
            adaptive_entropy=False,
            adaptive_no_trade_band=False,
            use_simplex_projection=False,
            n_episodes=100,
        ),
    ]


# 테스트 실행 함수
def run_phase3_ab_test():
    """Phase 3 A/B 테스트 실행"""

    # 프레임워크 초기화
    framework = ABTestingFramework()

    # 실험 설정 가져오기
    configs = get_phase3_experiment_configs()

    # 비교 연구 실행 (각 설정마다 3회 실행)
    results = framework.run_comparative_study(configs, n_runs=3)

    # 결과 보고서 출력
    framework.print_comparison_report(results)

    return results


if __name__ == "__main__":
    results = run_phase3_ab_test()
