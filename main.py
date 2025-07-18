# main.py

import numpy as np
import torch
import time
import traceback
import gc
from core import ImmunePortfolioBacktester
from constant import create_directories
from utils.logger import stop_logging

# 디렉토리 초기화
create_directories()

# 실행
if __name__ == "__main__":
    # 설정
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "AMD", "TSLA", "JPM", "JNJ", "PG", "V"]
    train_start = "2008-01-02"
    train_end = "2020-12-31"
    test_start = "2021-01-01"
    test_end = "2024-12-31"

    # 시드 설정 옵션
    USE_FIXED_SEED = False
    ENABLE_ALL_FEATURES = True  # 모든 기능 활성화

    if USE_FIXED_SEED:
        global_seed = 42
        print(f"[설정] 고정 시드 사용: {global_seed} (재현 가능한 결과)")
    else:
        global_seed = int(time.time()) % 10000
        print(f"[설정] 랜덤 시드 사용: {global_seed} (매번 다른 결과)")

    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

    # 백테스터 초기화 (로깅은 백테스터 내부에서 자동 초기화)
    backtester = ImmunePortfolioBacktester(
        symbols, train_start, train_end, test_start, test_end
    )

    print("\n" + "=" * 80)
    print(" BIPD (Behavioral Immune Portfolio Defense) 통합 시스템 성능 평가")
    print("=" * 80)

    if ENABLE_ALL_FEATURES:
        print("\n[활성화된 기능]")
        print("- Actor-Critic 기반 B-Cell 네트워크")
        print("- T-Cell to B-Cell 어텐션 메커니즘")
        print("- 고도화된 보상 함수 (샤프지수, 거래비용, 목표기반)")
        print("- 기억 기반 의사결정 강화")
        print("- 계층적 강화학습 (Meta-Controller)")
        print("- 커리큘럼 학습 (3단계 난이도)")
        print("- XAI 기반 설명 가능성")
        print("- 체크포인트 자동 관리")
        print("- 데이터 검증 시스템")

    try:
        # 통합 백테스트 실행
        portfolio_returns, immune_system = backtester.backtest_single_run(
            seed=global_seed,
            return_model=True,
            use_learning_bcells=ENABLE_ALL_FEATURES,
            use_hierarchical=ENABLE_ALL_FEATURES,
            use_curriculum=ENABLE_ALL_FEATURES,
            logging_level="full",
        )

        # 시스템 저장
        backtester.immune_system = immune_system

        # 성과 계산
        metrics = backtester.calculate_metrics(portfolio_returns)
        print(f"\n=== 포트폴리오 성과 요약 ===")
        print(f"총 수익률: {metrics['Total Return']:.2%}")
        print(f"샤프 지수: {metrics['Sharpe Ratio']:.2f}")
        print(f"최대 낙폭: {metrics['Max Drawdown']:.2%}")
        print(f"변동성: {metrics['Volatility']:.3f}")

        # 통합 분석 결과 저장
        print(f"\n=== 통합 분석 결과 저장 중 ===")

        # 포괄적 분석 (모든 메트릭 포함)
        json_path, md_path = backtester.save_comprehensive_analysis(
            "2021-01-01", "2021-06-30"
        )

        # XAI 대시보드 및 시각화
        analysis_json, analysis_md, dashboard_html = backtester.save_analysis_results(
            "2021-01-01", "2021-06-30"
        )

        print(f"\n=== BIPD 통합 시스템 성능 평가 완료 ===")

        # 안정성 검증을 위한 다중 실행
        print(f"\n=== 다중 실행 안정성 검증 ===")
        results = backtester.run_multiple_backtests(
            n_runs=3,
            save_results=True,
            use_learning_bcells=ENABLE_ALL_FEATURES,
            use_hierarchical=ENABLE_ALL_FEATURES,
            use_curriculum=ENABLE_ALL_FEATURES,
            logging_level="sample",
            base_seed=global_seed,
        )

        print(f"\n=== 최종 성과 요약 ===")
        print(f"평균 샤프 지수: {results['Sharpe Ratio'].mean():.3f}")
        print(f"샤프 지수 표준편차: {results['Sharpe Ratio'].std():.3f}")
        print(f"평균 최대 낙폭: {results['Max Drawdown'].mean():.2%}")

    except KeyboardInterrupt:
        print("\n사용자에 의한 중단")
    except Exception as e:
        print(f"\n[오류] 주요 실행 실패: {e}")
        print(f"스택 트레이스:\n{traceback.format_exc()}")

        # 폴백 모드: 기본 기능만 활성화
        print(f"\n[폴백] 기본 기능으로 재시도...")
        try:
            basic_results = backtester.run_multiple_backtests(
                n_runs=1,
                save_results=True,
                use_learning_bcells=True,
                use_hierarchical=False,
                use_curriculum=False,
                logging_level="minimal",
                base_seed=global_seed,
            )
            print("폴백 모드 실행 완료")
        except Exception as fallback_error:
            print(f"폴백 모드도 실패: {fallback_error}")

    finally:
        # 시스템 정리
        print("\n시스템 정리 중...")

        # 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 로깅 시스템 정리
        if hasattr(backtester, "tee_output") and backtester.tee_output:
            stop_logging(backtester.tee_output)

        print("정리 완료")
