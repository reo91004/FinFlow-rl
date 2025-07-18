# main.py

import numpy as np
import torch
import time
from core import ImmunePortfolioBacktester
from constant import create_directories

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
    USE_FIXED_SEED = False  # True: 재현 가능한 결과, False: 매번 다른 결과

    if USE_FIXED_SEED:
        global_seed = 42
        print(f"[설정] 고정 시드 사용: {global_seed} (재현 가능한 결과)")
    else:
        global_seed = int(time.time()) % 10000
        print(f"[설정] 랜덤 시드 사용: {global_seed} (매번 다른 결과)")

    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

    # 백테스터 초기화
    backtester = ImmunePortfolioBacktester(
        symbols, train_start, train_end, test_start, test_end
    )

    print("\n" + "=" * 60)
    print(" BIPD (Behavioral Immune Portfolio Defense) 시스템 성능 평가")
    print("=" * 60)

    try:
        # 백테스트 실행 (전역 시드 사용)
        portfolio_returns, immune_system = backtester.backtest_single_run(
            seed=global_seed,
            return_model=True,
            use_learning_bcells=True,
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

        # 분석 결과 저장
        print(f"\n=== 분석 결과 저장 중 ===")

        # 통합 분석 (의사결정 분석 + 전문성 분석)
        json_path, md_path = backtester.save_comprehensive_analysis(
            "2021-01-01", "2021-06-30"
        )

        # 분석 결과 저장 (HTML 대시보드 + 면역 시스템 시각화)
        analysis_json, analysis_md, dashboard_html = backtester.save_analysis_results(
            "2021-01-01", "2021-06-30"
        )

        print(f"\n=== BIPD 시스템 성능 평가 완료 ===")

        # 다중 실행 성능 검증 (다양한 시드로 안정성 확인)
        print(f"\n=== 다중 실행 안정성 검증 ===")
        results = backtester.run_multiple_backtests(
            n_runs=3,
            save_results=True,
            use_learning_bcells=True,
            logging_level="sample",
            base_seed=global_seed,
        )

    except Exception as e:
        print(f"\n[오류] 주요 실행 실패: {e}")
        import traceback

        traceback.print_exc()

        # 폴백 모드: 기본 백테스트 (최소 로깅으로 성능 확보)
        basic_results = backtester.run_multiple_backtests(
            n_runs=1,
            save_results=True,
            use_learning_bcells=True,
            logging_level="minimal",
            base_seed=global_seed,
        )
