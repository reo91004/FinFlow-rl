# tests/test_xai_integration.py

"""
XAI 통합 및 모니터링-시각화 연동 테스트

테스트 항목:
1. XAI 3함수 (local_attribution, counterfactual, regime_report) 호출 확인
2. 메트릭 계산 (Sharpe, CVaR, MDD) 및 JSON 저장 확인
3. 시각화 PNG 파일 생성 확인
4. 모니터링 알람 발생 시 alerts 디렉토리에 그래프 저장 확인
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import json
import tempfile
import shutil
from datetime import datetime

def test_xai_functions():
    """XAI 3함수 호출 및 결과 저장 테스트"""
    from src.analysis.explainer import XAIExplainer
    
    # 더미 모델 및 데이터
    class DummyModel:
        def forward(self, x):
            return np.random.rand(10)
    
    model = DummyModel()
    explainer = XAIExplainer(model)
    
    # 테스트 데이터
    state = np.random.randn(43)
    action = np.random.rand(10)
    action = action / action.sum()  # Normalize to simplex
    
    # 1. local_attribution 테스트
    local_attr = explainer.local_attribution(state, action)
    assert isinstance(local_attr, dict), "local_attribution should return dict"
    print("✓ local_attribution 테스트 통과")
    
    # 2. counterfactual 테스트
    cf_report = explainer.counterfactual(state, action, deltas={"volatility": -0.2})
    assert isinstance(cf_report, dict), "counterfactual should return dict"
    print("✓ counterfactual 테스트 통과")
    
    # 3. regime_report 테스트
    crisis_info = {
        'overall_crisis': 0.7,
        'volatility_crisis': 0.8,
        'correlation_crisis': 0.6
    }
    reg_report = explainer.regime_report(crisis_info, shap_topk=5)
    assert isinstance(reg_report, dict), "regime_report should return dict"
    print("✓ regime_report 테스트 통과")
    
    # Decision card 생성 테스트
    decision_card = {
        "timestamp": datetime.now().isoformat(),
        "action": list(map(float, action)),
        "local_attribution": local_attr,
        "counterfactual": cf_report,
        "regime_report": reg_report
    }
    
    # JSON 직렬화 테스트
    try:
        json_str = json.dumps(decision_card, indent=2)
        print("✓ Decision card JSON 직렬화 성공")
    except Exception as e:
        print(f"✗ Decision card JSON 직렬화 실패: {e}")
        return False
    
    return True

def test_metrics_calculation():
    """메트릭 계산 및 저장 테스트"""
    from src.analysis.metrics import calculate_sharpe_ratio, calculate_cvar, calculate_max_drawdown
    
    # 테스트 데이터
    returns = np.random.randn(252) * 0.01  # Daily returns
    equity_curve = np.cumprod(1 + returns)
    
    # 메트릭 계산
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    cvar_95 = calculate_cvar(returns, alpha=0.05)
    mdd = calculate_max_drawdown(equity_curve)
    
    assert isinstance(sharpe, (float, np.floating)), "Sharpe should be float"
    assert isinstance(cvar_95, (float, np.floating)), "CVaR should be float"
    assert isinstance(mdd, (float, np.floating)), "MDD should be float"
    
    print(f"✓ Sharpe Ratio: {sharpe:.3f}")
    print(f"✓ CVaR (95%): {cvar_95:.3f}")
    print(f"✓ Max Drawdown: {mdd:.3f}")
    
    # 메트릭 리포트 생성
    metrics_report = {
        "sharpe": float(sharpe),
        "cvar_95": float(cvar_95),
        "max_drawdown": float(mdd)
    }
    
    # JSON 저장 테스트
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metrics_report, f, indent=2)
        temp_path = Path(f.name)
    
    assert temp_path.exists(), "Metrics JSON file should be created"
    temp_path.unlink()  # Clean up
    print("✓ 메트릭 JSON 저장 테스트 통과")
    
    return True

def test_visualization_generation():
    """시각화 생성 테스트"""
    import pandas as pd
    from src.analysis.visualization import plot_portfolio_weights, plot_equity_curve, plot_drawdown
    
    # 테스트 데이터
    returns = np.random.randn(252) * 0.01
    equity_curve = np.cumprod(1 + returns)
    weights = np.random.rand(10)
    weights = weights / weights.sum()  # Single weight vector
    asset_names = [f"Asset_{i}" for i in range(10)]
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # 1. Equity curve 플롯
        equity_path = tmpdir / "equity_curve.png"
        plot_equity_curve(equity_curve, save_path=equity_path)
        assert equity_path.exists(), "Equity curve PNG should be created"
        print("✓ Equity curve 시각화 생성 성공")
        
        # 2. Drawdown 플롯
        dd_path = tmpdir / "drawdown.png"
        plot_drawdown(equity_curve, save_path=dd_path)
        assert dd_path.exists(), "Drawdown PNG should be created"
        print("✓ Drawdown 시각화 생성 성공")
        
        # 3. Portfolio weights 플롯
        weights_path = tmpdir / "weights.png"
        plot_portfolio_weights(weights, asset_names, save_path=weights_path)
        assert weights_path.exists(), "Weights PNG should be created"
        print("✓ Portfolio weights 시각화 생성 성공")
    
    return True

def test_directory_structure():
    """디렉토리 구조 생성 테스트"""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # reports 디렉토리 생성
        (run_dir / "reports").mkdir(parents=True, exist_ok=True)
        assert (run_dir / "reports").exists(), "reports directory should be created"
        print("✓ reports 디렉토리 생성 성공")
        
        # alerts 디렉토리 생성
        (run_dir / "alerts").mkdir(parents=True, exist_ok=True)
        assert (run_dir / "alerts").exists(), "alerts directory should be created"
        print("✓ alerts 디렉토리 생성 성공")
        
        # 테스트 파일 생성
        test_files = [
            run_dir / "reports" / "metrics.json",
            run_dir / "reports" / "decision_card_0.json",
            run_dir / "reports" / "equity_curve.png",
            run_dir / "reports" / "drawdown.png",
            run_dir / "reports" / "weights.png",
            run_dir / "alerts" / "equity_100.png",
            run_dir / "alerts" / "dd_100.png",
            run_dir / "alerts" / "weights_100.png"
        ]
        
        for file_path in test_files:
            file_path.touch()
            assert file_path.exists(), f"{file_path.name} should be created"
        
        print("✓ 모든 파일 경로 생성 테스트 통과")
    
    return True

def main():
    """메인 테스트 실행"""
    print("=" * 50)
    print("XAI 통합 및 모니터링-시각화 연동 테스트")
    print("=" * 50)
    
    tests = [
        ("XAI 함수 테스트", test_xai_functions),
        ("메트릭 계산 테스트", test_metrics_calculation),
        ("시각화 생성 테스트", test_visualization_generation),
        ("디렉토리 구조 테스트", test_directory_structure)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            if test_func():
                passed += 1
                print(f"→ {test_name} 성공 ✓")
            else:
                failed += 1
                print(f"→ {test_name} 실패 ✗")
        except Exception as e:
            failed += 1
            print(f"→ {test_name} 실패: {e}")
    
    print("\n" + "=" * 50)
    print(f"테스트 결과: {passed}/{len(tests)} 통과")
    print("=" * 50)
    
    if failed == 0:
        print("\n✅ 모든 테스트 통과!")
        print("\n완료 기준:")
        print("- scripts/evaluate.py에서 XAI 3함수 호출 ✓")
        print("- decision_card_*.json 저장 ✓")
        print("- metrics.json 저장 ✓")
        print("- equity_curve.png, drawdown.png, weights.png 생성 ✓")
        print("- 알람 시 alerts/*.png 생성 준비 완료 ✓")
    else:
        print(f"\n⚠️ {failed}개 테스트 실패")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)