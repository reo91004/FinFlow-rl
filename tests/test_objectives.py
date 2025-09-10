# tests/test_objectives.py

import pytest
import numpy as np
from src.core.objectives import DifferentialSharpe, CVaRConstraint, TurnoverPenalty

class TestDifferentialSharpe:
    """Differential Sharpe Ratio 테스트"""
    
    def test_initialization(self):
        ds = DifferentialSharpe(beta=0.5, epsilon=1e-8)
        assert ds.beta == 0.5
        assert ds.epsilon == 1e-8
        assert ds.A == 0
        assert ds.B == 0
    
    def test_update(self):
        ds = DifferentialSharpe()
        
        # 양의 수익률
        sharpe1 = ds.update(0.01)
        assert sharpe1 > 0
        assert ds.A > 0
        assert ds.B > 0
        
        # 연속 업데이트
        sharpe2 = ds.update(0.02)
        assert sharpe2 != sharpe1
        
        # 음의 수익률
        sharpe3 = ds.update(-0.01)
        assert sharpe3 < sharpe2
    
    def test_convergence(self):
        """수렴 테스트: 일정한 수익률에서 안정화"""
        ds = DifferentialSharpe(beta=0.95)
        
        sharpe_values = []
        for _ in range(100):
            sharpe = ds.update(0.01)
            sharpe_values.append(sharpe)
        
        # 마지막 10개 값의 변동이 작아야 함
        last_10 = sharpe_values[-10:]
        variance = np.var(last_10)
        assert variance < 0.001
    
    def test_reset(self):
        ds = DifferentialSharpe()
        ds.update(0.01)
        ds.update(0.02)
        
        assert ds.A > 0
        assert ds.B > 0
        
        ds.reset()
        assert ds.A == 0
        assert ds.B == 0

class TestCVaRConstraint:
    """CVaR 제약 테스트"""
    
    def test_initialization(self):
        cvar = CVaRConstraint(alpha=0.05, epsilon=1e-8)
        assert cvar.alpha == 0.05
        assert len(cvar.returns_buffer) == 0
    
    def test_compute_cvar_empty(self):
        cvar = CVaRConstraint()
        result = cvar.compute_cvar(0.01)
        assert result == 0.01  # 버퍼가 비어있을 때는 현재 수익률 반환
    
    def test_compute_cvar_with_data(self):
        cvar = CVaRConstraint(alpha=0.05)
        
        # 데이터 추가
        returns = np.random.randn(100) * 0.01
        for r in returns:
            cvar.compute_cvar(r)
        
        # CVaR는 하위 5% 평균이므로 음수여야 함 (대부분의 경우)
        final_cvar = cvar.compute_cvar(-0.05)
        assert final_cvar < 0
    
    def test_buffer_limit(self):
        cvar = CVaRConstraint()
        
        # 버퍼 한계 테스트
        for i in range(1500):
            cvar.compute_cvar(i * 0.001)
        
        assert len(cvar.returns_buffer) == 1000  # maxlen=1000
    
    def test_violation_penalty(self):
        cvar = CVaRConstraint(alpha=0.05)
        
        # 충분한 데이터 추가
        for _ in range(100):
            cvar.compute_cvar(-0.02)
        
        # 큰 손실에 대한 CVaR
        cvar_value = cvar.compute_cvar(-0.1)
        
        # 타겟보다 나쁘면 페널티
        target = -0.02
        penalty = cvar.get_penalty(cvar_value, target)
        
        if cvar_value < target:
            assert penalty < 0
        else:
            assert penalty == 0

class TestTurnoverPenalty:
    """턴오버 페널티 테스트"""
    
    def test_initialization(self):
        tp = TurnoverPenalty(cost=0.001)
        assert tp.cost == 0.001
    
    def test_compute_zero_turnover(self):
        tp = TurnoverPenalty()
        
        # 동일한 포트폴리오
        old_weights = np.array([0.5, 0.3, 0.2])
        new_weights = np.array([0.5, 0.3, 0.2])
        
        turnover = tp.compute(old_weights, new_weights)
        assert turnover == 0
    
    def test_compute_full_turnover(self):
        tp = TurnoverPenalty(cost=0.001)
        
        # 완전히 다른 포트폴리오
        old_weights = np.array([1.0, 0.0, 0.0])
        new_weights = np.array([0.0, 0.0, 1.0])
        
        turnover = tp.compute(old_weights, new_weights)
        assert turnover == pytest.approx(-0.002, rel=1e-5)  # -2 * 0.001
    
    def test_compute_partial_turnover(self):
        tp = TurnoverPenalty(cost=0.002)
        
        old_weights = np.array([0.4, 0.3, 0.3])
        new_weights = np.array([0.5, 0.2, 0.3])
        
        turnover = tp.compute(old_weights, new_weights)
        expected = -np.sum(np.abs(new_weights - old_weights)) * 0.002
        assert turnover == pytest.approx(expected, rel=1e-5)
    
    def test_different_dimensions(self):
        """차원이 다른 경우 처리"""
        tp = TurnoverPenalty()
        
        old_weights = np.array([0.5, 0.5])
        new_weights = np.array([0.3, 0.3, 0.4])
        
        # 패딩 후 계산
        turnover = tp.compute(old_weights, new_weights)
        assert turnover < 0  # 페널티 존재

if __name__ == "__main__":
    pytest.main([__file__, "-v"])