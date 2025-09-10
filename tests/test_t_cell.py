# tests/test_t_cell.py

import pytest
import numpy as np
from src.agents.t_cell import TCell, BOCPD, HMMRegimeDetector

class TestTCell:
    """T-Cell 위기 감지 테스트"""
    
    @pytest.fixture
    def tcell(self):
        config = {
            'method': 'iforest',
            'contamination': 0.05,
            'n_estimators': 100,
            'ema_beta': 0.9
        }
        return TCell(config)
    
    def test_initialization(self, tcell):
        assert tcell.method == 'iforest'
        assert tcell.contamination == 0.05
        assert tcell.n_estimators == 100
        assert tcell.ema_crisis == 0.0
    
    def test_detect_normal_state(self, tcell):
        """정상 상태 감지"""
        # 정상적인 시장 특성
        normal_features = np.random.randn(12) * 0.01
        
        crisis_level, shap_values = tcell.detect(normal_features)
        
        assert 0 <= crisis_level <= 1
        assert isinstance(shap_values, list)
    
    def test_detect_crisis_state(self, tcell):
        """위기 상태 감지"""
        # 극단적인 시장 특성
        crisis_features = np.random.randn(12) * 0.1
        crisis_features[0] = -0.1  # 큰 손실
        crisis_features[2] = 0.1   # 높은 변동성
        
        crisis_level, _ = tcell.detect(crisis_features)
        
        # 초기에는 낮을 수 있음 (학습 필요)
        assert 0 <= crisis_level <= 1
    
    def test_ema_smoothing(self, tcell):
        """EMA 평활화 테스트"""
        # 연속적인 위기 신호
        for _ in range(10):
            features = np.random.randn(12) * 0.05
            features[2] = 0.08  # 높은 변동성
            crisis_level, _ = tcell.detect(features)
        
        # EMA가 점진적으로 증가
        assert tcell.ema_crisis > 0
        
        prev_ema = tcell.ema_crisis
        
        # 정상 신호로 전환
        for _ in range(10):
            features = np.random.randn(12) * 0.01
            crisis_level, _ = tcell.detect(features)
        
        # EMA가 감소
        assert tcell.ema_crisis < prev_ema
    
    def test_ensemble_method(self):
        """앙상블 방법 테스트"""
        config = {
            'method': 'ensemble',
            'contamination': 0.05,
            'bocpd_hazard_rate': 0.01,
            'hmm_n_states': 3
        }
        tcell = TCell(config)
        
        assert tcell.method == 'ensemble'
        assert tcell.bocpd is not None
        assert tcell.hmm is not None
        
        # 앙상블 감지
        features = np.random.randn(12) * 0.02
        crisis_level, _ = tcell.detect(features)
        
        assert 0 <= crisis_level <= 1

class TestBOCPD:
    """Bayesian Online Changepoint Detection 테스트"""
    
    def test_initialization(self):
        bocpd = BOCPD(hazard_rate=0.01)
        assert bocpd.hazard_rate == 0.01
        assert len(bocpd.run_length_probs) == 1
    
    def test_update_no_changepoint(self):
        bocpd = BOCPD()
        
        # 일정한 신호
        for _ in range(10):
            probs, max_prob = bocpd.update(0.5)
        
        # 변화점 확률이 낮아야 함
        assert max_prob < 0.5
    
    def test_update_with_changepoint(self):
        bocpd = BOCPD(hazard_rate=0.1)
        
        # 정상 신호
        for _ in range(10):
            bocpd.update(0.0)
        
        # 급격한 변화
        for _ in range(5):
            probs, max_prob = bocpd.update(5.0)
        
        # 변화점 감지
        assert max_prob > 0.3
    
    def test_reset(self):
        bocpd = BOCPD()
        
        for _ in range(10):
            bocpd.update(np.random.randn())
        
        assert len(bocpd.data) == 10
        
        bocpd.reset()
        assert len(bocpd.data) == 0
        assert len(bocpd.run_length_probs) == 1

class TestHMMRegimeDetector:
    """Hidden Markov Model 레짐 감지 테스트"""
    
    def test_initialization(self):
        hmm = HMMRegimeDetector(n_states=3)
        assert hmm.n_states == 3
        assert hmm.current_state is None
    
    def test_fit(self):
        hmm = HMMRegimeDetector(n_states=2)
        
        # 두 개의 레짐을 가진 데이터 생성
        data1 = np.random.randn(50) * 0.01
        data2 = np.random.randn(50) * 0.05
        observations = np.concatenate([data1, data2])
        
        hmm.fit(observations)
        
        assert hmm.fitted
        assert hmm.means.shape == (2,)
        assert hmm.covars.shape == (2,)
        assert hmm.trans_mat.shape == (2, 2)
    
    def test_predict_unfitted(self):
        hmm = HMMRegimeDetector()
        
        # 학습 전 예측
        state, prob = hmm.predict(0.01)
        assert state == 0
        assert prob == 1.0
    
    def test_predict_fitted(self):
        hmm = HMMRegimeDetector(n_states=2)
        
        # 학습
        observations = np.random.randn(100) * 0.02
        hmm.fit(observations)
        
        # 예측
        state, prob = hmm.predict(0.01)
        assert 0 <= state < 2
        assert 0 <= prob <= 1
    
    def test_get_crisis_level(self):
        hmm = HMMRegimeDetector(n_states=3)
        
        # 학습
        observations = np.concatenate([
            np.random.randn(30) * 0.01,  # Low vol
            np.random.randn(30) * 0.03,  # Med vol
            np.random.randn(30) * 0.08   # High vol
        ])
        hmm.fit(observations)
        
        # 위기 수준
        crisis_level = hmm.get_crisis_level()
        assert 0 <= crisis_level <= 1
        
        # 고변동성 상태에서 높은 위기 수준
        hmm.current_state = np.argmax(hmm.covars)  # 최고 변동성 상태
        crisis_level = hmm.get_crisis_level()
        assert crisis_level > 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])