# tests/test_stability.py

import pytest
import numpy as np
from unittest.mock import Mock
from src.utils.monitoring import StabilityMonitor

class TestStabilityMonitor:
    """StabilityMonitor 테스트"""
    
    @pytest.fixture
    def config(self):
        return {
            'max_weight_change': 0.2,
            'min_effective_assets': 3,
            'n_sigma': 3.0,
            'intervention_threshold': 3,
            'rollback_window': 100,
            'rollback_enabled': True
        }
    
    @pytest.fixture
    def monitor(self, config):
        return StabilityMonitor(config)
    
    def test_initialization(self, monitor, config):
        assert monitor.max_weight_change == config['max_weight_change']
        assert monitor.min_effective_assets == config['min_effective_assets']
        assert monitor.n_sigma == config['n_sigma']
        assert len(monitor.history['q_value']) == 0
        assert monitor.alert_count == 0
    
    def test_push_normal_metrics(self, monitor):
        """정상 메트릭 푸시"""
        metrics = {
            'q_value': 10.0,
            'entropy': 2.5,
            'actor_loss': 0.1,
            'critic_loss': 0.2,
            'weight_change': 0.1,
            'effective_assets': 5
        }
        
        monitor.push(metrics)
        
        assert len(monitor.history['q_value']) == 1
        assert monitor.history['q_value'][0] == 10.0
    
    def test_push_invalid_metrics(self, monitor):
        """무효한 메트릭 거부"""
        metrics = {
            'q_value': np.nan,  # NaN
            'entropy': 2.5,
            'actor_loss': 0.1,
            'critic_loss': 0.2,
            'weight_change': 0.1,
            'effective_assets': 5
        }
        
        with pytest.raises(AssertionError):
            monitor.push(metrics)
    
    def test_check_no_alerts(self, monitor):
        """정상 상태에서 알림 없음"""
        # 정상 메트릭 추가
        for i in range(10):
            metrics = {
                'q_value': 10.0 + np.random.randn() * 0.1,
                'entropy': 2.5 + np.random.randn() * 0.1,
                'actor_loss': 0.1,
                'critic_loss': 0.2,
                'weight_change': 0.05,
                'effective_assets': 10
            }
            monitor.push(metrics)
        
        alerts = monitor.check()
        
        assert alerts['q_value_outlier'] == False
        assert alerts['entropy_low'] == False
        assert alerts['weight_instability'] == False
        assert alerts['severity'] == 'none'
    
    def test_check_q_value_outlier(self, monitor):
        """Q-value 이상치 감지"""
        # 정상 값들
        for i in range(20):
            metrics = {
                'q_value': 10.0 + np.random.randn() * 0.5,
                'entropy': 2.5,
                'actor_loss': 0.1,
                'critic_loss': 0.2,
                'weight_change': 0.05,
                'effective_assets': 10
            }
            monitor.push(metrics)
        
        # 이상치
        metrics = {
            'q_value': 100.0,  # 극단적 값
            'entropy': 2.5,
            'actor_loss': 0.1,
            'critic_loss': 0.2,
            'weight_change': 0.05,
            'effective_assets': 10
        }
        monitor.push(metrics)
        
        alerts = monitor.check()
        assert alerts['q_value_outlier'] == True
    
    def test_check_entropy_low(self, monitor):
        """낮은 엔트로피 감지"""
        metrics = {
            'q_value': 10.0,
            'entropy': 0.01,  # 매우 낮은 엔트로피
            'actor_loss': 0.1,
            'critic_loss': 0.2,
            'weight_change': 0.05,
            'effective_assets': 10
        }
        monitor.push(metrics)
        
        alerts = monitor.check()
        assert alerts['entropy_low'] == True
    
    def test_check_weight_instability(self, monitor):
        """가중치 불안정성 감지"""
        metrics = {
            'q_value': 10.0,
            'entropy': 2.5,
            'actor_loss': 0.1,
            'critic_loss': 0.2,
            'weight_change': 0.5,  # 큰 변화
            'effective_assets': 10
        }
        monitor.push(metrics)
        
        alerts = monitor.check()
        assert alerts['weight_instability'] == True
    
    def test_check_concentration_risk(self, monitor):
        """집중 리스크 감지"""
        metrics = {
            'q_value': 10.0,
            'entropy': 2.5,
            'actor_loss': 0.1,
            'critic_loss': 0.2,
            'weight_change': 0.05,
            'effective_assets': 2  # 너무 적은 자산
        }
        monitor.push(metrics)
        
        alerts = monitor.check()
        assert alerts['concentration_risk'] == True
    
    def test_severity_levels(self, monitor):
        """심각도 수준 테스트"""
        # 경고 수준
        metrics_warning = {
            'q_value': 10.0,
            'entropy': 0.5,  # 낮음
            'actor_loss': 0.1,
            'critic_loss': 0.2,
            'weight_change': 0.05,
            'effective_assets': 10
        }
        monitor.push(metrics_warning)
        alerts = monitor.check()
        assert alerts['severity'] == 'warning'
        
        # 위험 수준 (여러 문제)
        monitor.alert_count = 0  # 리셋
        metrics_critical = {
            'q_value': 100.0,  # 이상치
            'entropy': 0.01,   # 매우 낮음
            'actor_loss': 10.0,  # 높음
            'critic_loss': 10.0,  # 높음
            'weight_change': 0.5,  # 불안정
            'effective_assets': 1  # 집중
        }
        monitor.push(metrics_critical)
        alerts = monitor.check()
        assert alerts['severity'] == 'critical'
    
    def test_intervene(self, monitor):
        """개입 테스트"""
        # Mock trainer
        trainer = Mock()
        trainer.bcell = Mock()
        trainer.bcell.alpha = 0.2
        trainer.bcell.actor_optimizer = Mock()
        trainer.bcell.critic_optimizer = Mock()
        trainer.config = {'bcell': {'cql_alpha_start': 0.01}}
        trainer.cql_alpha = 0.01
        
        # 개입 실행
        monitor.intervene(trainer)
        
        # 학습률 감소 확인
        assert trainer.bcell.actor_optimizer.param_groups[0]['lr'] < 1e-3
        
        # CQL alpha 증가 확인
        assert trainer.cql_alpha > 0.01
        
        # 엔트로피 증가 확인
        assert trainer.bcell.alpha > 0.2
    
    def test_checkpoint_management(self, monitor):
        """체크포인트 관리 테스트"""
        # Mock trainer
        trainer = Mock()
        trainer.bcell = Mock()
        trainer.step = 0
        
        # 체크포인트 저장
        checkpoint = {
            'step': 100,
            'bcell_state': 'mock_state',
            'metrics': {'sharpe': 1.5}
        }
        
        monitor.checkpoints.append(checkpoint)
        
        assert len(monitor.checkpoints) == 1
        
        # 윈도우 크기 초과 시 제거
        for i in range(15):
            monitor.checkpoints.append({
                'step': 100 + i * 100,
                'bcell_state': f'state_{i}',
                'metrics': {}
            })
        
        # 최대 10개 유지
        assert len(monitor.checkpoints) <= 10
    
    def test_get_statistics(self, monitor):
        """통계 반환 테스트"""
        # 데이터 추가
        for i in range(50):
            metrics = {
                'q_value': 10.0 + np.random.randn(),
                'entropy': 2.5 + np.random.randn() * 0.1,
                'actor_loss': 0.1 + np.random.randn() * 0.01,
                'critic_loss': 0.2 + np.random.randn() * 0.01,
                'weight_change': 0.05,
                'effective_assets': 10
            }
            monitor.push(metrics)
        
        stats = monitor.get_statistics()
        
        assert 'q_value' in stats
        assert 'mean' in stats['q_value']
        assert 'std' in stats['q_value']
        assert 'min' in stats['q_value']
        assert 'max' in stats['q_value']
        
        assert stats['total_alerts'] >= 0
        assert stats['total_interventions'] >= 0
    
    def test_reset(self, monitor):
        """리셋 테스트"""
        # 데이터 추가
        for i in range(10):
            metrics = {
                'q_value': 10.0,
                'entropy': 2.5,
                'actor_loss': 0.1,
                'critic_loss': 0.2,
                'weight_change': 0.05,
                'effective_assets': 10
            }
            monitor.push(metrics)
        
        monitor.alert_count = 5
        monitor.intervention_count = 2
        
        # 리셋
        monitor.reset()
        
        assert len(monitor.history['q_value']) == 0
        assert monitor.alert_count == 0
        assert monitor.intervention_count == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])