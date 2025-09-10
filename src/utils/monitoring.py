# src/utils/monitoring.py

"""
StabilityMonitor: 실시간 학습 안정성 모니터링 및 자동 개입 시스템

연구용 코드로 try-except 없이 assert로 명확한 검증
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass, field
import time
from src.utils.logger import FinFlowLogger

@dataclass
class MetricHistory:
    """메트릭 히스토리 관리"""
    window_size: int = 100
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add(self, value: float):
        """값 추가"""
        assert np.isfinite(value), f"Non-finite value: {value}"
        self.values.append(value)
    
    def get_stats(self) -> Dict[str, float]:
        """통계 계산"""
        if len(self.values) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        values_array = np.array(self.values)
        return {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'recent': float(self.values[-1]) if self.values else 0
        }
    
    def detect_anomaly(self, n_sigma: float = 3.0) -> Tuple[bool, str]:
        """이상치 감지 (n-sigma 규칙)"""
        if len(self.values) < 10:
            return False, "insufficient_data"
        
        stats = self.get_stats()
        recent = stats['recent']
        mean = stats['mean']
        std = stats['std']
        
        if std < 1e-8:  # 분산이 거의 없음
            return False, "no_variance"
        
        z_score = abs(recent - mean) / std
        
        if z_score > n_sigma:
            direction = "high" if recent > mean else "low"
            return True, f"anomaly_{direction}_z{z_score:.2f}"
        
        return False, "normal"

class StabilityMonitor:
    """
    학습 안정성 모니터링 및 자동 개입 시스템
    
    감시 항목:
    - Q-value 폭발/붕괴
    - 엔트로피 급락
    - 보상 클리프
    - 그래디언트 발산
    - 학습률 드리프트
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 모니터링 설정
                - window_size: 히스토리 윈도우 크기
                - n_sigma: 이상치 판정 시그마
                - intervention_threshold: 개입 임계값
                - rollback_enabled: 체크포인트 롤백 여부
        """
        self.config = config
        self.logger = FinFlowLogger("StabilityMonitor")
        
        # 설정 파라미터
        self.window_size = config.get('window_size', 100)
        self.n_sigma = config.get('n_sigma', 3.0)
        self.intervention_threshold = config.get('intervention_threshold', 3)
        self.rollback_enabled = config.get('rollback_enabled', True)
        
        # 메트릭 히스토리
        self.histories = {
            'q_value': MetricHistory(self.window_size),
            'q_value_std': MetricHistory(self.window_size),
            'entropy': MetricHistory(self.window_size),
            'reward': MetricHistory(self.window_size),
            'loss': MetricHistory(self.window_size),
            'gradient_norm': MetricHistory(self.window_size),
            'learning_rate': MetricHistory(self.window_size),
            'cql_alpha': MetricHistory(self.window_size),
            'portfolio_concentration': MetricHistory(self.window_size),
            'turnover': MetricHistory(self.window_size)
        }
        
        # 경고 카운터
        self.alert_counts = {
            'q_explosion': 0,
            'q_collapse': 0,
            'entropy_collapse': 0,
            'reward_cliff': 0,
            'gradient_explosion': 0,
            'concentration_risk': 0
        }
        
        # 개입 히스토리
        self.interventions = []
        self.last_checkpoint = None
        self.steps_since_intervention = 0
        
        # 임계값 설정
        self.thresholds = {
            'q_value_max': config.get('q_value_max', 100.0),
            'q_value_min': config.get('q_value_min', -100.0),
            'entropy_min': config.get('entropy_min', 0.1),
            'gradient_max': config.get('gradient_max', 10.0),
            'concentration_max': config.get('concentration_max', 0.5),
            'turnover_max': config.get('turnover_max', 0.5)
        }
        
        self.logger.info(f"StabilityMonitor 초기화 - window_size={self.window_size}, n_sigma={self.n_sigma}")
    
    def push(self, metrics: Dict[str, float]) -> None:
        """
        메트릭 추가 및 검증
        
        Args:
            metrics: 현재 스텝의 메트릭 딕셔너리
        """
        # 필수 메트릭 검증
        required_metrics = ['q_value', 'entropy', 'loss']
        for metric in required_metrics:
            assert metric in metrics, f"Missing required metric: {metric}"
            assert np.isfinite(metrics[metric]), f"Non-finite {metric}: {metrics[metric]}"
        
        # 메트릭 기록
        for key, value in metrics.items():
            if key in self.histories:
                self.histories[key].add(value)
        
        # Q-value 표준편차 계산 및 기록
        if 'q_values' in metrics:  # 여러 Q값이 있는 경우
            q_std = np.std(metrics['q_values'])
            self.histories['q_value_std'].add(q_std)
        
        self.steps_since_intervention += 1
        
        # 즉시 위험 체크
        self._immediate_safety_check(metrics)
    
    def _immediate_safety_check(self, metrics: Dict[str, float]) -> None:
        """즉시 안전성 체크 (assert 기반)"""
        # Q-value 범위 체크
        q_value = metrics.get('q_value', 0)
        assert q_value > self.thresholds['q_value_min'], \
            f"Q-value collapsed: {q_value} < {self.thresholds['q_value_min']}"
        assert q_value < self.thresholds['q_value_max'], \
            f"Q-value exploded: {q_value} > {self.thresholds['q_value_max']}"
        
        # 엔트로피 체크
        entropy = metrics.get('entropy', 1.0)
        assert entropy > 0, f"Entropy is non-positive: {entropy}"
        
        # 그래디언트 체크
        if 'gradient_norm' in metrics:
            grad_norm = metrics['gradient_norm']
            assert grad_norm < self.thresholds['gradient_max'], \
                f"Gradient explosion: {grad_norm} > {self.thresholds['gradient_max']}"
    
    def check(self) -> Dict[str, Any]:
        """
        이상 징후 체크
        
        Returns:
            alerts: 경고 정보
                - severity: 'normal', 'warning', 'critical'
                - issues: 발견된 문제 리스트
                - recommendations: 권장 조치
        """
        issues = []
        severity = 'normal'
        recommendations = []
        
        # 각 메트릭별 이상치 검사
        for metric_name, history in self.histories.items():
            if len(history.values) < 10:
                continue
            
            is_anomaly, anomaly_type = history.detect_anomaly(self.n_sigma)
            
            if is_anomaly:
                issues.append(f"{metric_name}: {anomaly_type}")
                
                # 메트릭별 특수 처리
                if metric_name == 'q_value' and 'high' in anomaly_type:
                    self.alert_counts['q_explosion'] += 1
                    recommendations.append("reduce_learning_rate")
                    recommendations.append("increase_cql_alpha")
                    
                elif metric_name == 'q_value' and 'low' in anomaly_type:
                    self.alert_counts['q_collapse'] += 1
                    recommendations.append("reduce_cql_alpha")
                    
                elif metric_name == 'entropy' and 'low' in anomaly_type:
                    self.alert_counts['entropy_collapse'] += 1
                    recommendations.append("increase_target_entropy")
                    recommendations.append("increase_alpha_min")
                    
                elif metric_name == 'gradient_norm' and 'high' in anomaly_type:
                    self.alert_counts['gradient_explosion'] += 1
                    recommendations.append("reduce_learning_rate")
                    recommendations.append("increase_gradient_clip")
                    
                elif metric_name == 'portfolio_concentration' and 'high' in anomaly_type:
                    self.alert_counts['concentration_risk'] += 1
                    recommendations.append("increase_entropy_bonus")
        
        # 추세 분석
        trends = self._analyze_trends()
        if trends['q_diverging']:
            issues.append("Q-values diverging")
            severity = 'warning'
        
        if trends['entropy_collapsing']:
            issues.append("Entropy collapsing")
            severity = 'warning'
        
        if trends['reward_degrading']:
            issues.append("Reward degrading")
            recommendations.append("check_exploration")
        
        # 심각도 판정
        critical_alerts = sum([
            self.alert_counts['q_explosion'] >= self.intervention_threshold,
            self.alert_counts['q_collapse'] >= self.intervention_threshold,
            self.alert_counts['entropy_collapse'] >= self.intervention_threshold,
            self.alert_counts['gradient_explosion'] >= 1
        ])
        
        if critical_alerts > 0:
            severity = 'critical'
        elif len(issues) > 2:
            severity = 'warning'
        
        return {
            'severity': severity,
            'issues': issues,
            'recommendations': list(set(recommendations)),  # 중복 제거
            'alert_counts': dict(self.alert_counts),
            'metrics_summary': self._get_metrics_summary()
        }
    
    def _analyze_trends(self) -> Dict[str, bool]:
        """추세 분석"""
        trends = {
            'q_diverging': False,
            'entropy_collapsing': False,
            'reward_degrading': False
        }
        
        # Q-value 발산 체크
        if len(self.histories['q_value'].values) >= 20:
            recent_q = list(self.histories['q_value'].values)[-10:]
            old_q = list(self.histories['q_value'].values)[-20:-10]
            
            recent_std = np.std(recent_q)
            old_std = np.std(old_q)
            
            if recent_std > old_std * 2:
                trends['q_diverging'] = True
        
        # 엔트로피 붕괴 체크
        if len(self.histories['entropy'].values) >= 20:
            recent_entropy = list(self.histories['entropy'].values)[-10:]
            if all(e < self.thresholds['entropy_min'] for e in recent_entropy):
                trends['entropy_collapsing'] = True
        
        # 보상 악화 체크
        if len(self.histories['reward'].values) >= 50:
            recent_rewards = list(self.histories['reward'].values)[-20:]
            old_rewards = list(self.histories['reward'].values)[-50:-30]
            
            if np.mean(recent_rewards) < np.mean(old_rewards) * 0.5:
                trends['reward_degrading'] = True
        
        return trends
    
    def intervene(self, trainer) -> None:
        """
        자동 개입 실행
        
        Args:
            trainer: 학습 트레이너 객체
        """
        alerts = self.check()
        
        if alerts['severity'] != 'critical':
            return
        
        self.logger.warning(f"자동 개입 시작 - Issues: {alerts['issues']}")
        
        intervention = {
            'step': trainer.global_step if hasattr(trainer, 'global_step') else 0,
            'timestamp': time.time(),
            'issues': alerts['issues'],
            'actions': []
        }
        
        # 권장 조치 실행
        for recommendation in alerts['recommendations']:
            if recommendation == 'reduce_learning_rate':
                # 학습률 감소
                for param_group in trainer.agent.actor_optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] *= 0.5
                    intervention['actions'].append(f"lr: {old_lr:.2e} -> {param_group['lr']:.2e}")
                    self.logger.info(f"학습률 감소: {old_lr:.2e} -> {param_group['lr']:.2e}")
            
            elif recommendation == 'increase_cql_alpha':
                # CQL 정규화 강화
                if hasattr(trainer.agent, 'cql_alpha'):
                    old_alpha = trainer.agent.cql_alpha
                    trainer.agent.cql_alpha = min(old_alpha * 1.5, 0.1)
                    intervention['actions'].append(f"cql_alpha: {old_alpha:.3f} -> {trainer.agent.cql_alpha:.3f}")
                    self.logger.info(f"CQL alpha 증가: {old_alpha:.3f} -> {trainer.agent.cql_alpha:.3f}")
            
            elif recommendation == 'reduce_cql_alpha':
                # CQL 정규화 완화
                if hasattr(trainer.agent, 'cql_alpha'):
                    old_alpha = trainer.agent.cql_alpha
                    trainer.agent.cql_alpha = max(old_alpha * 0.7, 0.001)
                    intervention['actions'].append(f"cql_alpha: {old_alpha:.3f} -> {trainer.agent.cql_alpha:.3f}")
                    self.logger.info(f"CQL alpha 감소: {old_alpha:.3f} -> {trainer.agent.cql_alpha:.3f}")
            
            elif recommendation == 'increase_target_entropy':
                # 타겟 엔트로피 증가
                if hasattr(trainer.agent, 'target_entropy'):
                    old_entropy = trainer.agent.target_entropy
                    trainer.agent.target_entropy *= 1.2
                    intervention['actions'].append(f"target_entropy: {old_entropy:.3f} -> {trainer.agent.target_entropy:.3f}")
                    self.logger.info(f"타겟 엔트로피 증가: {old_entropy:.3f} -> {trainer.agent.target_entropy:.3f}")
            
            elif recommendation == 'increase_alpha_min':
                # 최소 온도 증가
                if hasattr(trainer.agent, 'alpha_min'):
                    old_alpha_min = trainer.agent.alpha_min
                    trainer.agent.alpha_min = min(old_alpha_min * 2, 0.01)
                    intervention['actions'].append(f"alpha_min: {old_alpha_min:.4f} -> {trainer.agent.alpha_min:.4f}")
                    self.logger.info(f"최소 온도 증가: {old_alpha_min:.4f} -> {trainer.agent.alpha_min:.4f}")
            
            elif recommendation == 'increase_gradient_clip':
                # 그래디언트 클리핑 강화
                if hasattr(trainer.agent, 'grad_clip'):
                    old_clip = trainer.agent.grad_clip
                    trainer.agent.grad_clip = max(old_clip * 0.7, 0.1)
                    intervention['actions'].append(f"grad_clip: {old_clip:.2f} -> {trainer.agent.grad_clip:.2f}")
                    self.logger.info(f"그래디언트 클리핑 강화: {old_clip:.2f} -> {trainer.agent.grad_clip:.2f}")
        
        # 체크포인트 롤백 (극단적 경우)
        if self.rollback_enabled and len([c for c in self.alert_counts.values() if c >= self.intervention_threshold]) >= 2:
            if self.last_checkpoint and hasattr(trainer, 'load_checkpoint'):
                self.logger.warning("체크포인트 롤백 실행")
                trainer.load_checkpoint(self.last_checkpoint)
                intervention['actions'].append("checkpoint_rollback")
                # 경고 카운터 리셋
                self.alert_counts = {k: 0 for k in self.alert_counts}
        
        # 개입 기록
        self.interventions.append(intervention)
        self.steps_since_intervention = 0
        
        # 경고 카운터 감소 (롤백하지 않은 경우)
        if 'checkpoint_rollback' not in intervention['actions']:
            for key in self.alert_counts:
                self.alert_counts[key] = max(0, self.alert_counts[key] - 1)
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """체크포인트 저장"""
        self.last_checkpoint = checkpoint_path
        self.logger.debug(f"체크포인트 저장: {checkpoint_path}")
    
    def _get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """메트릭 요약 통계"""
        summary = {}
        for name, history in self.histories.items():
            if len(history.values) > 0:
                summary[name] = history.get_stats()
        return summary
    
    def get_report(self) -> Dict[str, Any]:
        """상세 리포트 생성"""
        return {
            'metrics_summary': self._get_metrics_summary(),
            'alert_counts': dict(self.alert_counts),
            'interventions': self.interventions,
            'steps_since_intervention': self.steps_since_intervention,
            'current_alerts': self.check()
        }
    
    def reset_alerts(self) -> None:
        """경고 카운터 리셋"""
        self.alert_counts = {k: 0 for k in self.alert_counts}
        self.logger.info("경고 카운터 리셋")