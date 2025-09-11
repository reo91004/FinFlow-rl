# src/analysis/monitor.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import json
import datetime
import threading
import time
from collections import deque
from pathlib import Path

from src.utils.logger import FinFlowLogger
from src.analysis.metrics import MetricsCalculator
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn

@dataclass
class Alert:
    """알림 정보"""
    level: str  # 'info', 'warning', 'critical', 'emergency'
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: str
    action_required: bool = False
    notification_sent: bool = False

class PerformanceMonitor:
    """
    실시간 성능 모니터링 및 안정성 추적
    
    - 실시간 대시보드
    - 강화된 알림 시스템
    - 상세 메트릭 추적
    - 거래 비용 분석
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 use_wandb: bool = False,
                 use_tensorboard: bool = True,
                 use_dashboard: bool = False,
                 dashboard_port: int = 8050,
                 wandb_config: Optional[Dict] = None,
                 alert_thresholds: Optional[Dict] = None,
                 notification_config: Optional[Dict] = None):
        """
        Args:
            log_dir: 로그 디렉토리
            use_wandb: Wandb 사용 여부
            use_tensorboard: TensorBoard 사용 여부
            wandb_config: Wandb 설정 (project, entity 등)
            alert_thresholds: 알림 임계값
        """
        self.logger = FinFlowLogger("PerformanceMonitor")
        self.metrics_calc = MetricsCalculator()
        
        # 모니터링 백엔드 조건부 활성화
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.use_dashboard = use_dashboard
        self.dashboard_port = dashboard_port
        
        # 알림 설정
        self.notification_config = notification_config or {}
        self.notification_handlers = self._setup_notification_handlers()
        
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            wandb_config = wandb_config or {}
            wandb.init(
                project=wandb_config.get('wandb_project', 'finflow-rl'),
                entity=wandb_config.get('wandb_entity'),
                tags=wandb_config.get('wandb_tags', []),
                dir=log_dir
            )
            self.logger.info("Wandb 초기화 완료")
        
        if self.use_tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.logger.info(f"TensorBoard 초기화 완료: {log_dir}")
        
        # 알림 임계값 (4단계)
        self.alert_thresholds = alert_thresholds or {
            'sharpe_ratio': {'info': 2.0, 'warning': 1.0, 'critical': 0.5, 'emergency': 0},
            'max_drawdown': {'info': -0.10, 'warning': -0.15, 'critical': -0.25, 'emergency': -0.40},
            'cvar_95': {'info': -0.02, 'warning': -0.03, 'critical': -0.05, 'emergency': -0.10},
            'volatility': {'info': 0.15, 'warning': 0.20, 'critical': 0.30, 'emergency': 0.50},
            'turnover': {'info': 0.1, 'warning': 0.3, 'critical': 0.5, 'emergency': 0.8},
            'concentration': {'info': 0.3, 'warning': 0.5, 'critical': 0.7, 'emergency': 0.9}
        }
        
        # 메트릭 히스토리
        self.history = []
        self.alerts = deque(maxlen=1000)  # 최근 1000개 알림만 유지
        
        # 실시간 메트릭 버퍼
        self.realtime_buffer = {
            'timestamps': deque(maxlen=1000),
            'portfolio_value': deque(maxlen=1000),
            'returns': deque(maxlen=1000),
            'positions': deque(maxlen=1000),
            'trades': deque(maxlen=100)
        }
        
        # 거래 비용 추적
        self.cost_tracker = {
            'transaction_costs': [],
            'slippage_costs': [],
            'market_impact_costs': [],
            'total_costs': 0
        }
        
        # 이상 감지 모델
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            n_estimators=100,
            random_state=42
        )
        self.anomaly_buffer = deque(maxlen=1000)
        self.anomaly_fitted = False
        
        # 자동 개입 설정
        self.auto_intervention = {
            'enabled': True,
            'intervention_count': 0,
            'last_intervention': None,
            'cooldown_steps': 100
        }
        
        # 대시보드 시작
        if self.use_dashboard:
            self._start_dashboard()
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """메트릭 로깅"""
        # 히스토리 저장
        self.history.append({'step': step, **metrics})
        
        # Wandb 로깅
        if self.use_wandb:
            self.wandb.log(metrics, step=step)
        
        # TensorBoard 로깅
        if self.use_tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
        
        # 이상 감지 및 자동 개입
        anomalies = self._detect_anomalies(metrics)
        if anomalies:
            interventions = self._auto_intervene(metrics, anomalies)
            if interventions:
                metrics['interventions'] = interventions
        
        # 알림 체크
        self._check_alerts(metrics, step)
    
    def log_portfolio(self, weights: np.ndarray, asset_names: List[str], step: int):
        """포트폴리오 가중치 로깅"""
        portfolio_dict = {f"portfolio/{name}": w for name, w in zip(asset_names, weights)}
        
        if self.use_wandb:
            self.wandb.log(portfolio_dict, step=step)
        
        if self.use_tensorboard:
            for name, weight in portfolio_dict.items():
                self.writer.add_scalar(name, weight, step)
    
    def log_gradients(self, model: Any, step: int):
        """그래디언트 통계 로깅"""
        grad_stats = self._compute_gradient_stats(model)
        
        if self.use_wandb:
            self.wandb.log({f"gradients/{k}": v for k, v in grad_stats.items()}, step=step)
        
        if self.use_tensorboard:
            for key, value in grad_stats.items():
                self.writer.add_scalar(f"gradients/{key}", value, step)
    
    def _compute_gradient_stats(self, model: Any) -> Dict[str, float]:
        """그래디언트 통계 계산"""
        total_norm = 0
        grad_norms = []
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                grad_norms.append(param_norm)
        
        total_norm = total_norm ** 0.5
        
        return {
            'norm': total_norm,
            'max': max(grad_norms) if grad_norms else 0,
            'min': min(grad_norms) if grad_norms else 0,
            'mean': np.mean(grad_norms) if grad_norms else 0
        }
    
    def _detect_anomalies(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """머신러닝 기반 이상 감지"""
        anomalies = {}
        
        # 특징 벡터 생성
        feature_keys = ['sharpe_ratio', 'volatility', 'max_drawdown', 'turnover']
        features = []
        for key in feature_keys:
            if key in metrics:
                features.append(metrics[key])
            else:
                features.append(0.0)
        
        # 버퍼에 추가
        self.anomaly_buffer.append(features)
        
        # 충분한 데이터가 모이면 모델 학습/예측
        if len(self.anomaly_buffer) >= 100:
            if not self.anomaly_fitted:
                # 초기 학습
                self.anomaly_detector.fit(list(self.anomaly_buffer))
                self.anomaly_fitted = True
            
            # 이상치 예측
            score = self.anomaly_detector.decision_function([features])[0]
            is_anomaly = self.anomaly_detector.predict([features])[0] == -1
            
            if is_anomaly:
                anomalies['ml_detection'] = {
                    'score': score,
                    'features': dict(zip(feature_keys, features)),
                    'severity': 'high' if score < -0.5 else 'medium'
                }
        
        # 통계적 이상치 검출
        for key, value in metrics.items():
            if key in self.history and len(self.history) > 30:
                recent_values = [h.get(key, 0) for h in self.history[-30:]]
                mean = np.mean(recent_values)
                std = np.std(recent_values)
                
                if std > 1e-8:
                    z_score = abs(value - mean) / std
                    if z_score > 3:
                        anomalies[f'{key}_statistical'] = {
                            'z_score': z_score,
                            'value': value,
                            'mean': mean,
                            'std': std
                        }
        
        return anomalies
    
    def _auto_intervene(self, metrics: Dict[str, float], anomalies: Dict[str, Any]) -> List[str]:
        """자동 개입 시스템"""
        interventions = []
        
        if not self.auto_intervention['enabled']:
            return interventions
        
        # 쿨다운 체크
        if self.auto_intervention['last_intervention']:
            steps_since = metrics.get('step', 0) - self.auto_intervention['last_intervention']
            if steps_since < self.auto_intervention['cooldown_steps']:
                return interventions
        
        # Q-value 폭발 체크
        if 'q_value' in metrics:
            q_value = metrics['q_value']
            if q_value > 100:
                interventions.append('reduce_learning_rate')
                self.logger.warning(f"Q-value 폭발 감지: {q_value:.2f}, 학습률 감소")
            elif q_value < -100:
                interventions.append('reset_q_network')
                self.logger.warning(f"Q-value 붕괴 감지: {q_value:.2f}, Q-network 재초기화")
        
        # 엔트로피 붕괴 체크
        if 'entropy' in metrics and metrics['entropy'] < 0.1:
            interventions.append('increase_exploration')
            self.logger.warning(f"엔트로피 붕괴: {metrics['entropy']:.3f}, 탐색 증가")
        
        # 그래디언트 폭발 체크
        if 'gradient_norm' in metrics and metrics['gradient_norm'] > 10:
            interventions.append('clip_gradients')
            self.logger.warning(f"그래디언트 폭발: {metrics['gradient_norm']:.2f}, 클리핑 적용")
        
        # 보상 클리프 체크
        if 'reward' in metrics and len(self.history) > 10:
            recent_rewards = [h.get('reward', 0) for h in self.history[-10:]]
            if metrics['reward'] < np.mean(recent_rewards) * 0.5:
                interventions.append('adjust_reward_scale')
                self.logger.warning(f"보상 클리프 감지, 보상 스케일 조정")
        
        # ML 기반 이상치에 대한 개입
        if 'ml_detection' in anomalies and anomalies['ml_detection']['severity'] == 'high':
            interventions.append('reduce_update_frequency')
            self.logger.warning("ML 모델이 심각한 이상 패턴 감지, 업데이트 빈도 감소")
        
        # 다중 문제 발생 시 체크포인트 롤백
        if len(interventions) >= 3:
            interventions = ['rollback_checkpoint']
            self.logger.error("다중 문제 감지, 체크포인트 롤백")
        
        # 개입 기록
        if interventions:
            self.auto_intervention['intervention_count'] += 1
            self.auto_intervention['last_intervention'] = metrics.get('step', 0)
            self.logger.info(f"자동 개입 수행: {interventions}")
        
        return interventions
    
    def _check_alerts(self, metrics: Dict[str, float], step: int):
        """강화된 다단계 알림 체크"""
        
        for metric_name, thresholds in self.alert_thresholds.items():
            if metric_name not in metrics:
                continue
                
            value = metrics[metric_name]
            alert_level = None
            threshold_value = None
            
            # 메트릭 타입별 비교 방향 결정
            is_lower_bad = metric_name in ['sharpe_ratio', 'cvar_95']
            is_higher_bad = metric_name in ['volatility', 'turnover', 'concentration', 'max_drawdown']
            
            # 레벨별 체크 (emergency -> critical -> warning -> info)
            for level in ['emergency', 'critical', 'warning', 'info']:
                if level not in thresholds:
                    continue
                    
                threshold = thresholds[level]
                
                if is_lower_bad and value < threshold:
                    alert_level = level
                    threshold_value = threshold
                    break
                elif is_higher_bad:
                    # max_drawdown은 음수이므로 특별 처리
                    if metric_name == 'max_drawdown' and value < threshold:
                        alert_level = level
                        threshold_value = threshold
                        break
                    elif metric_name != 'max_drawdown' and value > threshold:
                        alert_level = level
                        threshold_value = threshold
                        break
            
            if alert_level:
                # 알림 생성
                alert = Alert(
                    level=alert_level,
                    metric=metric_name,
                    value=value,
                    threshold=threshold_value,
                    message=self._format_alert_message(metric_name, value, alert_level),
                    timestamp=datetime.datetime.now().isoformat(),
                    action_required=(alert_level in ['critical', 'emergency'])
                )
                
                self.alerts.append(alert)
                
                # 로깅
                if alert_level == 'emergency':
                    self.logger.critical(f"[EMERGENCY] {alert.message}")
                elif alert_level == 'critical':
                    self.logger.critical(alert.message)
                elif alert_level == 'warning':
                    self.logger.warning(alert.message)
                else:
                    self.logger.info(alert.message)
                
                # 알림 전송
                self._send_notification(alert)
    
    def get_stability_report(self) -> Dict:
        """안정성 리포트 생성"""
        if len(self.history) < 10:
            return {'status': 'insufficient_data'}
        
        recent_metrics = pd.DataFrame(self.history[-100:])
        
        # 안정성 메트릭 계산
        stability_metrics = {}
        
        for col in recent_metrics.columns:
            if col != 'step' and recent_metrics[col].dtype in [np.float64, np.float32, np.int64]:
                values = recent_metrics[col].values
                stability_metrics[col] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'trend': float(np.polyfit(range(len(values)), values, 1)[0]),
                    'volatility': float(np.std(values) / (np.mean(values) + 1e-8))
                }
        
        # 전체 안정성 점수
        volatilities = [m['volatility'] for m in stability_metrics.values()]
        avg_volatility = np.mean(volatilities)
        stability_score = max(0, min(100, 100 * (1 - avg_volatility)))
        
        return {
            'stability_score': stability_score,
            'metrics': stability_metrics,
            'recent_alerts': [vars(a) for a in self.alerts[-10:]],
            'status': 'stable' if stability_score > 70 else 'unstable'
        }
    
    def save_history(self, path: str):
        """히스토리 저장"""
        pd.DataFrame(self.history).to_csv(path, index=False)
        self.logger.info(f"히스토리 저장: {path}")
    
    def _setup_notification_handlers(self) -> Dict[str, Callable]:
        """알림 핸들러 설정"""
        handlers = {}
        
        # 이메일 핸들러
        if self.notification_config.get('email_enabled'):
            handlers['email'] = self._send_email_notification
        
        # Slack 핸들러
        if self.notification_config.get('slack_enabled'):
            handlers['slack'] = self._send_slack_notification
        
        # 콘솔 핸들러 (기본)
        handlers['console'] = self._send_console_notification
        
        return handlers
    
    def _format_alert_message(self, metric: str, value: float, level: str) -> str:
        """알림 메시지 포맷팅"""
        emoji_map = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'critical': '🚨',
            'emergency': '🆘'
        }
        
        # 연구용 코드이므로 이모지 제거
        return f"[{level.upper()}] {metric}: {value:.4f}"
    
    def _send_notification(self, alert: Alert):
        """알림 전송"""
        # 레벨별 필터링
        min_level = self.notification_config.get('min_notification_level', 'warning')
        level_priority = {'info': 0, 'warning': 1, 'critical': 2, 'emergency': 3}
        
        if level_priority.get(alert.level, 0) < level_priority.get(min_level, 1):
            return
        
        # 각 핸들러로 전송
        for handler_name, handler_func in self.notification_handlers.items():
            handler_func(alert)
        
        alert.notification_sent = True
    
    def _send_console_notification(self, alert: Alert):
        """콘솔 알림"""
        print(f"\n{'='*60}")
        print(f"ALERT: {alert.level.upper()}")
        print(f"Metric: {alert.metric}")
        print(f"Value: {alert.value:.4f}")
        print(f"Threshold: {alert.threshold:.4f}")
        print(f"Time: {alert.timestamp}")
        print(f"Action Required: {alert.action_required}")
        print(f"{'='*60}\n")
    
    def _send_email_notification(self, alert: Alert):
        """이메일 알림 (구현 예시)"""
        # 실제 구현 시 SMTP 설정 필요
        pass
    
    def _send_slack_notification(self, alert: Alert):
        """Slack 알림 (구현 예시)"""
        # 실제 구현 시 Slack Webhook URL 필요
        pass
    
    def log_trade(self, trade: Dict[str, Any]):
        """거래 로깅 및 비용 추적"""
        self.realtime_buffer['trades'].append(trade)
        
        # 비용 추적
        if 'costs' in trade:
            costs = trade['costs']
            if 'transaction_cost' in costs:
                self.cost_tracker['transaction_costs'].append(costs['transaction_cost'])
            if 'slippage_cost' in costs:
                self.cost_tracker['slippage_costs'].append(costs['slippage_cost'])
            if 'market_impact_cost' in costs:
                self.cost_tracker['market_impact_costs'].append(costs['market_impact_cost'])
            
            total_cost = sum(costs.values())
            self.cost_tracker['total_costs'] += total_cost
    
    def update_realtime(self, portfolio_value: float, returns: float, positions: np.ndarray):
        """실시간 데이터 업데이트"""
        timestamp = datetime.datetime.now()
        
        self.realtime_buffer['timestamps'].append(timestamp)
        self.realtime_buffer['portfolio_value'].append(portfolio_value)
        self.realtime_buffer['returns'].append(returns)
        self.realtime_buffer['positions'].append(positions.copy())
        
        # 실시간 메트릭 계산
        if len(self.realtime_buffer['returns']) > 20:
            recent_returns = list(self.realtime_buffer['returns'])[-20:]
            realtime_metrics = {
                'realtime_sharpe': np.mean(recent_returns) / (np.std(recent_returns) + 1e-8) * np.sqrt(252),
                'realtime_volatility': np.std(recent_returns) * np.sqrt(252),
                'realtime_drawdown': self._calculate_realtime_drawdown()
            }
            
            # 실시간 알림 체크
            self._check_alerts(realtime_metrics, len(self.history))
    
    def _calculate_realtime_drawdown(self) -> float:
        """실시간 드로다운 계산"""
        if len(self.realtime_buffer['portfolio_value']) < 2:
            return 0
        
        values = np.array(list(self.realtime_buffer['portfolio_value']))
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        return float(np.min(drawdown))
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """거래 비용 분석"""
        if not self.cost_tracker['transaction_costs']:
            return {'total_costs': 0, 'cost_breakdown': {}}
        
        return {
            'total_costs': self.cost_tracker['total_costs'],
            'cost_breakdown': {
                'transaction_costs': sum(self.cost_tracker['transaction_costs']),
                'slippage_costs': sum(self.cost_tracker['slippage_costs']),
                'market_impact_costs': sum(self.cost_tracker['market_impact_costs'])
            },
            'avg_costs': {
                'transaction': np.mean(self.cost_tracker['transaction_costs']),
                'slippage': np.mean(self.cost_tracker['slippage_costs']),
                'market_impact': np.mean(self.cost_tracker['market_impact_costs'])
            },
            'cost_per_trade': self.cost_tracker['total_costs'] / len(self.realtime_buffer['trades']) if self.realtime_buffer['trades'] else 0
        }
    
    def _start_dashboard(self):
        """실시간 대시보드 시작"""
        def run_dashboard():
            # Dash 대시보드 구현 (선택적)
            self.logger.info(f"대시보드 시작: http://localhost:{self.dashboard_port}")
            # 실제 구현 시 Dash/Plotly 사용
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """대시보드용 데이터 반환"""
        return {
            'portfolio_value': list(self.realtime_buffer['portfolio_value']),
            'returns': list(self.realtime_buffer['returns']),
            'positions': list(self.realtime_buffer['positions']),
            'timestamps': [t.isoformat() for t in self.realtime_buffer['timestamps']],
            'recent_alerts': [vars(a) for a in list(self.alerts)[-10:]],
            'cost_analysis': self.get_cost_analysis(),
            'stability_report': self.get_stability_report()
        }
    
    def close(self):
        """리소스 정리"""
        if self.use_tensorboard:
            self.writer.close()
        if self.use_wandb:
            self.wandb.finish()
        
        # 최종 리포트 생성
        final_report = {
            'total_alerts': len(self.alerts),
            'alert_breakdown': self._get_alert_breakdown(),
            'cost_analysis': self.get_cost_analysis(),
            'final_metrics': self.history[-1] if self.history else {}
        }
        
        report_path = Path('logs') / f"monitor_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        self.logger.info(f"최종 리포트 저장: {report_path}")
    
    def _get_alert_breakdown(self) -> Dict[str, int]:
        """알림 분류별 집계"""
        breakdown = {'info': 0, 'warning': 0, 'critical': 0, 'emergency': 0}
        for alert in self.alerts:
            breakdown[alert.level] = breakdown.get(alert.level, 0) + 1
        return breakdown