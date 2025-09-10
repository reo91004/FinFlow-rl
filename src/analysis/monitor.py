# src/analysis/monitor.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import datetime

from src.utils.logger import FinFlowLogger
from src.analysis.metrics import MetricsCalculator

@dataclass
class Alert:
    """알림 정보"""
    level: str  # 'info', 'warning', 'critical'
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: str

class PerformanceMonitor:
    """
    실시간 성능 모니터링 및 안정성 추적
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 use_wandb: bool = False,
                 use_tensorboard: bool = True,
                 wandb_config: Optional[Dict] = None,
                 alert_thresholds: Optional[Dict] = None):
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
        
        # 알림 임계값
        self.alert_thresholds = alert_thresholds or {
            'sharpe_ratio': {'warning': 1.0, 'critical': 0.5},
            'max_drawdown': {'warning': -0.15, 'critical': -0.25},
            'cvar_95': {'warning': -0.03, 'critical': -0.05},
            'volatility': {'warning': 0.20, 'critical': 0.30}
        }
        
        # 메트릭 히스토리
        self.history = []
        self.alerts = []
        
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
    
    def _check_alerts(self, metrics: Dict[str, float], step: int):
        """알림 체크"""
        
        for metric_name, thresholds in self.alert_thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                # Critical 체크
                if 'critical' in thresholds:
                    if metric_name in ['sharpe_ratio', 'cvar_95'] and value < thresholds['critical']:
                        alert = Alert(
                            level='critical',
                            metric=metric_name,
                            value=value,
                            threshold=thresholds['critical'],
                            message=f"{metric_name} critically low: {value:.4f}",
                            timestamp=datetime.datetime.now().isoformat()
                        )
                        self.alerts.append(alert)
                        self.logger.critical(alert.message)
                    elif metric_name in ['max_drawdown'] and value < thresholds['critical']:
                        alert = Alert(
                            level='critical',
                            metric=metric_name,
                            value=value,
                            threshold=thresholds['critical'],
                            message=f"{metric_name} critically high: {value:.4f}",
                            timestamp=datetime.datetime.now().isoformat()
                        )
                        self.alerts.append(alert)
                        self.logger.critical(alert.message)
                    elif metric_name in ['volatility'] and value > thresholds['critical']:
                        alert = Alert(
                            level='critical',
                            metric=metric_name,
                            value=value,
                            threshold=thresholds['critical'],
                            message=f"{metric_name} critically high: {value:.4f}",
                            timestamp=datetime.datetime.now().isoformat()
                        )
                        self.alerts.append(alert)
                        self.logger.critical(alert.message)
                
                # Warning 체크
                elif 'warning' in thresholds:
                    if metric_name in ['sharpe_ratio', 'cvar_95'] and value < thresholds['warning']:
                        alert = Alert(
                            level='warning',
                            metric=metric_name,
                            value=value,
                            threshold=thresholds['warning'],
                            message=f"{metric_name} warning: {value:.4f}",
                            timestamp=datetime.datetime.now().isoformat()
                        )
                        self.alerts.append(alert)
                        self.logger.warning(alert.message)
                    elif metric_name in ['max_drawdown'] and value < thresholds['warning']:
                        alert = Alert(
                            level='warning',
                            metric=metric_name,
                            value=value,
                            threshold=thresholds['warning'],
                            message=f"{metric_name} warning: {value:.4f}",
                            timestamp=datetime.datetime.now().isoformat()
                        )
                        self.alerts.append(alert)
                        self.logger.warning(alert.message)
                    elif metric_name in ['volatility'] and value > thresholds['warning']:
                        alert = Alert(
                            level='warning',
                            metric=metric_name,
                            value=value,
                            threshold=thresholds['warning'],
                            message=f"{metric_name} warning: {value:.4f}",
                            timestamp=datetime.datetime.now().isoformat()
                        )
                        self.alerts.append(alert)
                        self.logger.warning(alert.message)
    
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
    
    def close(self):
        """리소스 정리"""
        if self.use_tensorboard:
            self.writer.close()
        if self.use_wandb:
            self.wandb.finish()