# bipd/utils/rolling_stats.py

import collections
import numpy as np
from typing import Dict, Any, Optional

# 사용되지 않는 RollingCounter, RollingStatistics 클래스 제거
# MultiRollingStats만 유지 (agents/bcell.py에서 사용 중)

class RollingStatistic:
    """단일 지표에 대한 슬라이딩 윈도우 통계 객체"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data = collections.deque(maxlen=window_size)
    
    def update(self, value: float) -> Dict[str, float]:
        """값 업데이트 및 통계 반환"""
        self.data.append(float(value))
        return self.get_stats()
    
    def get_stats(self) -> Dict[str, float]:
        """현재 통계 반환 (카운터 비율 포함)"""
        if len(self.data) == 0:
            return {
                'sliding_mean': 0.0,
                'sliding_std': 0.0,
                'sliding_min': 0.0,
                'sliding_max': 0.0,
                'sliding_size': 0,
                'sliding_rate': 0.0,  # 카운터용 비율 (평균과 동일)
                'cumulative_count': 0
            }
        
        values = np.array(list(self.data))
        mean_val = float(values.mean())
        return {
            'sliding_mean': mean_val,
            'sliding_std': float(values.std()) if len(values) > 1 else 0.0,
            'sliding_min': float(values.min()),
            'sliding_max': float(values.max()),
            'sliding_size': len(values),
            'sliding_rate': mean_val,  # boolean 값들의 평균 = 비율
            'cumulative_count': len(values)  # 단순화된 버전에서는 동일
        }
    
    def reset(self):
        """통계 초기화"""
        self.data.clear()


class MultiRollingStats:
    """여러 지표에 대한 슬라이딩 윈도운 통계 관리자"""
    
    def __init__(self, names: Optional[list] = None, window_size: int = 100):
        self.window_size = window_size
        self.data: Dict[str, collections.deque] = {}
        self.last_report_step = 0  # 보고 주기 관리
        if names:
            for name in names:
                self.data[name] = collections.deque(maxlen=window_size)
    
    def add_statistics(self, name: str) -> RollingStatistic:
        """새로운 통계 대상 추가 및 객체 반환"""
        return RollingStatistic(self.window_size)
    
    def add_counter(self, name: str) -> RollingStatistic:
        """새로운 카운터 대상 추가 및 객체 반환 (통계 객체와 동일하게 처리)"""
        return RollingStatistic(self.window_size)
    
    def should_report(self, current_step: int, report_interval: int = 500) -> bool:
        """보고 주기 확인"""
        if current_step - self.last_report_step >= report_interval:
            self.last_report_step = current_step
            return True
        return False
    
    def update(self, updates: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """여러 통계 동시 업데이트 (기존 API 호환성용)"""
        results = {}
        for name, value in updates.items():
            if name not in self.data:
                self.data[name] = collections.deque(maxlen=self.window_size)
            self.data[name].append(float(value))
            results[name] = self._get_stat(name)
        return results
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """모든 통계 반환 (기존 API 호환성용)"""
        return {name: self._get_stat(name) for name in self.data.keys()}
    
    def _get_stat(self, name: str) -> Dict[str, float]:
        """특정 통계 내부 계산"""
        if name not in self.data or len(self.data[name]) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
        
        values = np.array(list(self.data[name]))
        return {
            'mean': float(values.mean()),
            'std': float(values.std()) if len(values) > 1 else 0.0,
            'min': float(values.min()),
            'max': float(values.max()),
            'count': len(values)
        }