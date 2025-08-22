# bipd/utils/rolling_stats.py

import collections
import numpy as np
from typing import Dict, Any, Optional

class RollingCounter:
    """슬라이딩 윈도우 기반 카운터 (최근 K개 기록만 유지)"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.buffer = collections.deque(maxlen=window_size)
        self.cumulative_count = 0  # 누적 카운터 (별도)
    
    def update(self, condition: bool):
        """조건 만족 여부를 기록하고 통계 반환"""
        self.buffer.append(bool(condition))
        self.cumulative_count += 1
        
        # 슬라이딩 윈도우 통계
        window_count = sum(self.buffer)
        window_size = len(self.buffer)
        window_rate = window_count / window_size if window_size > 0 else 0.0
        
        return {
            'sliding_count': window_count,
            'sliding_size': window_size, 
            'sliding_rate': window_rate,
            'cumulative_count': self.cumulative_count
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """현재 통계 반환 (업데이트 없이)"""
        window_count = sum(self.buffer)
        window_size = len(self.buffer)
        window_rate = window_count / window_size if window_size > 0 else 0.0
        
        return {
            'sliding_count': window_count,
            'sliding_size': window_size,
            'sliding_rate': window_rate,
            'cumulative_count': self.cumulative_count
        }

class RollingStatistics:
    """슬라이딩 윈도우 기반 통계 (평균, 표준편차, 분위수 등)"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.buffer = collections.deque(maxlen=window_size)
        self.cumulative_count = 0
        self.cumulative_sum = 0.0
    
    def update(self, value: float) -> Dict[str, float]:
        """값을 추가하고 통계 반환"""
        self.buffer.append(float(value))
        self.cumulative_count += 1
        self.cumulative_sum += value
        
        # 슬라이딩 윈도우 통계
        if len(self.buffer) > 0:
            values = np.array(list(self.buffer))
            stats = {
                'sliding_mean': float(values.mean()),
                'sliding_std': float(values.std()),
                'sliding_min': float(values.min()),
                'sliding_max': float(values.max()),
                'sliding_size': len(self.buffer),
                'cumulative_count': self.cumulative_count,
                'cumulative_mean': self.cumulative_sum / self.cumulative_count
            }
            
            # 분위수 계산
            if len(values) >= 5:
                stats['sliding_p25'] = float(np.percentile(values, 25))
                stats['sliding_p75'] = float(np.percentile(values, 75))
            
            return stats
        else:
            return {
                'sliding_mean': 0.0,
                'sliding_std': 0.0,
                'sliding_min': 0.0,
                'sliding_max': 0.0,
                'sliding_size': 0,
                'cumulative_count': self.cumulative_count,
                'cumulative_mean': 0.0
            }
    
    def get_stats(self) -> Dict[str, float]:
        """현재 통계 반환 (업데이트 없이)"""
        if len(self.buffer) > 0:
            values = np.array(list(self.buffer))
            stats = {
                'sliding_mean': float(values.mean()),
                'sliding_std': float(values.std()),
                'sliding_min': float(values.min()),
                'sliding_max': float(values.max()),
                'sliding_size': len(self.buffer),
                'cumulative_count': self.cumulative_count,
                'cumulative_mean': self.cumulative_sum / self.cumulative_count if self.cumulative_count > 0 else 0.0
            }
            
            if len(values) >= 5:
                stats['sliding_p25'] = float(np.percentile(values, 25))
                stats['sliding_p75'] = float(np.percentile(values, 75))
                
            return stats
        else:
            return {
                'sliding_mean': 0.0,
                'sliding_std': 0.0,
                'sliding_min': 0.0,
                'sliding_max': 0.0,
                'sliding_size': 0,
                'cumulative_count': self.cumulative_count,
                'cumulative_mean': 0.0
            }

class MultiRollingStats:
    """여러 지표에 대한 슬라이딩 윈도우 통계 관리자"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.counters: Dict[str, RollingCounter] = {}
        self.statistics: Dict[str, RollingStatistics] = {}
        self.last_report_step = 0
    
    def add_counter(self, name: str) -> RollingCounter:
        """새 카운터 추가"""
        counter = RollingCounter(self.window_size)
        self.counters[name] = counter
        return counter
    
    def add_statistics(self, name: str) -> RollingStatistics:
        """새 통계 추가"""
        stats = RollingStatistics(self.window_size)
        self.statistics[name] = stats
        return stats
    
    def get_counter(self, name: str) -> Optional[RollingCounter]:
        """카운터 획득"""
        return self.counters.get(name)
    
    def get_statistics(self, name: str) -> Optional[RollingStatistics]:
        """통계 획득"""
        return self.statistics.get(name)
    
    def should_report(self, current_step: int, report_interval: int = 100) -> bool:
        """보고 시점 판단"""
        if (current_step - self.last_report_step) >= report_interval:
            self.last_report_step = current_step
            return True
        return False
    
    def generate_report(self, context: str = "") -> str:
        """통합 보고서 생성"""
        lines = [f"=== Rolling Statistics Report {context} ==="]
        
        # 카운터 보고서
        if self.counters:
            lines.append("Counters:")
            for name, counter in self.counters.items():
                stats = counter.get_stats()
                lines.append(
                    f"  {name}: sliding({stats['sliding_count']}/{stats['sliding_size']}, "
                    f"{stats['sliding_rate']:.1%}) | cumulative({stats['cumulative_count']})"
                )
        
        # 통계 보고서
        if self.statistics:
            lines.append("Statistics:")
            for name, stat in self.statistics.items():
                stats = stat.get_stats()
                lines.append(
                    f"  {name}: sliding(μ={stats['sliding_mean']:.3f}, σ={stats['sliding_std']:.3f}, "
                    f"range=[{stats['sliding_min']:.2f}, {stats['sliding_max']:.2f}]) | "
                    f"cumulative(μ={stats['cumulative_mean']:.3f}, n={stats['cumulative_count']})"
                )
        
        return "\n".join(lines)