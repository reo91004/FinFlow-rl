# utils/extreme_q_monitor.py

import numpy as np
import collections
from utils.logger import BIPDLogger


class ExtremeQMonitor:
    """
    적응형 분위수 기반 Q-value 극단치 모니터링 시스템
    
    기존 고정 임계값 대신 슬라이딩 윈도우의 분위수를 기반으로 
    극단치를 판단하여 정보량 있는 경고만 발생시킵니다.
    """
    
    def __init__(self, window_size=1000, q_low=0.01, q_high=0.99, 
                 extreme_threshold=0.5, logger_name="ExtremeQMonitor"):
        """
        Args:
            window_size: 슬라이딩 윈도우 크기
            q_low: 하위 분위수 (0.01 = 1%)
            q_high: 상위 분위수 (0.99 = 99%) 
            extreme_threshold: 극단치 비율 경고 임계값 (0.5 = 50%)
            logger_name: 로거 이름
        """
        self.window_size = window_size
        self.q_low = q_low
        self.q_high = q_high
        self.extreme_threshold = extreme_threshold
        
        # 슬라이딩 윈도우 버퍼
        self.q_buffer = collections.deque(maxlen=window_size)
        
        # 통계 추적
        self.total_updates = 0
        self.total_extreme_count = 0
        self.recent_extreme_rates = collections.deque(maxlen=100)  # 최근 100회 극단비율
        
        # 로거 초기화
        self.logger = BIPDLogger(logger_name)
        
        # 초기 로그
        self.logger.info(
            f"ExtremeQ Monitor 초기화: 윈도우={window_size}, "
            f"분위수=({q_low:.1%},{q_high:.1%}), 경고임계값={extreme_threshold:.1%}"
        )
    
    def update_and_check(self, q_values: np.ndarray, context: str = ""):
        """
        Q-values 업데이트 및 극단치 검사
        
        Args:
            q_values: Q-value 배열
            context: 컨텍스트 정보 (예: "Critic1", "Critic2")
            
        Returns:
            dict: 극단치 분석 결과
        """
        # 입력 검증
        if not isinstance(q_values, np.ndarray):
            q_values = np.array(q_values)
        
        # NaN/Inf 처리
        finite_q = q_values[np.isfinite(q_values)]
        if len(finite_q) == 0:
            self.logger.warning(f"[{context}] 모든 Q-value가 NaN/Inf입니다.")
            return {"extreme_rate": 1.0, "warning": True}
        
        # 버퍼에 추가
        self.q_buffer.extend(finite_q.tolist())
        self.total_updates += 1
        
        # 충분한 데이터가 쌓일 때까지 대기
        if len(self.q_buffer) < max(200, int(0.2 * self.window_size)):
            return {"extreme_rate": 0.0, "warning": False}
        
        # 분위수 기반 극단치 경계 계산
        buffer_array = np.array(self.q_buffer)
        q_low_value = np.quantile(buffer_array, self.q_low)
        q_high_value = np.quantile(buffer_array, self.q_high)
        
        # 현재 배치의 극단치 비율 계산
        extreme_mask = (finite_q < q_low_value) | (finite_q > q_high_value)
        extreme_count = np.sum(extreme_mask)
        extreme_rate = extreme_count / len(finite_q) if len(finite_q) > 0 else 0.0
        
        # 극단치 통계 업데이트
        self.total_extreme_count += extreme_count
        self.recent_extreme_rates.append(extreme_rate)
        
        # 경고 발생 조건
        warning_issued = extreme_rate >= self.extreme_threshold
        
        # 결과 딕셔너리
        result = {
            "extreme_rate": extreme_rate,
            "extreme_count": extreme_count,
            "total_count": len(finite_q),
            "q_range": [float(finite_q.min()), float(finite_q.max())],
            "adaptive_bounds": [float(q_low_value), float(q_high_value)],
            "buffer_size": len(self.q_buffer),
            "warning": warning_issued
        }
        
        # 경고 로깅 (임계값 초과시)
        if warning_issued:
            avg_recent_rate = np.mean(self.recent_extreme_rates) if self.recent_extreme_rates else 0.0
            self.logger.warning(
                f"[{context}] 높은 극단 Q-value 비율: {extreme_rate:.1%} "
                f"({extreme_count}/{len(finite_q)}) | "
                f"Q 범위: [{finite_q.min():.1f}, {finite_q.max():.1f}] | "
                f"적응 경계: [{q_low_value:.1f}, {q_high_value:.1f}] | "
                f"최근 평균: {avg_recent_rate:.1%}"
            )
        
        return result
    
    def get_statistics(self):
        """전체 통계 반환"""
        if len(self.q_buffer) == 0:
            return {"status": "insufficient_data"}
        
        buffer_array = np.array(self.q_buffer)
        recent_rates = list(self.recent_extreme_rates) if self.recent_extreme_rates else [0.0]
        
        return {
            "status": "active",
            "total_updates": self.total_updates,
            "buffer_size": len(self.q_buffer),
            "buffer_stats": {
                "mean": float(buffer_array.mean()),
                "std": float(buffer_array.std()),
                "min": float(buffer_array.min()),
                "max": float(buffer_array.max()),
                "q01": float(np.quantile(buffer_array, 0.01)),
                "q99": float(np.quantile(buffer_array, 0.99))
            },
            "extreme_rates": {
                "recent_avg": float(np.mean(recent_rates)),
                "recent_max": float(np.max(recent_rates)),
                "recent_min": float(np.min(recent_rates)),
                "total_extreme_count": self.total_extreme_count
            }
        }
    
    def reset(self):
        """모니터 상태 초기화"""
        self.q_buffer.clear()
        self.recent_extreme_rates.clear()
        self.total_updates = 0
        self.total_extreme_count = 0
        self.logger.info("ExtremeQ Monitor 초기화 완료")


class DualQMonitor:
    """Twin Critics를 위한 듀얼 Q-value 모니터"""
    
    def __init__(self, window_size=1000, q_low=0.01, q_high=0.99, 
                 extreme_threshold=0.5, logger_name="DualQMonitor"):
        self.monitor_q1 = ExtremeQMonitor(window_size, q_low, q_high, 
                                         extreme_threshold, f"{logger_name}_Q1")
        self.monitor_q2 = ExtremeQMonitor(window_size, q_low, q_high,
                                         extreme_threshold, f"{logger_name}_Q2")
        self.logger = BIPDLogger(logger_name)
    
    def update_and_check_both(self, q1_values: np.ndarray, q2_values: np.ndarray):
        """두 Critic의 Q-values를 동시에 모니터링"""
        result_q1 = self.monitor_q1.update_and_check(q1_values, "Q1")
        result_q2 = self.monitor_q2.update_and_check(q2_values, "Q2") 
        
        # 통합 분석
        combined_extreme_rate = (
            result_q1.get("extreme_rate", 0.0) + result_q2.get("extreme_rate", 0.0)
        ) / 2.0
        
        # 심각한 불일치 감지
        rate_diff = abs(result_q1.get("extreme_rate", 0.0) - result_q2.get("extreme_rate", 0.0))
        if rate_diff > 0.3:  # 30% 이상 차이
            self.logger.warning(
                f"Twin Critics 극단비율 불일치: Q1={result_q1.get('extreme_rate', 0.0):.1%}, "
                f"Q2={result_q2.get('extreme_rate', 0.0):.1%} (차이: {rate_diff:.1%})"
            )
        
        return {
            "q1_result": result_q1,
            "q2_result": result_q2,
            "combined_extreme_rate": combined_extreme_rate,
            "rate_difference": rate_diff,
            "severe_mismatch": rate_diff > 0.3
        }
    
    def get_combined_statistics(self):
        """통합 통계 반환"""
        stats_q1 = self.monitor_q1.get_statistics()
        stats_q2 = self.monitor_q2.get_statistics()
        
        return {
            "q1_stats": stats_q1,
            "q2_stats": stats_q2,
            "summary": {
                "active": stats_q1.get("status") == "active" and stats_q2.get("status") == "active"
            }
        }