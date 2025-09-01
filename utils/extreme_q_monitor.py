# utils/extreme_q_monitor.py

import numpy as np
import collections
from utils.logger import BIPDLogger

# scipy import with fallback for statistical significance testing
try:
    from scipy.stats import t
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ExtremeQMonitor:
    """
    개선된 적응형 분위수 기반 Q-value 극단치 모니터링 시스템
    
    콜드스타트 유예, 퇴화 분포 감지, 히스테리시스 기반으로
    오경보를 최소화하고 진짜 위험만 감지합니다.
    """
    
    def __init__(self, window_size=1000, q_low=0.05, q_high=0.95, 
                 extreme_threshold=0.30, min_samples=1000, min_iqr=1e-6,
                 logger_name="ExtremeQMonitor"):
        """
        Args:
            window_size: 슬라이딩 윈도우 크기
            q_low: 하위 분위수 (0.05 = 5%)
            q_high: 상위 분위수 (0.95 = 95%) 
            extreme_threshold: 극단치 비율 경고 임계값 (0.30 = 30%)
            min_samples: 콜드스타트 유예 최소 샘플 수
            min_iqr: 퇴화 분포 감지 최소 IQR
            logger_name: 로거 이름
        """
        self.window_size = window_size
        self.q_low = q_low
        self.q_high = q_high
        self.extreme_threshold = extreme_threshold
        self.min_samples = min_samples  # 콜드스타트 유예
        self.min_iqr = min_iqr         # 퇴화 분포 감지
        
        # 슬라이딩 윈도우 버퍼
        self.q_buffer = collections.deque(maxlen=window_size)
        
        # 통계 추적
        self.total_updates = 0
        self.total_extreme_count = 0
        self.recent_extreme_rates = collections.deque(maxlen=100)  # 최근 100회 극단비율
        
        # 히스테리시스 기반 안정적 경고 시스템
        self.warning_history = collections.deque(maxlen=5)  # 최근 5회 경고 상태
        self.cold_start_phase = True  # 콜드스타트 상태
        
        # 로거 초기화
        self.logger = BIPDLogger(logger_name)
        
        # 초기 로그
        self.logger.info(
            f"개선된 ExtremeQ Monitor 초기화: 윈도우={window_size}, "
            f"분위수=({q_low:.1%},{q_high:.1%}), 경고임계값={extreme_threshold:.1%}, "
            f"콜드스타트_유예={min_samples}, 최소_IQR={min_iqr:.2e}"
        )
    
    def update_and_check(self, q_values: np.ndarray, context: str = ""):
        """
        개선된 Q-values 업데이트 및 극단치 검사 (오경보 방지)
        
        Args:
            q_values: Q-value 배열 (클리핑 전 원시값 권장)
            context: 컨텍스트 정보 (예: "Critic1", "Critic2")
            
        Returns:
            dict: 극단치 분석 결과
        """
        # 입력 검증
        if not isinstance(q_values, np.ndarray):
            q_values = np.array(q_values, dtype=np.float64)
        
        # NaN/Inf 처리
        finite_q = q_values[np.isfinite(q_values)]
        if len(finite_q) == 0:
            self.logger.warning(f"[{context}] 모든 Q-value가 NaN/Inf입니다.")
            return {"extreme_rate": 1.0, "warning": True, "status": "all_invalid"}
        
        # -0.0 → 0.0 정규화 (부동소수점 정밀도 문제)
        finite_q = np.where(np.isclose(finite_q, 0.0, atol=1e-12), 0.0, finite_q)
        
        # 버퍼에 추가
        self.q_buffer.extend(finite_q.tolist())
        self.total_updates += 1
        
        # 1단계: 콜드스타트 유예 (업데이트 횟수 기준으로 변경)
        # min_samples를 업데이트 횟수로 해석 (배치당 32개씩 추가되므로)
        min_updates = max(10, self.min_samples // 32)  # 최소 10회 업데이트
        if self.total_updates < min_updates:
            if self.cold_start_phase and self.total_updates % 5 == 0:
                self.logger.debug(f"[{context}] 콜드스타트 진행중: {self.total_updates}/{min_updates} 업데이트")
            return {"extreme_rate": 0.0, "warning": False, "status": "cold_start"}
        
        # 콜드스타트 종료 로그
        if self.cold_start_phase:
            self.cold_start_phase = False
            self.logger.info(f"[{context}] 콜드스타트 완료: {self.total_updates}회 업데이트, {len(self.q_buffer)}개 샘플로 모니터링 시작")
        
        # 분위수 기반 극단치 경계 계산
        buffer_array = np.array(self.q_buffer, dtype=np.float64)
        q_low_value = np.quantile(buffer_array, self.q_low)
        q_high_value = np.quantile(buffer_array, self.q_high)
        iqr = q_high_value - q_low_value
        
        # 2단계: 퇴화 분포 감지 (IQR ≈ 0인 경우 스킵)
        if iqr < self.min_iqr:
            return {
                "extreme_rate": 0.0, "warning": False, "status": "degenerate_distribution",
                "iqr": float(iqr), "q_range": [float(finite_q.min()), float(finite_q.max())],
                "adaptive_bounds": [float(q_low_value), float(q_high_value)]
            }
        
        # 3단계: 엄격한 극단치 비율 계산 (경계값 제외)
        extreme_mask = (finite_q < q_low_value) | (finite_q > q_high_value)
        extreme_count = np.sum(extreme_mask)
        extreme_rate = extreme_count / len(finite_q) if len(finite_q) > 0 else 0.0
        
        # 극단치 통계 업데이트
        self.total_extreme_count += extreme_count
        self.recent_extreme_rates.append(extreme_rate)
        
        # 4단계: 히스테리시스 + 신뢰구간 기반 안정적 경고
        current_warning = extreme_rate >= self.extreme_threshold
        
        # 3단계: 통계적 유의성 검정 (연속 3회 이상 데이터 있을 때만)
        statistical_significance = False
        if len(self.recent_extreme_rates) >= 3:
            recent_rates = list(self.recent_extreme_rates)[-3:]  # 최근 3회
            mean_rate = np.mean(recent_rates)
            std_rate = np.std(recent_rates, ddof=1) if len(recent_rates) > 1 else 0.0
            
            # 95% 신뢰구간 계산 (t-분포 사용, n=3이라 작은 샘플)
            if std_rate > 0:
                n = len(recent_rates)
                if HAS_SCIPY:
                    t_critical = t.ppf(0.975, df=n-1)  # 95% 신뢰구간
                else:
                    # scipy 없을 때 근사값 (n=3일 때 t_{0.025,2} ≈ 4.303)
                    t_critical = 4.303 if n == 3 else 2.92 if n == 4 else 2.57
                
                margin_error = t_critical * std_rate / np.sqrt(n)
                ci_lower = mean_rate - margin_error
                
                # 신뢰구간 하한이 임계값을 위에 있는가?
                statistical_significance = ci_lower > self.extreme_threshold
        
        # 히스테리시스 + 통계적 유의성 결합
        self.warning_history.append(current_warning)
        stable_warning = (len(self.warning_history) == 5 and all(self.warning_history) 
                         and statistical_significance)
        
        # 결과 딕셔너리
        result = {
            "extreme_rate": extreme_rate,
            "extreme_count": extreme_count,
            "total_count": len(finite_q),
            "q_range": [float(finite_q.min()), float(finite_q.max())],
            "adaptive_bounds": [float(q_low_value), float(q_high_value)],
            "iqr": float(iqr),
            "buffer_size": len(self.q_buffer),
            "warning": stable_warning,
            "status": "active"
        }
        
        # 안정적 경고 로깅 (5회 연속 + 통계적 유의성)
        if stable_warning:
            avg_recent_rate = np.mean(self.recent_extreme_rates) if self.recent_extreme_rates else 0.0
            significance_note = "통계유의" if statistical_significance else "히스테리시스만"
            self.logger.warning(
                f"[{context}] 안정적 극단 Q-value 경고 ({significance_note}): {extreme_rate:.1%} "
                f"({extreme_count}/{len(finite_q)}) | "
                f"Q범위: [{finite_q.min():.3f}, {finite_q.max():.3f}] | "
                f"적응경계: [{q_low_value:.3f}, {q_high_value:.3f}] (IQR={iqr:.3f}) | "
                f"최근평균: {avg_recent_rate:.1%}"
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
    """Twin Critics를 위한 듀얼 Q-value 모니터 (z-score 정규화 기반 불일치 감지)"""
    
    def __init__(self, window_size=1000, q_low=0.01, q_high=0.99, 
                 extreme_threshold=0.5, min_samples=1000, min_iqr=1e-6, logger_name="DualQMonitor"):
        self.monitor_q1 = ExtremeQMonitor(window_size, q_low, q_high, 
                                         extreme_threshold, min_samples, min_iqr, f"{logger_name}_Q1")
        self.monitor_q2 = ExtremeQMonitor(window_size, q_low, q_high,
                                         extreme_threshold, min_samples, min_iqr, f"{logger_name}_Q2")
        self.logger = BIPDLogger(logger_name)
        
        # z-score 기반 불일치 모니터링
        self.q1_q2_diff_history = collections.deque(maxlen=100)  # 최근 100회 Q1-Q2 차이
    
    def update_and_check_both(self, q1_values: np.ndarray, q2_values: np.ndarray):
        """두 Critic의 Q-values를 동시에 모니터링"""
        result_q1 = self.monitor_q1.update_and_check(q1_values, "Q1")
        result_q2 = self.monitor_q2.update_and_check(q2_values, "Q2") 
        
        # 통합 분석
        combined_extreme_rate = (
            result_q1.get("extreme_rate", 0.0) + result_q2.get("extreme_rate", 0.0)
        ) / 2.0
        
        # 4단계: z-score 기반 Twin 불일치 정규화 감지
        q1_mean = np.mean(q1_values) if len(q1_values) > 0 else 0.0
        q2_mean = np.mean(q2_values) if len(q2_values) > 0 else 0.0
        q_diff = q1_mean - q2_mean
        
        # 차이 히스토리 업데이트
        self.q1_q2_diff_history.append(q_diff)
        
        # z-score 기반 이상치 감지
        z_score_anomaly = False
        normalized_diff = 0.0
        if len(self.q1_q2_diff_history) >= 10:  # 최소 10회 데이터
            diff_array = np.array(self.q1_q2_diff_history)
            diff_mean = np.mean(diff_array)
            diff_std = np.std(diff_array, ddof=1)
            
            if diff_std > 1e-6:  # 상수가 아닌 경우만
                normalized_diff = (q_diff - diff_mean) / diff_std
                z_score_anomaly = abs(normalized_diff) > 2.5  # 99% 신뢰구간 밖
        
        # 기존 방식도 유지 (백업)
        rate_diff = abs(result_q1.get("extreme_rate", 0.0) - result_q2.get("extreme_rate", 0.0))
        severe_mismatch = rate_diff > 0.3 or z_score_anomaly
        
        if severe_mismatch:
            anomaly_type = "z-score" if z_score_anomaly else "rate"
            self.logger.warning(
                f"Twin Critics 불일치 ({anomaly_type}): Q1평균={q1_mean:.4f}, Q2평균={q2_mean:.4f} | "
                f"z-score={normalized_diff:.2f} | 비율차이: {rate_diff:.1%}"
            )
        
        return {
            "q1_result": result_q1,
            "q2_result": result_q2,
            "combined_extreme_rate": combined_extreme_rate,
            "rate_difference": rate_diff,
            "q_mean_difference": q_diff,
            "normalized_diff_zscore": normalized_diff,
            "zscore_anomaly": z_score_anomaly,
            "severe_mismatch": severe_mismatch
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