# src/data/validator.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from pathlib import Path
import json
from src.utils.logger import FinFlowLogger

class DataValidator:
    """데이터 무결성 검증 및 정제"""
    
    def __init__(self, config: Optional[Dict] = None):
        # 기본값과 제공된 config를 병합
        default = self._default_config()
        if config:
            default.update(config)
        self.config = default
        self.logger = FinFlowLogger("DataValidator")
        self.validation_report = []

    def _default_config(self) -> Dict:
        """기본 검증 설정"""
        return {
            'min_samples': 252,  # 최소 1년 데이터 (기본값)
            'max_nan_ratio': 0.2,  # 최대 20% 결측치
            'outlier_iqr_factor': 5,  # IQR 5배 이상 이상치
            'extreme_return_threshold': 0.5,  # 일일 ±50% 이상
            'min_variance_threshold': 1e-8,  # 최소 분산
            'max_correlation_threshold': 0.99,  # 최대 상관관계
            'max_time_gap_days': 5,  # 최대 날짜 간격
            'forward_fill_limit': 5,  # forward fill 최대 제한
            'backward_fill_limit': 5,  # backward fill 최대 제한
            'interpolate_limit': 10  # 보간 최대 제한
        }
    
    def validate_and_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """완전한 데이터 검증 및 정제 파이프라인"""
        
        original_shape = data.shape
        self.logger.info(f"데이터 검증 시작: {original_shape}")
        
        # 1. 기본 검증
        data = self._check_basic_integrity(data)
        
        # 2. NaN 처리
        data = self._handle_missing_values(data)
        
        # 3. 이상치 처리
        data = self._handle_outliers(data)
        
        # 4. 데이터 타입 검증
        data = self._validate_dtypes(data)
        
        # 5. 시계열 일관성 검증
        data = self._validate_temporal_consistency(data)
        
        # 6. 통계적 검증
        self._statistical_validation(data)
        
        # 7. 최종 검증
        self._final_validation(data)
        
        self.logger.info(f"검증 완료: {original_shape} → {data.shape}")
        self._generate_validation_report()
        
        return data
    
    def _check_basic_integrity(self, data: pd.DataFrame) -> pd.DataFrame:
        """기본 무결성 검사"""
        # 빈 데이터프레임
        if data.empty:
            raise ValueError("빈 데이터프레임")
        
        # 최소 데이터 요구사항
        min_samples = self.config['min_samples']
        if len(data) < min_samples:
            raise ValueError(f"데이터 부족: {len(data)} < {min_samples}")
        
        # 중복 인덱스
        if data.index.duplicated().any():
            self.logger.warning("중복 인덱스 발견, 제거")
            data = data[~data.index.duplicated(keep='first')]
            self.validation_report.append({
                'issue': 'duplicate_index',
                'count': data.index.duplicated().sum(),
                'action': 'removed_duplicates'
            })
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        nan_count = data.isnull().sum().sum()
        
        if nan_count > 0:
            nan_ratio = nan_count / data.size
            self.logger.warning(f"NaN 발견: {nan_count}개 ({nan_ratio:.2%})")
            
            if nan_ratio > self.config['max_nan_ratio']:
                raise ValueError(f"너무 많은 결측치: {nan_ratio:.2%}")
            
            # Forward fill → Backward fill → 보간
            data = data.ffill(limit=self.config['forward_fill_limit'])
            data = data.bfill(limit=self.config['backward_fill_limit'])
            data = data.interpolate(method='linear', limit=self.config['interpolate_limit'])
            
            # 여전히 NaN이 있으면 해당 행/열 제거
            if data.isnull().any().any():
                # 열 중 50% 이상 NaN인 경우 제거
                thresh = len(data) * 0.5
                data = data.dropna(axis=1, thresh=thresh)
                # 남은 NaN 행 제거
                data = data.dropna()
            
            self.validation_report.append({
                'issue': 'missing_values',
                'count': nan_count,
                'action': 'filled_and_interpolated'
            })
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """이상치 처리"""
        outlier_count = 0
        
        for col in data.columns:
            # IQR 방법
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # 극단적 이상치만 제거 (5 IQR)
            iqr_factor = self.config['outlier_iqr_factor']
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            
            outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            col_outliers = outliers_mask.sum()
            
            if col_outliers > 0:
                outlier_count += col_outliers
                # Winsorization (클리핑)
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        if outlier_count > 0:
            self.logger.warning(f"이상치 발견: {outlier_count}개")
            self.validation_report.append({
                'issue': 'outliers',
                'count': outlier_count,
                'action': 'winsorized'
            })
        
        # Inf 값 처리
        inf_mask = np.isinf(data.values)
        if inf_mask.any():
            inf_count = inf_mask.sum()
            self.logger.warning(f"Inf 값 발견: {inf_count}개, NaN으로 변환 후 처리")
            data = data.replace([np.inf, -np.inf], np.nan)
            data = self._handle_missing_values(data)
            self.validation_report.append({
                'issue': 'infinite_values',
                'count': inf_count,
                'action': 'replaced_with_nan_and_filled'
            })
        
        return data
    
    def _validate_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 타입 검증"""
        # 모든 컬럼이 numeric인지 확인
        non_numeric = []
        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                non_numeric.append(col)
        
        if non_numeric:
            self.logger.warning(f"비숫자형 컬럼 발견: {non_numeric}")
            # 변환 시도
            for col in non_numeric:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            self.validation_report.append({
                'issue': 'non_numeric_columns',
                'columns': non_numeric,
                'action': 'converted_to_numeric'
            })
        
        # Float32로 변환 (메모리 효율)
        data = data.astype(np.float32)
        
        return data
    
    def _validate_temporal_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """시계열 일관성 검증"""
        # 날짜 인덱스 확인
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("날짜 인덱스가 아님, DatetimeIndex로 변환 시도")
            data.index = pd.to_datetime(data.index)
        
        # 시계열 정렬
        if not data.index.is_monotonic_increasing:
            self.logger.warning("시계열 정렬 필요")
            data = data.sort_index()
            self.validation_report.append({
                'issue': 'unsorted_time_series',
                'action': 'sorted_by_index'
            })
        
        # 날짜 간격 확인
        date_diffs = data.index.to_series().diff()
        max_gap = pd.Timedelta(days=self.config['max_time_gap_days'])
        large_gaps = date_diffs[date_diffs > max_gap]
        
        if len(large_gaps) > 0:
            self.logger.warning(f"큰 시간 간격 발견: {len(large_gaps)}개")
            # 큰 간격에 대한 정보 기록
            gap_info = []
            for idx, gap in large_gaps.items():
                gap_info.append({
                    'date': str(idx),
                    'gap_days': gap.days
                })
            
            self.validation_report.append({
                'issue': 'large_time_gaps',
                'count': len(large_gaps),
                'gaps': gap_info[:10]  # 최대 10개만 기록
            })
        
        return data
    
    def _statistical_validation(self, data: pd.DataFrame) -> None:
        """통계적 검증"""
        # 수익률 계산
        returns = data.pct_change().dropna()
        
        # 비현실적 수익률 체크
        extreme_threshold = self.config['extreme_return_threshold']
        extreme_returns = (returns.abs() > extreme_threshold).sum().sum()
        if extreme_returns > 0:
            self.logger.warning(f"극단적 수익률 발견: {extreme_returns}개")
            self.validation_report.append({
                'issue': 'extreme_returns',
                'count': extreme_returns,
                'threshold': extreme_threshold
            })
        
        # 분산 0 체크 (거래 정지)
        min_var_threshold = self.config['min_variance_threshold']
        zero_variance_cols = []
        for col in returns.columns:
            if returns[col].std() < min_var_threshold:
                zero_variance_cols.append(col)
        
        if zero_variance_cols:
            self.logger.warning(f"분산 0인 자산: {zero_variance_cols}")
            self.validation_report.append({
                'issue': 'zero_variance',
                'columns': zero_variance_cols
            })
        
        # 상관관계 1 체크 (중복 자산)
        if len(returns.columns) > 1:
            corr = returns.corr()
            max_corr_threshold = self.config['max_correlation_threshold']
            
            # 대각선 제외하고 높은 상관관계 찾기
            np.fill_diagonal(corr.values, 0)
            high_corr_pairs = []
            
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > max_corr_threshold:
                        high_corr_pairs.append({
                            'asset1': corr.columns[i],
                            'asset2': corr.columns[j],
                            'correlation': corr.iloc[i, j]
                        })
            
            if high_corr_pairs:
                self.logger.warning(f"거의 동일한 자산 쌍: {len(high_corr_pairs)}개")
                self.validation_report.append({
                    'issue': 'high_correlation',
                    'pairs': high_corr_pairs[:10]  # 최대 10개만 기록
                })
    
    def _final_validation(self, data: pd.DataFrame) -> None:
        """최종 검증"""
        # NaN 체크
        if data.isnull().any().any():
            raise ValueError("최종 데이터에 여전히 NaN 존재")
        
        # Inf 체크
        if np.isinf(data.values).any():
            raise ValueError("최종 데이터에 여전히 Inf 존재")
        
        # 최소 샘플 수 재확인
        if len(data) < self.config['min_samples']:
            raise ValueError(f"최종 데이터가 최소 요구사항 미달: {len(data)} < {self.config['min_samples']}")
        
        self.logger.info("최종 검증 통과")
    
    def _generate_validation_report(self) -> Dict:
        """검증 보고서 생성"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'config': self.config,
            'issues_found': len(self.validation_report),
            'details': self.validation_report
        }
        
        # 보고서 저장
        report_path = Path('logs') / 'validation_report.json'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"검증 보고서 저장: {report_path}")
        
        return report
    
    def validate_realtime_data(self, new_data: pd.DataFrame, 
                              historical_stats: Optional[Dict] = None) -> bool:
        """실시간 데이터 검증 (간소화)"""
        # 기본 체크
        if new_data.empty:
            return False
        
        if new_data.isnull().any().any():
            self.logger.warning("실시간 데이터에 NaN 존재")
            return False
        
        if np.isinf(new_data.values).any():
            self.logger.warning("실시간 데이터에 Inf 존재")
            return False
        
        # 통계적 체크 (이전 데이터와 비교)
        if historical_stats:
            for col in new_data.columns:
                value = new_data[col].iloc[-1]
                
                # 이전 평균/표준편차와 비교
                if col in historical_stats:
                    mean = historical_stats[col]['mean']
                    std = historical_stats[col]['std']
                    
                    # 10 시그마 이상 벗어나면 이상
                    z_score = abs((value - mean) / (std + 1e-8))
                    if z_score > 10:
                        self.logger.warning(f"{col} 실시간 데이터 이상: z-score={z_score:.2f}")
                        return False
        
        return True