# src/data/features.py

import pandas as pd
import numpy as np
import ta
from typing import Optional
from src.utils.logger import FinFlowLogger

class FeatureExtractor:
    """
    12차원 시장 특성 추출기 (기존 BIPD 스타일 유지)
    """
    
    def __init__(self, window: int = 20):
        self.window = window
        self.logger = FinFlowLogger("FeatureExtractor")
        self.logger.info(f"특성 추출기 초기화 - window={window}")
    
    def extract_features(self, price_data: pd.DataFrame, 
                        current_idx: Optional[int] = None) -> np.ndarray:
        """
        12차원 시장 특성 추출
        
        Args:
            price_data: 가격 데이터 (DataFrame)
            current_idx: 현재 시점 인덱스 (None이면 마지막 시점)
            
        Returns:
            features: np.array of shape (12,)
        """
        if current_idx is None:
            current_idx = len(price_data) - 1
        
        # 충분한 데이터가 있는지 확인
        start_idx = max(0, current_idx - self.window + 1)
        if current_idx - start_idx < 5:  # 최소 5일 데이터 필요
            self.logger.warning(f"데이터 부족: {current_idx - start_idx + 1}일")
            return np.zeros(12, dtype=np.float32)
        
        # 현재 시점까지의 데이터 추출
        data_slice = price_data.iloc[start_idx:current_idx + 1]
        
        features = []
        
        # 1. 수익률 통계 (3개)
        returns = data_slice.pct_change().dropna()
        if len(returns) > 0:
            # 평균 수익률로 집계
            mean_returns = returns.mean(axis=1)
            
            # 최근 수익률
            recent_return = mean_returns.iloc[-1] if len(mean_returns) > 0 else 0
            features.append(np.clip(recent_return, -0.1, 0.1))
            
            # 평균 수익률
            avg_return = mean_returns.mean()
            features.append(np.clip(avg_return, -0.1, 0.1))
            
            # 변동성
            volatility = mean_returns.std()
            features.append(np.clip(volatility, 0, 0.1))
        else:
            features.extend([0, 0, 0])
        
        # 2. 기술적 지표 (4개) - 대표 자산으로 계산
        if len(data_slice.columns) > 0:
            # 첫 번째 자산 또는 평균 가격 사용
            if len(data_slice.columns) == 1:
                repr_prices = data_slice.iloc[:, 0]
            else:
                repr_prices = data_slice.mean(axis=1)
            
            # RSI
            if len(repr_prices) >= 14:
                rsi = ta.momentum.RSIIndicator(repr_prices, window=14).rsi()
                rsi_value = rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50
                features.append((rsi_value - 50) / 50)  # 정규화 [-1, 1]
            else:
                features.append(0)
            
            # MACD
            if len(repr_prices) >= 26:
                macd = ta.trend.MACD(repr_prices)
                macd_diff = macd.macd_diff()
                macd_value = macd_diff.iloc[-1] if len(macd_diff) > 0 and not pd.isna(macd_diff.iloc[-1]) else 0
                features.append(np.clip(macd_value / repr_prices.iloc[-1], -0.1, 0.1))
            else:
                features.append(0)
            
            # Bollinger Bands 위치
            if len(repr_prices) >= 20:
                bb = ta.volatility.BollingerBands(repr_prices, window=20)
                bb_high = bb.bollinger_hband()
                bb_low = bb.bollinger_lband()
                if len(bb_high) > 0 and not pd.isna(bb_high.iloc[-1]) and not pd.isna(bb_low.iloc[-1]):
                    bb_width = bb_high.iloc[-1] - bb_low.iloc[-1]
                    if bb_width > 0:
                        bb_position = (repr_prices.iloc[-1] - bb_low.iloc[-1]) / bb_width
                        features.append(np.clip(bb_position, 0, 1))
                    else:
                        features.append(0.5)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
            # 거래량 비율 (더미 - 가격 변동성으로 대체)
            volume_proxy = returns.std(axis=1).iloc[-1] if len(returns) > 0 else 0
            features.append(np.clip(volume_proxy * 10, 0, 1))
        else:
            features.extend([0, 0, 0.5, 0])
        
        # 3. 시장 구조 (3개)
        if len(returns) > 1:
            # 자산 간 상관관계
            if len(data_slice.columns) > 1:
                corr_matrix = returns.corr()
                avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                features.append(np.clip(avg_corr, -1, 1))
            else:
                features.append(0)
            
            # 시장 베타 (첫 번째 자산을 시장으로 가정)
            if len(data_slice.columns) > 1:
                market_returns = returns.iloc[:, 0]
                betas = []
                for col in returns.columns[1:]:
                    if market_returns.std() > 0:
                        beta = returns[col].cov(market_returns) / market_returns.var()
                        betas.append(beta)
                avg_beta = np.mean(betas) if betas else 1.0
                features.append(np.clip(avg_beta, 0, 2))
            else:
                features.append(1.0)
            
            # 최대 낙폭
            cum_returns = (1 + mean_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_dd = drawdown.min()
            features.append(np.clip(max_dd, -0.5, 0))
        else:
            features.extend([0, 1.0, 0])
        
        # 4. 모멘텀 (2개)
        if len(mean_returns) >= 5:
            # 단기 모멘텀 (5일)
            short_momentum = mean_returns.iloc[-5:].mean()
            features.append(np.clip(short_momentum, -0.05, 0.05))
        else:
            features.append(0)
        
        if len(mean_returns) >= self.window:
            # 장기 모멘텀 (20일)
            long_momentum = mean_returns.mean()
            features.append(np.clip(long_momentum, -0.05, 0.05))
        else:
            features.append(0)
        
        # NumPy 배열로 변환
        features = np.array(features, dtype=np.float32)
        
        # NaN 체크
        features = np.nan_to_num(features, 0)
        
        assert len(features) == 12, f"Feature dimension mismatch: {len(features)} != 12"
        
        return features