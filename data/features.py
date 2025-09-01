# bipd/data/features.py

import pandas as pd
import numpy as np
import ta
from typing import Optional
from utils.logger import BIPDLogger

class FeatureExtractor:
    """
    시장 특성 추출기
    
    12차원 시장 특성을 추출하여 BIPD 시스템의 입력으로 사용
    """
    
    def __init__(self, lookback_window: int = 20):
        self.lookback_window = lookback_window
        self.logger = BIPDLogger("FeatureExtractor")
        
        self.logger.info(f"특성 추출기가 초기화되었습니다. 윈도우={lookback_window}")
    
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
        try:
            if current_idx is None:
                current_idx = len(price_data) - 1
            
            # 충분한 데이터가 있는지 확인
            start_idx = max(0, current_idx - self.lookback_window + 1)
            if current_idx - start_idx < 5:  # 최소 5일 데이터 필요
                self.logger.warning(f"데이터가 부족합니다: {current_idx - start_idx + 1}일")
                return np.zeros(12, dtype=np.float32)
            
            # 현재 시점까지의 데이터 추출
            data_slice = price_data.iloc[start_idx:current_idx + 1]
            
            features = []
            
            # 1. 수익률 통계 (3개)
            returns = data_slice.pct_change().dropna()
            if len(returns) > 0:
                portfolio_returns = returns.mean(axis=1)  # 동일가중 포트폴리오 수익률
                
                features.append(portfolio_returns.iloc[-1])  # 최근 수익률
                features.append(portfolio_returns.mean())     # 평균 수익률
                features.append(portfolio_returns.std())      # 변동성
            else:
                features.extend([0.0, 0.0, 0.1])
            
            # 2. 기술적 지표 (4개)
            # 평균 RSI
            rsi_values = []
            for column in data_slice.columns:
                try:
                    rsi = ta.momentum.RSIIndicator(data_slice[column]).rsi()
                    if not rsi.empty and not np.isnan(rsi.iloc[-1]):
                        rsi_values.append(rsi.iloc[-1])
                except:
                    pass
            
            if rsi_values:
                avg_rsi = np.mean(rsi_values) / 100.0  # 정규화
                features.append(avg_rsi)
            else:
                features.append(0.5)  # 중립값
            
            # MACD (시장 인덱스 기준)
            try:
                # 동일가중 시장 인덱스 계산
                market_index = data_slice.mean(axis=1)
                macd_line = ta.trend.MACD(market_index).macd()
                if not macd_line.empty and not np.isnan(macd_line.iloc[-1]):
                    # MACD를 시장 인덱스 대비 비율로 정규화
                    macd_normalized = macd_line.iloc[-1] / market_index.iloc[-1]
                    features.append(np.clip(macd_normalized * 100, -1, 1))
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
            
            # 볼린저 밴드 위치 (평균)
            bb_positions = []
            for column in data_slice.columns:
                try:
                    bb_high = ta.volatility.BollingerBands(data_slice[column]).bollinger_hband()
                    bb_low = ta.volatility.BollingerBands(data_slice[column]).bollinger_lband()
                    
                    if not bb_high.empty and not bb_low.empty:
                        current_price = data_slice[column].iloc[-1]
                        bb_position = (current_price - bb_low.iloc[-1]) / (bb_high.iloc[-1] - bb_low.iloc[-1])
                        bb_positions.append(np.clip(bb_position, 0, 1))
                except:
                    pass
            
            if bb_positions:
                features.append(np.mean(bb_positions))
            else:
                features.append(0.5)
            
            # 거래량 비율 (가능한 경우)
            features.append(0.5)  # 거래량 데이터가 없으므로 중립값
            
            # 3. 시장 구조 (3개)
            # 자산 간 상관관계
            if len(returns) > 5 and returns.shape[1] > 1:
                try:
                    corr_matrix = returns.corr()
                    # 대각선 제외한 평균 상관관계
                    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                    avg_correlation = corr_matrix.values[mask].mean()
                    features.append(np.clip(avg_correlation, -1, 1))
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            # 평균 시장 베타 (전체 종목의 시장 대비 베타 평균)
            if len(returns) > 5:
                try:
                    market_return = returns.mean(axis=1)
                    market_variance = np.var(market_return)
                    
                    if market_variance > 0:
                        betas = []
                        for col in range(returns.shape[1]):
                            asset_return = returns.iloc[:, col]
                            covariance = np.cov(asset_return, market_return)[0, 1]
                            beta = covariance / market_variance
                            if not np.isnan(beta) and np.isfinite(beta):
                                betas.append(beta)
                        
                        if betas:
                            avg_beta = np.mean(betas)
                            features.append(np.clip(avg_beta, 0, 3))
                        else:
                            features.append(1.0)
                    else:
                        features.append(1.0)
                except:
                    features.append(1.0)
            else:
                features.append(1.0)
            
            # 최대 낙폭
            try:
                # NumPy 기반 계산으로 수정 (pandas expanding 메소드 오류 방지)
                portfolio_returns_np = portfolio_returns.values if hasattr(portfolio_returns, 'values') else np.asarray(portfolio_returns)
                cumulative = np.cumprod(1 + portfolio_returns_np)
                rolling_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - rolling_max) / rolling_max
                max_drawdown = abs(np.min(drawdown))
                features.append(np.clip(max_drawdown, 0, 1))
            except:
                features.append(0.0)
            
            # 4. 모멘텀 (2개)
            # 단기 모멘텀 (5일)
            if len(portfolio_returns) >= 5:
                short_momentum = portfolio_returns.tail(5).mean()
                features.append(np.clip(short_momentum * 100, -1, 1))
            else:
                features.append(0.0)
            
            # 장기 모멘텀 (20일)
            if len(portfolio_returns) >= 20:
                long_momentum = portfolio_returns.tail(20).mean()
                features.append(np.clip(long_momentum * 100, -1, 1))
            else:
                features.append(0.0)
            
            # 12개 특성 확인
            while len(features) < 12:
                features.append(0.0)
            
            features = features[:12]  # 정확히 12개만
            
            # NaN/Inf 처리
            features = np.array(features, dtype=np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"특성 추출 실패: {e}")
            return np.zeros(12, dtype=np.float32)
    
    def extract_features_batch(self, price_data: pd.DataFrame) -> np.ndarray:
        """
        전체 데이터에 대해 배치로 특성 추출
        
        Returns:
            features_matrix: np.array of shape (n_days, 12)
        """
        self.logger.info(f"배치 특성 추출을 시작합니다: {len(price_data)} 거래일")
        
        features_list = []
        
        for i in range(self.lookback_window, len(price_data)):
            features = self.extract_features(price_data, i)
            features_list.append(features)
        
        if not features_list:
            self.logger.warning("추출된 특성이 없습니다.")
            return np.array([]).reshape(0, 12)
        
        features_matrix = np.array(features_list)
        
        self.logger.info(f"배치 특성 추출 완료: {features_matrix.shape}")
        
        return features_matrix
    
    def get_feature_names(self) -> list:
        """특성 이름 리스트 반환"""
        return [
            'recent_return',      # 최근 수익률
            'avg_return',         # 평균 수익률  
            'volatility',         # 변동성
            'rsi',               # RSI
            'macd',              # MACD
            'bollinger_position', # 볼린저 밴드 위치
            'volume_ratio',       # 거래량 비율
            'correlation',        # 자산간 상관관계
            'beta',              # 시장 베타
            'max_drawdown',      # 최대 낙폭
            'short_momentum',    # 단기 모멘텀
            'long_momentum'      # 장기 모멘텀
        ]
    
    def get_feature_description(self) -> dict:
        """특성 설명 딕셔너리 반환"""
        names = self.get_feature_names()
        descriptions = [
            "Most recent portfolio return",
            "Average portfolio return over lookback window",
            "Portfolio volatility (standard deviation of returns)",
            "Average RSI across all assets (normalized to [0,1])",
            "MACD indicator normalized by price",
            "Average Bollinger Band position across assets",
            "Volume ratio indicator (placeholder)",
            "Average correlation between assets",
            "Market beta of first asset",
            "Maximum drawdown over lookback window",
            "Short-term momentum (5-day average return)",
            "Long-term momentum (20-day average return)"
        ]
        
        return dict(zip(names, descriptions))