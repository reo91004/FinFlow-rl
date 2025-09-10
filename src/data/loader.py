# src/data/loader.py

import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Dict
from src.utils.logger import FinFlowLogger

class DataLoader:
    """
    시장 데이터 로더 (기존 BIPD 스타일 유지)
    
    yfinance를 사용하여 주식 데이터를 다운로드하고
    캐싱 기능을 제공하여 반복 사용 시 효율성 증대
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.logger = FinFlowLogger("DataLoader")
        
    def download_data(self, symbols: List[str], start_date: str, end_date: str, 
                     use_cache: bool = True) -> pd.DataFrame:
        """
        주식 데이터 다운로드
        
        Args:
            symbols: 주식 심볼 리스트
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            use_cache: 캐시 사용 여부
            
        Returns:
            price_data: DataFrame with adjusted close prices
        """
        # 캐시 파일명
        cache_filename = f"{'_'.join(symbols[:5])}_{len(symbols)}tickers_{start_date}_{end_date}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        # 캐시 확인
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.logger.info(f"캐시에서 데이터를 로드했습니다: {cache_filename}")
                return data
            except Exception as e:
                self.logger.warning(f"캐시 로드 실패: {e}")
        
        # 데이터 다운로드
        self.logger.info(f"시장 데이터를 다운로드합니다: {len(symbols)}개 종목 ({start_date} ~ {end_date})")
        
        try:
            # yfinance로 데이터 다운로드
            raw_data = yf.download(
                symbols, 
                start=start_date, 
                end=end_date,
                progress=False,
                threads=True
            )
            
            if raw_data.empty:
                raise ValueError("다운로드된 데이터가 없습니다.")
            
            # 수정 종가 추출
            if len(symbols) == 1:
                if 'Adj Close' in raw_data.columns:
                    price_data = raw_data[['Adj Close']].copy()
                    price_data.columns = symbols
                else:
                    price_data = raw_data[['Close']].copy()
                    price_data.columns = symbols
                    self.logger.warning("Adj Close가 없어 Close를 사용합니다.")
            else:
                if 'Adj Close' in raw_data.columns.levels[0]:
                    price_data = raw_data['Adj Close'].copy()
                else:
                    price_data = raw_data['Close'].copy()
                    self.logger.warning("Adj Close가 없어 Close를 사용합니다.")
            
            # 데이터 정제
            price_data = self._clean_data(price_data)
            
            # 캐시 저장
            if use_cache:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(price_data, f)
                    self.logger.info(f"데이터를 캐시에 저장했습니다: {cache_filename}")
                except Exception as e:
                    self.logger.warning(f"캐시 저장 실패: {e}")
            
            self.logger.info(
                f"데이터 다운로드 완료: {len(price_data)} 거래일, "
                f"{len(price_data.columns)} 종목"
            )
            
            return price_data
            
        except Exception as e:
            self.logger.error(f"데이터 다운로드 실패: {e}")
            raise
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 정제"""
        # 결측치 처리 (forward fill -> backward fill)
        data = data.ffill().bfill()
        
        # 여전히 NaN이 있는 열 제거
        before_cols = len(data.columns)
        data = data.dropna(axis=1)
        after_cols = len(data.columns)
        
        if before_cols != after_cols:
            self.logger.warning(
                f"결측치가 있는 {before_cols - after_cols}개 종목 제거"
            )
        
        # 거래일 교집합만 사용
        data = data.dropna()
        
        return data
    
    def get_market_data(self, symbols: List[str], 
                       train_start: str, train_end: str,
                       test_start: str, test_end: str,
                       val_split: float = 0.0) -> Dict[str, pd.DataFrame]:
        """
        학습/테스트 데이터 분할 로드
        
        Args:
            symbols: 주식 심볼 리스트
            train_start/train_end: 학습 기간
            test_start/test_end: 테스트 기간
            val_split: 검증 데이터 비율 (0.2 = 20%)
            
        Returns:
            dict with 'train_data', 'test_data', (optional) 'val_data'
        """
        # 전체 기간 데이터 다운로드
        all_data = self.download_data(symbols, train_start, test_end)
        
        # 날짜 기준 분할
        train_mask = (all_data.index >= train_start) & (all_data.index <= train_end)
        test_mask = (all_data.index >= test_start) & (all_data.index <= test_end)
        
        train_data = all_data[train_mask].copy()
        test_data = all_data[test_mask].copy()
        
        result = {
            'train_data': train_data,
            'test_data': test_data
        }
        
        # Validation split (optional)
        if val_split > 0:
            val_size = int(len(train_data) * val_split)
            result['val_data'] = train_data[-val_size:].copy()
            result['train_data'] = train_data[:-val_size].copy()
            
            self.logger.info(
                f"데이터 분할 - Train: {len(result['train_data'])}, "
                f"Val: {len(result['val_data'])}, Test: {len(test_data)}"
            )
        else:
            self.logger.info(
                f"데이터 분할 - Train: {len(train_data)}, Test: {len(test_data)}"
            )
        
        return result