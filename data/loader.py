# bipd/data/loader.py

import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Tuple
from utils.logger import BIPDLogger

class DataLoader:
    """
    시장 데이터 로더
    
    yfinance를 사용하여 주식 데이터를 다운로드하고
    캐싱 기능을 제공하여 반복 사용 시 효율성 증대
    """
    
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.logger = BIPDLogger("DataLoader")
        
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
        cache_filename = f"{'_'.join(symbols)}_{start_date}_{end_date}.pkl"
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
        self.logger.info(f"시장 데이터를 다운로드합니다: {symbols} ({start_date} ~ {end_date})")
        
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
        self.logger.debug("데이터를 정제합니다.")
        
        # 결측값 처리
        initial_rows = len(data)
        data = data.dropna()
        
        if len(data) < initial_rows:
            self.logger.info(f"결측값 제거: {initial_rows - len(data)} 행 삭제")
        
        # 이상값 검출 및 처리
        for column in data.columns:
            # 일일 수익률 계산
            returns = data[column].pct_change().dropna()
            
            # 극단적 수익률 (±20%) 제한
            extreme_returns = (returns.abs() > 0.2)
            if extreme_returns.any():
                extreme_dates = returns[extreme_returns].index
                self.logger.warning(
                    f"{column}에서 극단적 수익률 발견: {len(extreme_dates)} 건"
                )
                
                # 극단값을 이전 값으로 대체
                for date in extreme_dates:
                    if date != data.index[0]:  # 첫 번째 날이 아닌 경우
                        prev_date = data.index[data.index < date][-1]
                        data.loc[date, column] = data.loc[prev_date, column]
        
        # 최소 데이터 길이 확인
        if len(data) < 252:  # 1년 미만
            self.logger.warning(f"데이터가 짧습니다: {len(data)} 거래일")
        
        return data
    
    def split_data(self, data: pd.DataFrame, train_end: str, 
                  test_start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        데이터를 훈련/테스트로 분할
        
        Args:
            data: 전체 데이터
            train_end: 훈련 데이터 종료일
            test_start: 테스트 데이터 시작일
            
        Returns:
            train_data, test_data
        """
        try:
            train_data = data[:train_end].copy()
            test_data = data[test_start:].copy()
            
            self.logger.info(
                f"데이터 분할 완료: 훈련={len(train_data)}일, 테스트={len(test_data)}일"
            )
            
            # 데이터 무결성 확인
            if len(train_data) == 0:
                raise ValueError("훈련 데이터가 비어있습니다.")
            if len(test_data) == 0:
                raise ValueError("테스트 데이터가 비어있습니다.")
            
            # 시간적 누수 확인
            last_train_date = train_data.index[-1]
            first_test_date = test_data.index[0]
            
            if last_train_date >= first_test_date:
                self.logger.warning("훈련과 테스트 데이터 간 시간적 중복이 있습니다.")
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"데이터 분할 실패: {e}")
            raise
    
    def get_market_data(self, symbols: List[str], train_start: str, 
                       train_end: str, test_start: str, test_end: str) -> dict:
        """
        완전한 시장 데이터 패키지 생성
        
        Returns:
            dict: {
                'full_data': 전체 데이터,
                'train_data': 훈련 데이터,
                'test_data': 테스트 데이터,
                'symbols': 심볼 리스트
            }
        """
        # 전체 데이터 다운로드
        full_data = self.download_data(symbols, train_start, test_end)
        
        # 훈련/테스트 분할
        train_data, test_data = self.split_data(full_data, train_end, test_start)
        
        return {
            'full_data': full_data,
            'train_data': train_data,
            'test_data': test_data,
            'symbols': symbols
        }