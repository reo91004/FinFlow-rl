# src/data/loader.py

"""
데이터 로더: yfinance 기반 시장 데이터 수집

목적: 주식 시장 데이터 다운로드 및 캐싱
의존: yfinance, validator.py
사용처: FinFlowTrainer._load_data(), 모든 학습/평가 스크립트
역할: 일관된 데이터 파이프라인 제공

구현 내용:
- yfinance로 다우존스 30 또는 사용자 정의 종목 다운로드
- pickle 캐싱으로 반복 다운로드 방지
- DataValidator 연계로 데이터 품질 검증
- 결측치 처리 (ffill→bfill)
- 학습/검증/테스트 자동 분할
"""

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
    
    def __init__(self, config: Optional[Dict] = None, cache_dir: str = "data/cache",
                 validate_data: bool = True, validation_config: Optional[Dict] = None):
        """
        Args:
            config: 설정 Dict (optional)
            cache_dir: 캐시 디렉토리
            validate_data: 데이터 검증 여부
            validation_config: 검증 설정
        """
        # config Dict가 있으면 우선 사용
        if config is not None:
            self.config = config
            self.cache_dir = config.get('cache_dir', 'data/cache')
            self.validate_data = config.get('validate', True)
            validation_config = config.get('validation', None)
        else:
            self.config = {}
            self.cache_dir = cache_dir
            self.validate_data = validate_data

        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger = FinFlowLogger("DataLoader")
        self.validator = None  # DataValidator removed in IRT refactoring
        
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

        # yfinance로 데이터 다운로드 (연구용: 실패시 즉시 종료)
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

        # 데이터 정제 및 검증
        price_data = self._clean_data(price_data)

        # DataValidator를 사용한 검증
        if self.validate_data and self.validator:
            self.logger.info("데이터 검증 시작...")
            price_data = self.validator.validate_and_clean(price_data)
            self.logger.info("데이터 검증 완료")

        # 캐시 저장 (실패해도 계속 진행)
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
    
    def load(self) -> pd.DataFrame:
        """
        Config 기반 데이터 로드 (trainer.py 호환용)

        Returns:
            price_data: 종가 데이터
        """
        # config에서 파라미터 추출
        symbols = self.config.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])
        interval = self.config.get('interval', '1d')
        use_cache = self.config.get('cache', True)

        # 날짜 설정 (필수)
        start_str = self.config.get('start')
        end_str = self.config.get('end')
        test_end_str = self.config.get('test_end')

        if not start_str or not end_str:
            raise ValueError("data.start와 data.end는 필수 설정입니다")

        # test_end가 있으면 전체 데이터 끝까지 로드
        if test_end_str:
            end_str = test_end_str

        self.logger.info(f"데이터 로드: {symbols} ({start_str} ~ {end_str})")

        # 데이터 다운로드
        return self.download_data(symbols, start_str, end_str, use_cache)

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