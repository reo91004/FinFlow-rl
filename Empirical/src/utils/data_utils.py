"""
데이터 처리 유틸리티 모듈

주식 데이터를 다운로드하고 전처리하는 기능들을 제공합니다.
기술적 지표 계산, 데이터 캐싱, 벤치마크 데이터 처리 등의 함수들을 포함합니다.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import os
import pickle
import logging
from datetime import datetime
import traceback
from src.constants import FEATURE_NAMES, DATA_SAVE_PATH


def compute_macd(close_series, span_fast=12, span_slow=26):
    """MACD 지표 계산 (Pandas EWM 사용)"""
    ema_fast = close_series.ewm(span=span_fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=span_slow, adjust=False).mean()
    return ema_fast - ema_slow


def compute_rsi(close_series, period=14):
    """RSI 지표 계산 (Pandas Rolling 사용)"""
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # 초기 NaN 방지를 위해 min_periods=period 설정 고려 가능
    avg_gain = gain.rolling(window=period, min_periods=1).mean()  # min_periods=1 추가
    avg_loss = loss.rolling(window=period, min_periods=1).mean()  # min_periods=1 추가
    rs = avg_gain / (avg_loss + 1e-8)  # 0으로 나누기 방지
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50)  # 초기 NaN은 중립값 50으로 채움


def fetch_and_preprocess_data(start_date, end_date, tickers, save_path=DATA_SAVE_PATH):
    """
    주식 데이터를 가져와 전처리합니다.

    먼저 캐시된 데이터를 확인하고, 없을 경우 Yahoo Finance API를 통해 가져옵니다.
    가져온 데이터는 기술적 지표(MACD, RSI, 이동평균 등)를 계산하여 확장합니다.

    Args:
        start_date (str): 시작 날짜
        end_date (str): 종료 날짜
        tickers (list): 주식 티커 리스트
        save_path (str): 데이터 저장 경로

    Returns:
        tuple: (data_array, common_dates)
               - data_array (np.ndarray): 전처리된 데이터 배열 (n_steps, n_assets, n_features)
               - common_dates (pd.DatetimeIndex): 공통 거래일 날짜 인덱스
    """
    logger = logging.getLogger("PortfolioRL")
    os.makedirs(save_path, exist_ok=True)
    tickers_str = "_".join(sorted(tickers))
    data_file = os.path.join(
        save_path, f"portfolio_data_{tickers_str}_{start_date}_{end_date}.pkl"
    )

    # 캐시 파일이 있는지 확인
    if os.path.exists(data_file):
        logger.info(f"캐시 파일 발견: {data_file}")
        try:
            with open(data_file, "rb") as f:
                data_array, common_dates = pickle.load(f)
            
            # 데이터 유효성 검사
            if not isinstance(data_array, np.ndarray) or not isinstance(common_dates, pd.DatetimeIndex):
                logger.warning("캐시된 데이터 타입 오류. 새로 처리합니다.")
            elif data_array.size == 0 or len(common_dates) == 0:
                logger.warning("캐시된 데이터가 비어있습니다. 새로 처리합니다.")
            elif np.isnan(data_array).any():
                logger.warning("캐시 데이터에 NaN 포함됨. nan_to_num 처리 후 반환합니다.")
                data_array = np.nan_to_num(data_array, nan=0.0)
                return data_array, common_dates
            else:
                logger.info(f"캐시 로드 완료. Shape: {data_array.shape}, 날짜 범위: {common_dates[0]} ~ {common_dates[-1]}")
                return data_array, common_dates
        except Exception as e:
            logger.warning(f"캐시 로드 오류 ({e}). 다시 처리합니다.")

    # 데이터 폴더 내의 파일 찾기
    all_pkl_files = [f for f in os.listdir(save_path) if f.startswith("portfolio_data_") and f.endswith(".pkl")]
    
    if all_pkl_files:
        logger.info(f"데이터 폴더에서 {len(all_pkl_files)}개의 포트폴리오 데이터 파일을 발견했습니다.")
        
        # 요청된 모든 티커가 포함된 파일 찾기
        suitable_files = []
        
        for file in all_pkl_files:
            file_tickers = file.replace("portfolio_data_", "").split("_")[0].split("_")
            
            # 파일명에서 티커 추출
            if file.count("_") >= 2:
                file_tickers_part = file.split("_")[1:-2]  # 첫번째(portfolio_data)와 마지막 두개(날짜들) 제외
                file_tickers = "_".join(file_tickers_part).split("_")
            
            # 파일명에 모든 요청 티커가 포함되어 있는지 확인
            all_tickers_included = all(ticker in file for ticker in tickers)
            if all_tickers_included:
                suitable_files.append((file, 1.0))  # 우선순위 1.0 (완벽한 매치)
            else:
                # 부분 매치 - 요청된 티커 중 몇 개가 포함되어 있는지 확인
                matched_tickers = sum(1 for ticker in tickers if any(ticker in part for part in file.split("_")))
                match_ratio = matched_tickers / len(tickers)
                if match_ratio > 0.5:  # 절반 이상 매치되면 사용
                    suitable_files.append((file, match_ratio))
        
        # 매치율이 높은 순으로 정렬
        suitable_files.sort(key=lambda x: x[1], reverse=True)
        
        if suitable_files:
            best_file, match_ratio = suitable_files[0]
            logger.info(f"가장 적합한 데이터 파일 발견: {best_file} (매치율: {match_ratio:.2f})")
            
            try:
                with open(os.path.join(save_path, best_file), "rb") as f:
                    data_array, common_dates = pickle.load(f)
                
                logger.info(f"데이터 파일 로드 성공: {best_file}")
                logger.info(f"데이터 Shape: {data_array.shape}, 날짜 범위: {common_dates[0]} ~ {common_dates[-1]}")
                
                # 날짜 필터링하여 요청된 범위만 반환
                start_date_dt = pd.to_datetime(start_date)
                end_date_dt = pd.to_datetime(end_date)
                
                date_mask = (common_dates >= start_date_dt) & (common_dates <= end_date_dt)
                if date_mask.any():
                    filtered_dates = common_dates[date_mask]
                    filtered_data = data_array[date_mask]
                    logger.info(f"필터링된 데이터 반환: Shape {filtered_data.shape}, 날짜 범위: {filtered_dates[0]} ~ {filtered_dates[-1]}")
                    
                    # 필터링된 데이터 캐싱
                    try:
                        with open(data_file, "wb") as f:
                            pickle.dump((filtered_data, filtered_dates), f)
                        logger.info(f"필터링된 데이터 캐싱 완료: {data_file}")
                    except Exception as e:
                        logger.error(f"필터링된 데이터 캐싱 오류: {e}")
                    
                    return filtered_data, filtered_dates
                else:
                    logger.warning(f"요청된 날짜 범위({start_date} ~ {end_date})에 해당하는 데이터가 없습니다.")
            except Exception as e:
                logger.warning(f"데이터 파일 로드 실패: {e}")

    # 캐시된 데이터가 없으면 Yahoo Finance에서 가져오기
    logger.info(f"yf.download 시작: {len(tickers)} 종목 ({start_date} ~ {end_date})")
    
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=True)
        if raw_data.empty:
            logger.error("yf.download 결과 비어있음.")
            return None, None
        logger.debug("yf.download 완료.")
    except Exception as e:
        logger.error(f"yf.download 중 오류: {e}")
        return None, None

    # 데이터 처리
    processed_dfs = {}
    error_count = 0
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    for ticker in tickers:
        try:
            stock_data_ticker = (
                raw_data.loc[:, pd.IndexSlice[:, ticker]]
                if isinstance(raw_data.columns, pd.MultiIndex)
                else raw_data
            )
            cols_to_use = [
                col for col in required_columns if col in stock_data_ticker.columns
            ]
            if len(cols_to_use) != len(required_columns):
                logger.warning(f"{ticker}: 필요한 컬럼 부족. 건너뜀.")
                error_count += 1
                continue
            stock_data = stock_data_ticker[cols_to_use].copy()
            if isinstance(raw_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)

            # NaN 처리
            if stock_data.isnull().values.any():
                stock_data.ffill(inplace=True)
                stock_data.bfill(inplace=True)

            # NaN이 여전히 남아있을 경우 0으로 채움
            if stock_data.isnull().values.any():
                stock_data.fillna(0, inplace=True)

            # 데이터 전체가 NaN이었는지 다시 확인
            if stock_data.isnull().values.all():
                logger.warning(f"{ticker}: 데이터 처리 후에도 전체 NaN. 건너뜀.")
                error_count += 1
                continue

            # 기술적 지표 계산
            stock_data["MACD"] = compute_macd(stock_data["Close"])
            stock_data["RSI"] = compute_rsi(stock_data["Close"])
            for window in [14, 21, 100]:
                stock_data[f"MA{window}"] = (
                    stock_data["Close"].rolling(window=window, min_periods=1).mean()
                )

            # 최종 NaN 처리
            stock_data.bfill(inplace=True)
            stock_data.ffill(inplace=True)
            stock_data.fillna(0, inplace=True)

            processed_dfs[ticker] = stock_data[FEATURE_NAMES]

        except Exception as e:
            logger.warning(f"{ticker}: 처리 중 오류 - {e}")
            error_count += 1

    valid_tickers = list(processed_dfs.keys())
    if not valid_tickers:
        logger.error("처리 가능한 유효 종목 없음.")
        return None, None
    if error_count > 0:
        logger.warning(f"처리 중 {error_count}개 종목 오류/경고 발생.")

    # 공통 날짜 찾기
    common_dates = pd.to_datetime(
        sorted(
            list(set.intersection(*[set(df.index) for df in processed_dfs.values()]))
        )
    ).tz_localize(None)
    if common_dates.empty:
        logger.error("모든 유효 티커 공통 거래일 없음.")
        return None, None

    # 최종 데이터 배열 생성
    asset_data = [
        processed_dfs[ticker].loc[common_dates].astype(np.float32).values
        for ticker in valid_tickers
    ]
    data_array = np.stack(asset_data, axis=1)
    if np.isnan(data_array).any():
        data_array = np.nan_to_num(data_array, nan=0.0)
    logger.info(
        f"데이터 전처리 완료. Shape: {data_array.shape} ({len(valid_tickers)} 종목)"
    )

    # 데이터 캐싱
    try:
        with open(data_file, "wb") as f:
            pickle.dump((data_array, common_dates), f)
        logger.info(f"전처리 데이터 저장 완료: {data_file}")
    except Exception as e:
        logger.error(f"데이터 캐싱 오류: {e}")

    return data_array, common_dates


def fetch_benchmark_data(
    benchmark_tickers, start_date, end_date, save_path=DATA_SAVE_PATH
):
    """
    벤치마크 지수 데이터를 가져와 처리합니다.

    Args:
        benchmark_tickers (list): 벤치마크 티커 리스트 (예: ["SPY", "QQQ"])
        start_date (str): 시작 날짜
        end_date (str): 종료 날짜
        save_path (str): 데이터 저장 경로

    Returns:
        dict: 각 벤치마크 티커에 대한 데이터프레임을 포함하는 딕셔너리
    """
    logger = logging.getLogger("PortfolioRL")
    logger.info(f"벤치마크 데이터 가져오기: {benchmark_tickers}")
    
    # 벤치마크 티커가 없으면 빈 딕셔너리 반환
    if not benchmark_tickers:
        logger.warning("벤치마크 티커가 지정되지 않았습니다.")
        return {}

    os.makedirs(save_path, exist_ok=True)
    cache_file = os.path.join(
        save_path,
        f'benchmark_data_{"-".join(benchmark_tickers)}_{start_date}_{end_date}.pkl',
    )

    # 캐시된 데이터가 있으면 로드
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                benchmark_data = pickle.load(f)
            
            if not benchmark_data:
                logger.warning("캐시된 벤치마크 데이터가 비어있습니다.")
                return {}
                
            logger.info(f"벤치마크 데이터 캐시 로드 완료: {list(benchmark_data.keys())}")
            return benchmark_data
        except Exception as e:
            logger.warning(f"벤치마크 데이터 캐시 로드 실패: {e}")

    # 데이터 폴더 내의 유사한 파일 찾기 (티커가 동일하고 날짜만 다른 경우)
    similar_files = [f for f in os.listdir(save_path) if f.startswith(f'benchmark_data_{"-".join(benchmark_tickers)}_') and f.endswith(".pkl")]
    
    if similar_files:
        logger.info(f"유사한 벤치마크 데이터 파일 발견: {len(similar_files)}개")
        # 가장 최신의 파일을 사용
        similar_files.sort(reverse=True)
        alt_cache_file = os.path.join(save_path, similar_files[0])
        
        try:
            with open(alt_cache_file, "rb") as f:
                benchmark_data = pickle.load(f)
            
            logger.info(f"대체 벤치마크 데이터 파일 로드 성공: {alt_cache_file}")
            logger.info(f"벤치마크 데이터: {list(benchmark_data.keys())}")
            
            # 새 캐시 파일에 복사
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(benchmark_data, f)
                logger.info(f"벤치마크 데이터 새 캐시 파일에 복사 완료: {cache_file}")
            except Exception as e:
                logger.error(f"벤치마크 데이터 캐싱 오류: {e}")
            
            return benchmark_data
        except Exception as e:
            logger.warning(f"대체 벤치마크 데이터 파일 로드 실패: {e}")

    # 새로 데이터 가져오기
    benchmark_data = {}

    try:
        # 진행 표시줄 없이 데이터 다운로드
        raw_data = yf.download(
            benchmark_tickers, start=start_date, end=end_date, progress=False
        )

        if raw_data.empty:
            logger.error("yf.download 결과 비어있음")
            return {}

        logger.debug(f"다운로드한 데이터 컬럼: {raw_data.columns}")

        # 티커별 데이터 처리
        for ticker in benchmark_tickers:
            try:
                # 여러 티커를 받아온 경우 (MultiIndex)
                if isinstance(raw_data.columns, pd.MultiIndex):
                    # 'Adj Close'가 있는지 확인
                    if ("Adj Close", ticker) in raw_data.columns:
                        ticker_data = raw_data[("Adj Close", ticker)]
                    # 'Close'가 있는지 확인 (대체 방법)
                    elif ("Close", ticker) in raw_data.columns:
                        ticker_data = raw_data[("Close", ticker)]
                        logger.warning(f"{ticker}: 'Adj Close' 없음, 'Close' 사용")
                    else:
                        # 첫 번째 가격 관련 컬럼 사용
                        price_cols = [
                            col
                            for col in raw_data.columns.get_level_values(0)
                            if col in ["Open", "High", "Low", "Close"]
                        ]
                        if price_cols:
                            ticker_data = raw_data[(price_cols[0], ticker)]
                            logger.warning(
                                f"{ticker}: 'Adj Close'/'Close' 없음, '{price_cols[0]}' 사용"
                            )
                        else:
                            logger.warning(f"{ticker}: 적절한 가격 데이터 없음, 건너뜀")
                            continue
                # 단일 티커를 받아온 경우
                else:
                    # 'Adj Close'가 있는지 확인
                    if "Adj Close" in raw_data.columns:
                        ticker_data = raw_data["Adj Close"]
                    # 'Close'가 있는지 확인 (대체 방법)
                    elif "Close" in raw_data.columns:
                        ticker_data = raw_data["Close"]
                        logger.warning(f"{ticker}: 'Adj Close' 없음, 'Close' 사용")
                    else:
                        # 첫 번째 가격 관련 컬럼 사용
                        price_cols = [
                            col
                            for col in raw_data.columns
                            if col in ["Open", "High", "Low", "Close"]
                        ]
                        if price_cols:
                            ticker_data = raw_data[price_cols[0]]
                            logger.warning(
                                f"{ticker}: 'Adj Close'/'Close' 없음, '{price_cols[0]}' 사용"
                            )
                        else:
                            logger.warning(f"{ticker}: 적절한 가격 데이터 없음, 건너뜀")
                            continue

                # 결측치 처리
                if ticker_data.isnull().any():
                    ticker_data = ticker_data.ffill().bfill()

                # 유효 확인
                if ticker_data.empty or ticker_data.isnull().all():
                    logger.warning(f"{ticker}: 유효 데이터 없음, 건너뜀")
                    continue

                benchmark_data[ticker] = ticker_data
                logger.info(f"{ticker} 데이터 처리 완료: {len(ticker_data)} 행")

            except Exception as e:
                logger.warning(f"{ticker} 처리 중 오류: {e}")
                continue

        # 결과 확인
        if not benchmark_data:
            logger.error("처리된 벤치마크 데이터 없음")
            return {}

        # 캐시에 저장
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(benchmark_data, f)
            logger.info(f"벤치마크 데이터 처리 완료 및 캐시 저장: {cache_file}")
        except Exception as e:
            logger.error(f"벤치마크 데이터 캐싱 오류: {e}")

        return benchmark_data

    except Exception as e:
        logger.error(f"벤치마크 데이터 가져오기 실패: {e}")
        logger.error(traceback.format_exc())
        return {}


def calculate_benchmark_performance(benchmark_data, test_dates):
    """
    벤치마크 포트폴리오의 성능을 계산합니다.

    Args:
        benchmark_data (dict): 벤치마크 데이터 딕셔너리
        test_dates (pd.DatetimeIndex): 테스트 기간의 날짜 인덱스

    Returns:
        dict: 각 벤치마크에 대한 성능 지표 딕셔너리
    """
    logger = logging.getLogger("PortfolioRL")
    benchmark_performance = {}

    # 각 벤치마크 처리
    for ticker, data in benchmark_data.items():
        # 테스트 기간과 날짜 맞추기
        aligned_data = data.reindex(test_dates).ffill().bfill()

        if aligned_data.empty or len(aligned_data) < 2:
            logger.warning(f"벤치마크 {ticker}의 데이터가 충분하지 않음")
            continue

        # 초기 투자금액을 1로 정규화하여 가치 계산
        initial_price = aligned_data.iloc[0]
        normalized_values = aligned_data / initial_price

        # 일별 수익률 계산
        daily_returns = normalized_values.pct_change().fillna(0)

        # 성능 지표 계산
        from src.evaluation.evaluation import calculate_performance_metrics
        metrics = calculate_performance_metrics(daily_returns.values)

        benchmark_performance[ticker] = {
            "values": normalized_values.values
            * 1e6,  # 포트폴리오와 동일한 초기 금액으로 스케일링
            "returns": daily_returns.values,
            "metrics": metrics,
        }

    return benchmark_performance 