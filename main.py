import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime
import os
import pickle
import logging
import sys
import gc
import time
from tqdm import tqdm
import traceback  # 오류 로깅을 위해 추가

# --- 상수 정의 ---
# 학습 및 앙상블 설정
NUM_EPISODES = 5000  # 학습 에피소드 수
ENSEMBLE_SIZE = 10  # 앙상블 에이전트 수

# 학습 스케줄 및 Early Stopping 설정
EARLY_STOPPING_PATIENCE = 100  # 성능 향상이 없는 최대 에피소드 수
LR_SCHEDULER_T_MAX = 1000  # Cosine Annealing 주기
LR_SCHEDULER_ETA_MIN = 1e-6  # 최소 학습률
VALIDATION_INTERVAL = 10  # 검증 수행 간격 (에피소드)
VALIDATION_EPISODES = 3  # 검증 시 평가할 에피소드 수

# 벤치마크 설정
BENCHMARK_TICKERS = ["SPY", "QQQ"]  # S&P 500 ETF, Nasdaq 100 ETF
USE_BENCHMARK = True  # 벤치마크 비교 기능 사용 여부

# GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 분석 대상 주식 티커 목록
STOCK_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "AMD",
    "TSLA",
    "JPM",
    "JNJ",
    "PG",
    "V",
]

# 학습/테스트 데이터 기간 설정
TRAIN_START_DATE = "2008-01-02"
TRAIN_END_DATE = "2020-12-31"
TEST_START_DATE = "2021-01-01"
TEST_END_DATE = "2024-12-31"

# 포트폴리오 초기 설정
INITIAL_CASH = 1e6
COMMISSION_RATE = 0.0005  # 수수료 현실화 (0.005 → 0.0005)
# 새로운 매개변수: 행동 변화 페널티 계수
ACTION_PENALTY_COEF = 0.001

# 새로운 행동 스케일링 계수
DIRICHLET_SCALE_FACTOR = 10.0

# 새로운 온도 스케일링 파라미터
SOFTMAX_TEMPERATURE_INITIAL = 1.0
SOFTMAX_TEMPERATURE_MIN = 0.1
SOFTMAX_TEMPERATURE_DECAY = 0.999

# 보상 누적 기간 (K-일)
REWARD_ACCUMULATION_DAYS = 5

# 보상 함수 관련 설정
REWARD_SHARPE_WINDOW = 20  # Sharpe ratio 계산 윈도우
REWARD_RETURN_WEIGHT = 0.6  # 수익률 가중치
REWARD_SHARPE_WEIGHT = 0.4  # Sharpe ratio 가중치
REWARD_DRAWDOWN_PENALTY = 0.2  # 드로우다운 페널티 계수
REWARD_VOL_SCALE_MIN = 0.5  # 변동성 기반 클리핑 최소값
REWARD_VOL_SCALE_MAX = 2.0  # 변동성 기반 클리핑 최대값

# PPO 하이퍼파라미터 (기본값)
DEFAULT_HIDDEN_DIM = 128
DEFAULT_LR = 3e-5  # 학습률 하향 조정 (3e-5 ~ 5e-4 범위에서)
DEFAULT_GAMMA = 0.99
DEFAULT_K_EPOCHS = 10
DEFAULT_EPS_CLIP = 0.1  # 안정성 위해 조정 (0.2 → 0.1)
PPO_UPDATE_TIMESTEP = 4000  # PPO 업데이트 주기 증가 (2000 → 4000)

# 환경 설정
MAX_EPISODE_LENGTH = 504  # 환경의 최대 에피소드 길이 (200 → 504, 2년)

# 상태/보상 정규화 설정
NORMALIZE_STATES = True
CLIP_OBS = 10.0
CLIP_REWARD = 10.0
RMS_EPSILON = 1e-8

# GAE 설정
LAMBDA_GAE = 0.95

# 모델, 데이터 캐시, 결과 저장 경로
MODEL_SAVE_PATH = "models"
# LOG_SAVE_PATH = 'logs' # 제거됨
# PLOT_SAVE_PATH = 'plots' # 제거됨
DATA_SAVE_PATH = "data"
RESULTS_BASE_PATH = "results"  # 새로운 결과 저장 기본 경로

# 설명 가능한 AI (XAI) 관련 설정
INTEGRATED_GRADIENTS_STEPS = 50
XAI_SAMPLE_COUNT = 5  # 통합 그래디언트 분석 샘플 수

# 피처 이름 정의 (데이터 처리 순서와 일치)
FEATURE_NAMES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MACD",
    "RSI",
    "MA14",
    "MA21",
    "MA100",
]


# --- 로깅 설정 ---
def setup_logger(run_dir):
    """
    지정된 실행 디렉토리 내에 로그 파일을 생성하도록 로깅 시스템을 설정합니다.
    파일 핸들러는 INFO 레벨 이상, 콘솔 핸들러는 WARNING 레벨 이상만 출력합니다.
    Args:
        run_dir (str): 로그 파일을 포함할 실행별 결과 디렉토리 경로.
    """
    # os.makedirs(log_dir, exist_ok=True) # main에서 생성하므로 제거
    log_file = os.path.join(run_dir, "training.log")  # 로그 파일 경로 수정

    logger = logging.getLogger("PortfolioRL")
    logger.setLevel(logging.DEBUG)

    # 기존 핸들러 제거
    if logger.hasHandlers():
        logger.handlers.clear()

    # 파일 핸들러 설정 (INFO 레벨 이상, 모든 정보 기록)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)  # 파일에는 INFO 레벨 이상 기록
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러 설정 (WARNING 레벨 이상, 간략 정보)
    console_formatter = logging.Formatter(
        "%(levelname)s: %(message)s"
    )  # 레벨 이름 포함
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # 콘솔에는 WARNING 레벨 이상만 출력
    console_handler.setFormatter(console_formatter)
    # console_handler.addFilter(lambda record: not record.getMessage().startswith('=== 에피소드')) # 필터 제거, 레벨로 제어
    logger.addHandler(console_handler)

    # logger.info -> logger.debug 로 변경 (초기화 메시지는 파일에만 기록)
    logger.debug(f"로거 초기화 완료. 로그 파일: {log_file}")
    return logger


# --- 유틸리티 클래스 ---
class RunningMeanStd:
    """
    Welford's online algorithm을 사용하여 이동 평균과 표준편차를 계산합니다.
    상태 및 보상 정규화에 사용됩니다.
    Source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, epsilon=RMS_EPSILON, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """배치 데이터로 평균과 분산을 업데이트합니다."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """계산된 평균과 분산으로 내부 상태를 업데이트합니다."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class Memory:
    """
    PPO 학습을 위한 경험(Experience) 저장 버퍼입니다.
    NumPy 기반으로 상태, 행동, 로그 확률, 보상, 종료 여부, 상태 가치를 저장합니다.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear_memory(self):
        """메모리에 저장된 모든 경험을 삭제합니다."""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]

    def add_experience(self, state, action, logprob, reward, is_terminal, value):
        """새로운 경험을 메모리에 추가합니다."""
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.values.append(value)


# --- 강화학습 환경 ---
class StockPortfolioEnv(gym.Env):
    """
    주식 포트폴리오 관리를 위한 강화학습 환경입니다.

    - 관측(Observation): 각 자산의 기술적 지표 (10개 피처)
    - 행동(Action): 각 자산에 대한 투자 비중 (0~1 사이 값, 총합 1)
    - 보상(Reward): 포트폴리오 가치의 선형 변화율 (tanh 클리핑 적용)
    - 상태 정규화(State Normalization): RunningMeanStd를 이용한 관측값 및 보상 정규화 기능 포함
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: np.ndarray,
        initial_cash=INITIAL_CASH,
        commission_rate=COMMISSION_RATE,
        max_episode_length=MAX_EPISODE_LENGTH,
        normalize_states=NORMALIZE_STATES,
        gamma=DEFAULT_GAMMA,
        action_penalty_coef=ACTION_PENALTY_COEF,
    ):
        super(StockPortfolioEnv, self).__init__()
        self.data = data  # (n_steps, n_assets, n_features)
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.max_episode_length = max_episode_length
        self.normalize_states = normalize_states
        self.gamma = gamma  # 보상 정규화 시 사용
        self.action_penalty_coef = action_penalty_coef

        self.n_steps, self.n_assets, self.n_features = data.shape

        # 상태 공간 정의
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets, self.n_features),
            dtype=np.float32,
        )

        # 행동 공간 정의 (무위험 자산(현금) 포함 n_assets + 1)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32
        )

        # 상태/보상 정규화 객체 초기화
        if self.normalize_states:
            self.obs_rms = RunningMeanStd(shape=(self.n_assets, self.n_features))
            self.ret_rms = RunningMeanStd(shape=())
            self.returns_norm = np.zeros(1)  # 정규화된 누적 보상 추적
        else:
            self.obs_rms = None
            self.ret_rms = None

        # 내부 상태 변수 초기화 (reset에서 수행)
        self.current_step = 0
        self.cash = 0.0
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)  # 보유 주식 수
        self.portfolio_value = 0.0  # 현재 포트폴리오 가치
        self.weights = np.ones(self.n_assets + 1) / (
            self.n_assets + 1
        )  # 현재 자산 비중 (현금 포함)
        self.prev_weights = np.ones(self.n_assets + 1) / (
            self.n_assets + 1
        )  # 이전 자산 비중 (행동 변화 페널티용)

        # K-일 보상 누적을 위한 가치 이력
        self.portfolio_value_history = []

        # Sharpe ratio 계산을 위한 수익률 이력
        self.returns_history = []

        # 최대 가치 추적 (드로우다운 계산용)
        self.max_portfolio_value = 0.0

        # 시장 변동성 추적 (가변 클리핑용)
        self.market_volatility_window = []
        self.volatility_scaling = 1.0  # 기본값은 1.0 (표준 클리핑)

    def _normalize_obs(self, obs):
        """관측값을 정규화합니다."""
        if not self.normalize_states or self.obs_rms is None:
            return obs
        # RunningMeanStd 업데이트 (차원 맞추기)
        self.obs_rms.update(obs.reshape(1, self.n_assets, self.n_features))
        # 정규화 및 클리핑
        return np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + RMS_EPSILON),
            -CLIP_OBS,
            CLIP_OBS,
        )

    def _normalize_reward(self, reward):
        """보상을 정규화합니다."""
        if not self.normalize_states or self.ret_rms is None:
            return reward
        # 누적 할인 보상 업데이트
        self.returns_norm = self.gamma * self.returns_norm + reward
        # RunningMeanStd 업데이트
        self.ret_rms.update(self.returns_norm)
        # 정규화 및 클리핑
        return np.clip(
            reward / np.sqrt(self.ret_rms.var + RMS_EPSILON), -CLIP_REWARD, CLIP_REWARD
        )

    def _calculate_sharpe_ratio(self, window_size=REWARD_SHARPE_WINDOW):
        """
        최근 window_size 기간의 Sharpe ratio를 계산합니다.

        Args:
            window_size (int): Sharpe ratio 계산에 사용할 기간 길이.

        Returns:
            float: 계산된 Sharpe ratio 값.
        """
        if len(self.returns_history) < 2:  # 최소 2개의 수익률이 필요
            return 0.0

        # 최근 수익률만 사용
        recent_returns = (
            self.returns_history[-window_size:]
            if len(self.returns_history) >= window_size
            else self.returns_history
        )

        # 평균 수익률과 표준편차 계산
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)

        # 표준편차가 0에 가까우면 Sharpe ratio는 정의되지 않음
        if std_return < 1e-8:
            return 0.0 if mean_return < 0 else 10.0  # 양수 수익률이면 높은 값, 음수면 0

        # 일별 Sharpe ratio 계산 (무위험 수익률 0 가정)
        daily_sharpe = mean_return / std_return

        # 연율화 (252 거래일 기준)
        annualized_sharpe = daily_sharpe * np.sqrt(252)

        return annualized_sharpe

    def _calculate_drawdown(self):
        """
        현재 드로우다운을 계산합니다.

        Returns:
            float: 현재 드로우다운 값 (0~1 사이, 높을수록 큰 손실)
        """
        if self.max_portfolio_value <= 1e-8:
            return 0.0

        current_drawdown = 1 - (self.portfolio_value / self.max_portfolio_value)
        return max(0.0, current_drawdown)  # 음수가 되지 않도록

    def _calculate_volatility_scaling(self, window_size=20):
        """
        시장 변동성에 기반한 보상 클리핑 스케일링 값을 계산합니다.
        변동성이 높을수록 클리핑이 약해지고, 낮을수록 강해집니다.

        Args:
            window_size (int): 변동성 계산에 사용할 기간 길이.

        Returns:
            float: 보상 클리핑에 사용할 스케일링 값.
        """
        if len(self.market_volatility_window) < 2:
            return 1.0  # 기본값

        # 최근 변동성 데이터만 사용
        recent_volatility = (
            self.market_volatility_window[-window_size:]
            if len(self.market_volatility_window) >= window_size
            else self.market_volatility_window
        )

        # 평균 변동성 계산
        avg_volatility = np.mean(recent_volatility)

        if avg_volatility < 1e-8:
            return REWARD_VOL_SCALE_MIN  # 변동성이 매우 낮은 경우

        # 변동성이 높을수록 클리핑이 약화됨 (스케일 값이 커짐)
        # 참고: market_volatility_window에는 표준편차 값이 저장되어 있다고 가정
        scaling = np.clip(
            avg_volatility / 0.01, REWARD_VOL_SCALE_MIN, REWARD_VOL_SCALE_MAX
        )

        return scaling

    def reset(self, *, seed=None, options=None, start_index=None):
        """환경을 초기 상태로 리셋합니다."""
        super().reset(seed=seed)
        logger = logging.getLogger("PortfolioRL")

        # 에피소드 시작 인덱스 설정 (데이터 길이 내 무작위 또는 0)
        if start_index is None:
            max_start_index = max(0, self.n_steps - self.max_episode_length)
            if max_start_index == 0:
                start_index = 0
            else:
                start_index = np.random.randint(
                    0, max_start_index + 1
                )  # 0부터 시작 가능하도록 +1
        elif start_index >= self.n_steps:
            logger.warning(
                f"제공된 시작 인덱스({start_index})가 데이터 범위({self.n_steps})를 벗어남. 0으로 설정."
            )
            start_index = 0

        self.current_step = start_index

        # 내부 상태 초기화
        self.cash = self.initial_cash
        self.holdings.fill(0)
        self.portfolio_value = self.cash
        self.weights = np.ones(self.n_assets + 1) / (
            self.n_assets + 1
        )  # 현금 포함 균등 비중
        self.prev_weights = self.weights.copy()

        # 보상 정규화 관련 변수 초기화
        if self.normalize_states:
            self.returns_norm = np.zeros(1)

        # K-일 보상 누적 이력 초기화
        self.portfolio_value_history = [self.portfolio_value]

        # 수익률 이력 초기화
        self.returns_history = []

        # 최대 가치 초기화
        self.max_portfolio_value = self.portfolio_value

        # 시장 변동성 초기화
        self.market_volatility_window = []
        self.volatility_scaling = 1.0

        # 초기 관측값 반환
        observation = self._get_observation()
        normalized_observation = self._normalize_obs(observation)
        info = self._get_info()  # 초기 정보 생성

        return normalized_observation.astype(np.float32), info

    def _get_observation(self):
        """현재 스텝의 원본 관측 데이터를 반환합니다."""
        # 데이터 인덱스 범위 확인 (방어 코드)
        step = min(self.current_step, self.n_steps - 1)  # 범위를 벗어나지 않도록 조정
        return self.data[step]

    def _get_info(self):
        """현재 환경 상태 정보를 담은 딕셔너리를 반환합니다."""
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "weights": self.weights.copy(),
            "return": 0.0,  # 초기 상태에서는 수익률 0
            "raw_reward": 0.0,  # 초기 상태에서는 보상 0
        }

    def step(self, action):
        """
        환경을 한 스텝 진행시킵니다.

        Args:
            action (np.ndarray): 에이전트가 선택한 행동 (자산별 목표 비중, 현금 포함).

        Returns:
            tuple: (next_observation, reward, terminated, truncated, info)
                   - next_observation (np.ndarray): 정규화된 다음 상태 관측값.
                   - reward (float): 정규화된 보상.
                   - terminated (bool): 에피소드 종료 여부 (파산 또는 데이터 끝 도달).
                   - truncated (bool): 에피소드 중단 여부 (최대 길이 도달).
                   - info (dict): 추가 정보 (포트폴리오 가치, 현금, 수익률 등).
        """
        logger = logging.getLogger("PortfolioRL")
        # 행동 크기 확인 및 조정 (현금 슬롯 추가)
        if len(action) == self.n_assets:  # 기존 호환성 유지
            action_with_cash = np.append(action, 0.0)  # 현금 비중 0 추가
        else:
            action_with_cash = action

        # 행동 정규화 (비중 합 1)
        action_with_cash = np.clip(action_with_cash, 0, 1)
        action_sum = action_with_cash.sum()
        if action_sum > 1e-6:
            action_with_cash = action_with_cash / action_sum
        else:  # 비중 합이 0에 가까우면 균등 분배
            action_with_cash = np.ones(self.n_assets + 1) / (self.n_assets + 1)

        # 현재 가격 정보 (원본 데이터 사용)
        current_obs = self._get_observation()
        current_prices = np.maximum(current_obs[:, 3], 1e-6)  # 종가, 0 방지

        # 이전 포트폴리오 가치
        prev_portfolio_value = self.cash + np.dot(self.holdings, current_prices)
        self.portfolio_value_history.append(prev_portfolio_value)

        # 최대 포트폴리오 가치 업데이트
        self.max_portfolio_value = max(self.max_portfolio_value, prev_portfolio_value)

        # 최근 K일 가치만 유지
        if len(self.portfolio_value_history) > REWARD_ACCUMULATION_DAYS + 1:
            self.portfolio_value_history.pop(0)

        # 파산 조건 확인
        if prev_portfolio_value <= 1e-6:
            terminated = True
            truncated = False
            raw_reward = -10.0  # 파산 시 큰 음수 보상
            info = {
                "portfolio_value": 0.0,
                "cash": 0.0,
                "holdings": self.holdings.copy(),
                "weights": np.zeros_like(self.weights),
                "return": -1.0,
                "raw_reward": raw_reward,
            }
            # 마지막 관측값은 현재 관측값 사용 (정규화)
            last_obs_norm = self._normalize_obs(current_obs)
            reward_norm = self._normalize_reward(raw_reward)
            return (
                last_obs_norm.astype(np.float32),
                float(reward_norm),
                terminated,
                truncated,
                info,
            )

        # 행동 변화에 따른 페널티 계산 (L1 거리)
        action_change_penalty = self.action_penalty_coef * np.sum(
            np.abs(action_with_cash - self.prev_weights)
        )

        # 목표 자산 가치 계산 (현금 제외한 주식 부분만)
        stock_weights = action_with_cash[:-1]
        cash_weight = action_with_cash[-1]
        target_value_allocation = stock_weights * prev_portfolio_value

        # 실제 거래 실행 (매수/매도)
        self._execute_trades(target_value_allocation, current_prices)

        # 현금 비중에 따라 현금 조정 (위 거래 후에 남은 현금 비율 조정)
        target_cash = cash_weight * prev_portfolio_value
        # 현재 현금이 목표보다 많으면 유지, 적으면 다른 자산 비중 줄여서 조정은 skip

        # 다음 스텝으로 이동
        self.current_step += 1
        terminated = self.current_step >= self.n_steps  # 종료 조건: 마지막 스텝 이후
        truncated = False  # Truncated는 학습 루프에서 제어

        # 다음 스텝 가격 및 새 포트폴리오 가치 계산
        next_obs_raw = self._get_observation()  # 다음 스텝 관측값 가져오기
        next_prices = np.maximum(next_obs_raw[:, 3], 1e-6)  # 다음 날 종가, 0 방지
        if terminated:
            next_obs_raw = current_obs  # 마지막 스텝이면 현재 관측값 사용
        else:
            next_obs_raw = self._get_observation()  # _get_observation 사용

        next_prices = np.maximum(next_obs_raw[:, 3], 1e-6)  # 다음 날 종가, 0 방지
        self.portfolio_value = self.cash + np.dot(self.holdings, next_prices)

        # 가중치 업데이트 (0으로 나누기 방지), 현금 포함
        if self.portfolio_value > 1e-8:
            stock_weights = (
                self.holdings * next_prices
            ) / self.portfolio_value  # 주식 비중
            cash_weight = self.cash / self.portfolio_value  # 현금 비중
            self.weights = np.append(stock_weights, cash_weight)
        else:
            self.weights = np.zeros(self.n_assets + 1)

        # 이전 가중치 저장 (다음 스텝의 변화 페널티 계산용)
        self.prev_weights = self.weights.copy()

        # 수익률 계산 및 이력 업데이트
        prev_value_safe = max(
            prev_portfolio_value, 1e-8
        )  # 이전 가치가 0에 가까울 때 대비
        current_value_safe = max(
            self.portfolio_value, 0.0
        )  # 현재 가치는 0이 될 수 있음
        daily_return = (current_value_safe / prev_value_safe) - 1
        self.returns_history.append(daily_return)

        # 시장 변동성 업데이트 (최근 수익률의 표준편차)
        if len(self.returns_history) > 1:
            window_size = min(20, len(self.returns_history))
            recent_vol = np.std(self.returns_history[-window_size:])
            self.market_volatility_window.append(recent_vol)
            # 최근 변동성만 저장
            if len(self.market_volatility_window) > 100:  # 충분히 긴 이력 유지
                self.market_volatility_window.pop(0)

            # 변동성 기반 클리핑 스케일 업데이트
            self.volatility_scaling = self._calculate_volatility_scaling()

        # --- 새로운 보상 계산 방식 ---

        # 1. K-일 누적 수익률 계산
        if len(self.portfolio_value_history) > REWARD_ACCUMULATION_DAYS:
            k_day_ago_value = self.portfolio_value_history[
                -REWARD_ACCUMULATION_DAYS - 1
            ]
            if k_day_ago_value > 1e-8:
                k_day_return = (current_value_safe / k_day_ago_value) - 1
            else:
                k_day_return = -1.0
        else:
            # K일치 데이터가 없으면 일간 수익률 사용
            k_day_return = daily_return

        # 2. Sharpe ratio 계산
        sharpe_ratio = self._calculate_sharpe_ratio()

        # 3. 드로우다운 페널티 계산
        drawdown = self._calculate_drawdown()
        drawdown_penalty = REWARD_DRAWDOWN_PENALTY * drawdown

        # 4. 최종 보상 계산 (가중 합)
        # - 수익률 요소: 변동성 스케일링된 tanh(k_day_return)
        # - Sharpe ratio 요소: sharpe_ratio를 -1~1 범위로 클리핑
        # - 드로우다운 페널티: 최대 가치 대비 손실 비율
        # - 행동 변화 페널티: 포트폴리오 조정에 따른 비용

        # 변동성 스케일링 적용 (높은 변동성 = 낮은 클리핑)
        scaled_return = k_day_return * self.volatility_scaling

        # 수익률 기여도 (tanh로 비선형 클리핑)
        return_component = np.tanh(scaled_return) * REWARD_RETURN_WEIGHT

        # Sharpe ratio 기여도 (-1~1 범위로 클리핑)
        sharpe_component = np.clip(sharpe_ratio / 3.0, -1, 1) * REWARD_SHARPE_WEIGHT

        # 최종 보상 계산
        raw_reward = (
            return_component
            + sharpe_component
            - drawdown_penalty
            - action_change_penalty
        )

        if np.isnan(raw_reward) or np.isinf(raw_reward):
            raw_reward = -1.0  # NaN/Inf 발생 시 페널티

        # 다음 상태 및 보상 정규화
        next_obs_norm = self._normalize_obs(next_obs_raw)
        reward_norm = self._normalize_reward(raw_reward)

        # 정보 업데이트
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "weights": self.weights.copy(),
            "return": daily_return,
            "raw_reward": raw_reward,
            "action_penalty": action_change_penalty,
            "k_day_return": k_day_return if "k_day_return" in locals() else 0.0,
            "sharpe_ratio": sharpe_ratio,
            "drawdown": drawdown,
            "volatility_scaling": self.volatility_scaling,
        }

        return (
            next_obs_norm.astype(np.float32),
            float(reward_norm),
            terminated,
            truncated,
            info,
        )

    def _execute_trades(self, target_value_allocation, current_prices):
        """목표 가치 배분에 따라 실제 주식 거래를 실행하고 수수료를 계산합니다."""
        current_value_allocation = self.holdings * current_prices
        trade_value = target_value_allocation - current_value_allocation
        # 0 가격으로 나누는 것 방지됨 (current_prices >= 1e-6)
        shares_to_trade = trade_value / current_prices

        # 거래 순서는 중요하지 않음 (모든 계산은 현재 가격 기준)
        for i in range(self.n_assets):
            shares = shares_to_trade[i]
            price = current_prices[i]
            commission_multiplier = 1 + self.commission_rate

            if shares > 1e-6:  # 매수
                cost = shares * price
                # commission = cost * self.commission_rate # 아래 total_cost 계산에 포함됨
                total_cost = cost * commission_multiplier  # 비용 + 수수료

                # 현금 부족 시 구매 가능 수량 조정
                if total_cost > self.cash + 1e-9:  # 부동 소수점 오차 고려
                    # 가격 * (1+수수료율) 이 0에 가까운 경우 처리
                    if price * commission_multiplier < 1e-8:
                        continue  # 살 수 없음
                    affordable_shares = self.cash / (price * commission_multiplier)
                    if affordable_shares < 1e-6:
                        continue  # 살 수 없음
                    # 실제 구매 가능량으로 조정
                    shares = affordable_shares
                    # total_cost = shares * price * commission_multiplier # 다시 계산
                    total_cost = self.cash  # 가진 현금 전부 사용 (근사)

                self.holdings[i] += shares
                self.cash -= total_cost

            elif shares < -1e-6:  # 매도
                # 실제 팔 수 있는 주식 수는 보유량 이하
                shares_to_sell = min(abs(shares), self.holdings[i])
                if shares_to_sell < 1e-6:
                    continue  # 팔 주식 없음

                revenue = shares_to_sell * price
                commission = revenue * self.commission_rate
                total_revenue = revenue - commission

                self.holdings[i] -= shares_to_sell
                self.cash += total_revenue

        # 거래 후 현금이 음수가 되는 경우 방지 (매우 작은 음수값 처리)
        self.cash = max(self.cash, 0.0)

    def render(self, mode="human"):
        """(선택적) 환경 상태를 간단히 출력합니다."""
        obs = self._get_observation()
        current_prices = obs[:, 3]
        print(f"스텝: {self.current_step}")
        print(f"현금: {self.cash:.2f}")
        print(f"주식 평가액: {np.dot(self.holdings, current_prices):.2f}")
        print(f"총 포트폴리오 가치: {self.portfolio_value:.2f}")

    def close(self):
        """환경 관련 리소스를 정리합니다 (현재는 불필요)."""
        pass


# --- 신경망 모델 ---
class ActorCritic(nn.Module):
    """
    PPO를 위한 액터-크리틱(Actor-Critic) 네트워크입니다.

    - 입력: 평탄화된 상태 (batch_size, n_assets * n_features)
    - LSTM: 시계열 패턴 포착을 위한 순환 레이어
    - 액터 출력: Softmax 기반 포트폴리오 분배 (온도 스케일링 적용)
    - 크리틱 출력: 상태 가치 (State Value)
    """

    def __init__(self, n_assets, n_features, hidden_dim=DEFAULT_HIDDEN_DIM):
        super(ActorCritic, self).__init__()
        self.input_dim = n_assets * n_features
        self.n_assets = n_assets + 1  # 현금 자산 추가
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # 온도 파라미터 (학습 가능)
        self.temperature = nn.Parameter(torch.tensor(SOFTMAX_TEMPERATURE_INITIAL))
        self.temp_min = SOFTMAX_TEMPERATURE_MIN
        self.temp_decay = SOFTMAX_TEMPERATURE_DECAY

        # LSTM 레이어 (시계열 패턴 포착)
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim, batch_first=True
        ).to(DEVICE)

        # 공통 특징 추출 레이어
        self.actor_critic_base = nn.Sequential(
            nn.Linear(hidden_dim * n_assets, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        ).to(DEVICE)

        # 액터 헤드 (로짓 출력)
        self.actor_head = nn.Linear(hidden_dim // 2, self.n_assets).to(DEVICE)

        # 크리틱 헤드 (상태 가치)
        self.critic_head = nn.Linear(hidden_dim // 2, 1).to(DEVICE)

        # 가중치 초기화 적용
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """신경망 가중치를 초기화합니다 (Kaiming He 초기화 사용)."""
        if isinstance(module, nn.Linear):
            # ReLU 활성화 함수에 적합한 Kaiming 초기화
            nn.init.kaiming_uniform_(
                module.weight, a=0, mode="fan_in", nonlinearity="relu"
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)  # 편향은 0으로 초기화
        elif isinstance(module, nn.LSTM):
            # LSTM 초기화
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, 1.0)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)

    def update_temperature(self):
        """학습 과정에서 온도 값을 점진적으로 감소시킵니다."""
        with torch.no_grad():
            # 최소값보다 작아지지 않도록 조정
            self.temperature.mul_(self.temp_decay).clamp_(min=self.temp_min)

    def forward(self, states):
        """
        네트워크의 순전파를 수행합니다.

        Args:
            states (torch.Tensor): 입력 상태 텐서.
                                   (batch_size, n_assets, n_features) 또는 (n_assets, n_features) 형태.

        Returns:
            tuple: (action_probs, value)
                   - action_probs (torch.Tensor): 각 자산에 대한 투자 비중 확률.
                   - value (torch.Tensor): 크리틱 헤드의 출력 (상태 가치).
        """
        batch_size = states.size(0) if states.dim() == 3 else 1

        # 단일 상태인 경우 배치 차원 추가
        if states.dim() == 2:
            states = states.unsqueeze(0)

        # NaN/Inf 입력 방지 (안정성 강화)
        if torch.isnan(states).any() or torch.isinf(states).any():
            # logger.warning(f"ActorCritic 입력에 NaN/Inf 발견. 0으로 대체합니다. Shape: {states.shape}")
            states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

        # LSTM 처리
        # 각 자산별로 피처 시퀀스를 LSTM에 통과시킴
        lstm_outputs = []
        # (batch_size, n_assets, n_features) → 각 자산별로 LSTM 통과
        for i in range(states.size(1)):
            asset_feats = states[:, i, :].view(
                batch_size, 1, -1
            )  # (batch, 1, n_features)
            lstm_out, _ = self.lstm(asset_feats)  # (batch, 1, hidden_dim)
            lstm_outputs.append(
                lstm_out[:, -1, :]
            )  # 마지막 hidden state: (batch, hidden_dim)

        # 모든 자산의 LSTM 출력을 연결
        lstm_concat = torch.cat(lstm_outputs, dim=1)  # (batch, n_assets*hidden_dim)
        lstm_flat = lstm_concat.reshape(batch_size, -1)  # 평탄화

        # 공통 베이스 네트워크 통과
        base_output = self.actor_critic_base(lstm_flat)

        # 액터 출력: 로짓 계산
        logits = self.actor_head(base_output)

        # 온도 스케일링 적용 (낮은 온도 = 더 높은 분산)
        # 온도가 낮을수록 확률 분포는 더 극단적으로 변환됨 (Sparsity 유도)
        scaled_logits = logits / (self.temperature + 1e-8)

        # Softmax로 확률 분포 계산
        action_probs = F.softmax(scaled_logits, dim=-1)

        # 수치적 안정성을 위한 클리핑
        action_probs = torch.clamp(action_probs, min=1e-7, max=1.0)

        # 확률 합이 1이 되도록 정규화
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        # 크리틱 출력: 상태 가치
        value = self.critic_head(base_output)

        return action_probs, value

    def act(self, state):
        """
        주어진 상태에 대해 행동(action), 로그 확률(log_prob), 상태 가치(value)를 반환합니다.
        추론(inference) 시 사용됩니다.

        Args:
            state (np.ndarray): 현재 환경 상태 (정규화된 값).

        Returns:
            tuple: (action, log_prob, value)
                   - action (np.ndarray): 샘플링된 행동 (자산 비중).
                   - log_prob (float): 샘플링된 행동의 로그 확률.
                   - value (float): 예측된 상태 가치.
        """
        # NumPy 배열을 Tensor로 변환하고 배치 차원 추가
        if isinstance(state, np.ndarray):
            # 올바른 형태로 변환 (n_assets, n_features) -> (1, n_assets, n_features) 가정?
            if state.ndim == 2:  # (n_assets, n_features)
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            elif state.ndim == 1:  # 이미 평탄화된 경우? (호환성 위해)
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            else:
                raise ValueError(
                    f"act 메서드: 예상치 못한 NumPy 상태 차원: {state.shape}"
                )
        elif torch.is_tensor(state):
            if state.dim() == 2:
                state_tensor = state.float().unsqueeze(0).to(DEVICE)
            elif state.dim() == 1:
                state_tensor = state.float().unsqueeze(0).to(DEVICE)
            else:
                raise ValueError(
                    f"act 메서드: 예상치 못한 Tensor 상태 차원: {state.shape}"
                )
        else:
            raise TypeError(f"act 메서드: 지원하지 않는 상태 타입: {type(state)}")

        # 그래디언트 계산 비활성화 (추론 모드)
        with torch.no_grad():
            action_probs, value = self.forward(state_tensor)

            # 확률 분포에서 행동 샘플링
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)

            # 인덱스에서 원-핫 인코딩으로 변환 (자산 비중 표현)
            action = torch.zeros_like(action_probs)
            action.scatter_(1, action_idx.unsqueeze(-1), 1.0)

        # 결과를 CPU NumPy 배열 및 스칼라 값으로 변환하여 반환
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def evaluate(self, states, actions):
        """
        주어진 상태(states)와 행동(actions)에 대한 로그 확률(log_prob),
        분포 엔트로피(entropy), 상태 가치(value)를 계산합니다.
        PPO 업데이트 시 사용됩니다.

        Args:
            states (torch.Tensor): 상태 배치.
            actions (torch.Tensor): 행동 배치 (원-핫 인코딩 형태).

        Returns:
            tuple: (log_prob, entropy, value)
                   - log_prob (torch.Tensor): 각 행동의 로그 확률.
                   - entropy (torch.Tensor): 분포의 엔트로피.
                   - value (torch.Tensor): 각 상태의 예측된 가치 (1D Tensor).
        """
        action_probs, value = self.forward(states)

        # 행동이 원-핫 인코딩된 경우, 인덱스로 변환
        if actions.size(-1) == action_probs.size(-1):
            actions_idx = torch.argmax(actions, dim=-1)
        else:
            actions_idx = actions

        # Categorical 분포 생성
        dist = torch.distributions.Categorical(action_probs)

        log_prob = dist.log_prob(actions_idx)
        entropy = dist.entropy()

        # value 텐서의 형태를 (batch_size,)로 일관성 있게 조정
        return log_prob, entropy, value.view(-1)


# --- PPO 알고리즘 구현 ---
class PPO:
    """
    Proximal Policy Optimization (PPO) 알고리즘 클래스입니다.
    Actor-Critic 모델을 사용하여 포트폴리오 관리 문제를 학습합니다.
    """

    def __init__(
        self,
        n_assets,
        n_features,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        lr=DEFAULT_LR,
        gamma=DEFAULT_GAMMA,
        k_epochs=DEFAULT_K_EPOCHS,
        eps_clip=DEFAULT_EPS_CLIP,
        model_path=MODEL_SAVE_PATH,
        logger=None,
        use_ema=True,
        ema_decay=0.99,
        use_lr_scheduler=True,
        use_early_stopping=True,
    ):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.model_path = model_path
        self.logger = logger or setup_logger()  # 로거 없으면 기본 설정 사용
        self.n_assets = n_assets
        self.n_features = n_features  # 추가

        # EMA 가중치 옵션
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # 학습률 스케줄러 및 Early Stopping 설정
        self.use_lr_scheduler = use_lr_scheduler
        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = EARLY_STOPPING_PATIENCE
        self.best_validation_reward = -float("inf")
        self.no_improvement_episodes = 0
        self.should_stop_early = False

        os.makedirs(model_path, exist_ok=True)

        # 정책 네트워크 (현재 정책, 이전 정책)
        self.policy = ActorCritic(n_assets, n_features, hidden_dim).to(DEVICE)
        self.policy_old = ActorCritic(n_assets, n_features, hidden_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())  # 가중치 복사

        # EMA 모델 (학습 안정성을 위한 Exponential Moving Average)
        if self.use_ema:
            self.policy_ema = ActorCritic(n_assets, n_features, hidden_dim).to(DEVICE)
            self.policy_ema.load_state_dict(self.policy.state_dict())
            # EMA 모델의 파라미터는 업데이트되지 않도록 설정
            for param in self.policy_ema.parameters():
                param.requires_grad = False

        # 옵티마이저 및 손실 함수
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()  # 크리틱 손실용

        # 학습률 스케줄러 (Cosine Annealing)
        if self.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=LR_SCHEDULER_T_MAX, eta_min=LR_SCHEDULER_ETA_MIN
            )
            self.logger.info(
                f"Cosine Annealing LR 스케줄러 설정: T_max={LR_SCHEDULER_T_MAX}, eta_min={LR_SCHEDULER_ETA_MIN}"
            )

        self.best_reward = -float("inf")  # 최고 성능 모델 저장을 위한 변수
        self.obs_rms = None  # 학습된 상태 정규화 통계 저장용

        # GPU 설정 (성능 향상 최적화 옵션)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 행렬 곱셈 연산 정밀도 설정 (A100/H100 등 TensorFloat32 지원 시 유리)
            # torch.set_float32_matmul_precision('high') # 또는 'medium'

    def update_lr_scheduler(self):
        """학습률 스케줄러를 업데이트합니다."""
        if self.use_lr_scheduler and self.scheduler:
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            return current_lr
        return None

    def validate(self, env, n_episodes=VALIDATION_EPISODES):
        """
        현재 정책을 검증하여 Early Stopping에 사용할 보상을 계산합니다.

        Args:
            env: 검증에 사용할 환경 (StockPortfolioEnv)
            n_episodes: 실행할 검증 에피소드 수

        Returns:
            float: 평균 검증 보상
        """
        # 평가 모드로 설정
        self.policy_old.eval()
        if self.use_ema:
            self.policy_ema.eval()

        total_reward = 0

        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # EMA 모델 사용 (있는 경우)
                if self.use_ema:
                    with torch.no_grad():
                        state_tensor = (
                            torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
                        )
                        action_probs, _ = self.policy_ema(state_tensor)
                        dist = torch.distributions.Categorical(action_probs)
                        action_idx = dist.sample()

                        # 원-핫 인코딩으로 변환
                        action = torch.zeros_like(action_probs)
                        action.scatter_(1, action_idx.unsqueeze(-1), 1.0)
                        action = action.squeeze(0).cpu().numpy()
                else:
                    action, _, _ = self.policy_old.act(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                episode_reward += info.get("raw_reward", reward)

                if terminated or truncated:
                    done = True
                else:
                    state = next_state

            total_reward += episode_reward

        # 학습 모드로 복원
        self.policy_old.train()
        if self.use_ema:
            self.policy_ema.train()

        # 평균 검증 보상 반환
        return total_reward / n_episodes

    def check_early_stopping(self, validation_reward):
        """
        검증 보상에 기반하여 Early Stopping 여부를 확인합니다.

        Args:
            validation_reward: 현재 검증 보상

        Returns:
            bool: True면 학습 중단, False면 계속 진행
        """
        if not self.use_early_stopping:
            return False

        if validation_reward > self.best_validation_reward:
            # 성능 향상이 있으면 최고 기록 갱신 및 인내심 카운터 리셋
            self.best_validation_reward = validation_reward
            self.no_improvement_episodes = 0
            return False
        else:
            # 성능 향상이 없으면 인내심 카운터 증가
            self.no_improvement_episodes += 1

            # 로깅
            self.logger.info(
                f"최고 검증 보상 {self.best_validation_reward:.4f} 대비 향상 없음. "
                f"인내심 카운터: {self.no_improvement_episodes}/{self.early_stopping_patience}"
            )

            # 인내심 카운터가 임계값을 넘으면 학습 중단
            if self.no_improvement_episodes >= self.early_stopping_patience:
                self.logger.warning(
                    f"Early Stopping 조건 충족! {self.early_stopping_patience} 에피소드 동안 "
                    f"성능 향상 없음. 최고 검증 보상: {self.best_validation_reward:.4f}"
                )
                self.should_stop_early = True
                return True

        return False

    def update_ema_model(self):
        """
        EMA(Exponential Moving Average) 모델의 가중치를 업데이트합니다.
        ema_weight = decay * ema_weight + (1 - decay) * current_weight
        """
        if not self.use_ema:
            return

        with torch.no_grad():
            for ema_param, current_param in zip(
                self.policy_ema.parameters(), self.policy.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    current_param.data, alpha=1.0 - self.ema_decay
                )

    def save_model(self, episode, reward):
        """최고 성능 모델의 가중치와 옵티마이저 상태, obs_rms 통계를 저장합니다."""
        if reward > self.best_reward:
            self.best_reward = reward
            save_file = os.path.join(self.model_path, "best_model.pth")
            try:
                # 저장할 데이터 구성
                checkpoint = {
                    "episode": episode,
                    "model_state_dict": self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_reward": self.best_reward,
                }

                # EMA 모델이 있으면 EMA 상태도 저장
                if self.use_ema:
                    checkpoint["ema_model_state_dict"] = self.policy_ema.state_dict()
                    checkpoint["ema_decay"] = self.ema_decay

                # obs_rms가 있으면 통계량 추가
                if self.obs_rms is not None:
                    checkpoint.update(
                        {
                            "obs_rms_mean": self.obs_rms.mean,
                            "obs_rms_var": self.obs_rms.var,
                            "obs_rms_count": self.obs_rms.count,
                        }
                    )

                torch.save(checkpoint, save_file)
                self.logger.info(
                    f"새로운 최고 성능 모델 저장! 에피소드: {episode}, 보상: {reward:.4f} -> {save_file}"
                )
            except Exception as e:
                self.logger.error(f"모델 저장 중 오류 발생: {e}")

    def load_model(self, model_file=None):
        """저장된 모델 가중치와 옵티마이저 상태, 상태 정규화 통계를 불러옵니다."""
        if model_file is None:
            model_file = os.path.join(self.model_path, "best_model.pth")

        if not os.path.exists(model_file):
            self.logger.warning(f"저장된 모델 파일 없음: {model_file}")
            return False

        try:
            checkpoint = torch.load(model_file, map_location=DEVICE, weights_only=False)

            self.policy.load_state_dict(checkpoint["model_state_dict"])
            self.policy_old.load_state_dict(checkpoint["model_state_dict"])

            # EMA 모델 로드 (있는 경우)
            if self.use_ema and "ema_model_state_dict" in checkpoint:
                self.policy_ema.load_state_dict(checkpoint["ema_model_state_dict"])
                self.ema_decay = checkpoint.get("ema_decay", self.ema_decay)
                self.logger.info(f"EMA 모델 로드 완료 (decay: {self.ema_decay})")
            elif self.use_ema:
                # EMA 모델이 저장되지 않았으면 일반 모델로 초기화
                self.policy_ema.load_state_dict(checkpoint["model_state_dict"])
                self.logger.info("EMA 모델이 저장되지 않아 일반 모델로 초기화됨")

            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self.best_reward = checkpoint.get("best_reward", -float("inf"))

            if "obs_rms_mean" in checkpoint and checkpoint["obs_rms_mean"] is not None:
                if self.obs_rms is None:
                    self.obs_rms = RunningMeanStd(
                        shape=(self.n_assets, self.n_features)
                    )
                self.obs_rms.mean = checkpoint["obs_rms_mean"]
                self.obs_rms.var = checkpoint["obs_rms_var"]
                self.obs_rms.count = checkpoint["obs_rms_count"]
                self.logger.info("저장된 상태 정규화(obs_rms) 통계 로드 완료.")
            else:
                self.obs_rms = None

            self.logger.info(
                f"모델 로드 성공! ({model_file}), 최고 보상: {self.best_reward:.4f}"
            )
            return True

        except (KeyError, TypeError) as load_err:
            self.logger.warning(
                f"모델 파일 로드 중 오류 ({model_file}): {load_err}. 가중치만 로드 시도합니다."
            )
            try:
                weights = torch.load(model_file, map_location=DEVICE, weights_only=True)
                self.policy.load_state_dict(weights)
                self.policy_old.load_state_dict(weights)
                if self.use_ema:
                    self.policy_ema.load_state_dict(weights)
                self.logger.info(
                    f"모델 가중치 로드 성공 (weights_only=True)! ({model_file})"
                )
                self.best_reward = -float("inf")
                self.obs_rms = None
                return True
            except Exception as e_inner:
                self.logger.error(
                    f"weights_only=True 로도 모델 로드 실패 ({model_file}): {e_inner}"
                )
                return False
        except Exception as e:
            self.logger.error(f"모델 로드 중 예상치 못한 오류 발생 ({model_file}): {e}")
            return False

    def select_action(self, state, use_ema=True):
        """
        추론 시 액션 선택 (EMA 모델 사용 옵션 추가)
        use_ema=True면 EMA 모델 사용, False면 일반 모델 사용
        """
        if self.use_ema and use_ema:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
                concentration, _ = self.policy_ema(state_tensor)
                dist = torch.distributions.Dirichlet(concentration)
                action = dist.mean  # 평균값 사용 (샘플링 없이 결정론적)
                return action.squeeze(0).cpu().numpy()
        else:
            action, _, _ = self.policy_old.act(state)
            return action

    def compute_returns_and_advantages(self, rewards, is_terminals, values):
        """
        Generalized Advantage Estimation (GAE)를 사용하여 Advantage와 Return을 계산합니다.

        Args:
            rewards (list): 에피소드/배치에서 얻은 보상 리스트.
            is_terminals (list): 각 스텝의 종료 여부 리스트.
            values (np.ndarray): 각 상태에 대한 크리틱의 가치 예측값 배열.

        Returns:
            tuple: (returns_tensor, advantages_tensor)
                   - returns_tensor (torch.Tensor): 계산된 Return (Target Value).
                   - advantages_tensor (torch.Tensor): 계산된 Advantage.
                   오류 발생 시 빈 텐서 반환.
        """
        if not rewards or values.size == 0:
            self.logger.warning("GAE 계산 시 rewards 또는 values 배열이 비어있습니다.")
            return torch.tensor([], device=DEVICE), torch.tensor([], device=DEVICE)

        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0.0

        next_value = values[-1] * (1.0 - float(is_terminals[-1]))

        for i in reversed(range(len(rewards))):
            mask = 1.0 - float(is_terminals[i])
            delta = rewards[i] + self.gamma * next_value * mask - values[i]
            last_gae_lam = delta + self.gamma * LAMBDA_GAE * mask * last_gae_lam
            advantages[i] = last_gae_lam
            returns[i] = last_gae_lam + values[i]
            next_value = values[i]

        try:
            returns_tensor = torch.from_numpy(returns).float().to(DEVICE)
            advantages_tensor = torch.from_numpy(advantages).float().to(DEVICE)
        except Exception as e:
            self.logger.error(f"Return/Advantage 텐서 변환 중 오류: {e}")
            return torch.tensor([], device=DEVICE), torch.tensor([], device=DEVICE)

        if torch.isnan(returns_tensor).any() or torch.isinf(returns_tensor).any():
            returns_tensor = torch.nan_to_num(returns_tensor, nan=0.0)
        if torch.isnan(advantages_tensor).any() or torch.isinf(advantages_tensor).any():
            advantages_tensor = torch.nan_to_num(advantages_tensor, nan=0.0)

        return returns_tensor, advantages_tensor

    def update(self, memory):
        """메모리에 저장된 경험을 사용하여 정책(policy)을 업데이트합니다."""
        if not memory.states:
            self.logger.warning("업데이트 시도: 메모리가 비어있습니다.")
            return 0.0

        total_loss_val = 0.0

        try:
            old_states = torch.stack(
                [torch.from_numpy(s).float() for s in memory.states]
            ).to(DEVICE)
            old_actions = torch.stack(
                [torch.from_numpy(a).float() for a in memory.actions]
            ).to(DEVICE)
            old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(DEVICE)
            old_values = torch.tensor(memory.values, dtype=torch.float32).to(DEVICE)

            old_values_np = old_values.cpu().numpy()
            returns, advantages = self.compute_returns_and_advantages(
                memory.rewards, memory.is_terminals, old_values_np
            )

            if returns.numel() == 0 or advantages.numel() == 0:
                self.logger.error("GAE 계산 실패로 PPO 업데이트 중단.")
                return 0.0

            adv_mean = advantages.mean()
            adv_std = advantages.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                self.logger.warning("Advantage 정규화 후 NaN/Inf 발견. 0으로 대체.")
                advantages = torch.nan_to_num(advantages, nan=0.0)

            for _ in range(self.k_epochs):
                logprobs, entropy, state_values = self.policy.evaluate(
                    old_states, old_actions
                )
                ratios = torch.exp(logprobs - old_logprobs.detach())
                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values, returns)
                entropy_loss = entropy.mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error(
                        f"손실 계산 중 NaN/Inf 발생! Actor: {actor_loss.item()}, Critic: {critic_loss.item()}, Entropy: {entropy_loss.item()}. 해당 배치 업데이트 건너<0xEB><0x9A><0x8D>니다."
                    )
                    total_loss_val = 0.0
                    break

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_loss_val += loss.item()

                # 온도 파라미터 업데이트
                self.policy.update_temperature()

                # EMA 모델 가중치 업데이트
                if self.use_ema:
                    self.update_ema_model()

            if total_loss_val != 0.0 or self.k_epochs == 0:
                self.policy_old.load_state_dict(self.policy.state_dict())
                return total_loss_val / self.k_epochs if self.k_epochs > 0 else 0.0
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"PPO 업데이트 중 예상치 못한 오류 발생: {e}")
            self.logger.error(traceback.format_exc())
            return 0.0


# --- 설명 가능한 AI (XAI) 관련 함수 ---
def integrated_gradients(model, state, baseline=None, steps=INTEGRATED_GRADIENTS_STEPS):
    """
    통합 그래디언트(Integrated Gradients)를 계산하여 특정 입력 상태에 대한 모델의 특성 기여도를 추정합니다.

    Args:
        model (ActorCritic): ActorCritic 모델 인스턴스.
        state (np.ndarray): 분석할 입력 상태 (n_assets, n_features).
        baseline (np.ndarray, optional): 기준선 상태 (보통 0 벡터). Defaults to None.
        steps (int, optional): 보간 스텝 수. Defaults to INTEGRATED_GRADIENTS_STEPS.

    Returns:
        np.ndarray: 각 특성에 대한 통합 그래디언트 값 (state와 동일한 형태).
                    오류 발생 시 입력 state와 동일한 형태의 0 배열 반환.
    """
    logger = logging.getLogger("PortfolioRL")
    if baseline is None:
        # 이전: 0 벡터 / 변경: 현재 상태의 평균값 사용
        # 가격·거래량과 같은 스케일링되지 않은 피처에 대해 더 의미 있는 기준점 제공
        state_mean = np.mean(state, keepdims=True)
        baseline = np.ones_like(state) * state_mean

    try:
        if state.shape != baseline.shape:
            raise ValueError(
                f"State shape {state.shape}와 Baseline shape {baseline.shape} 불일치"
            )

        state_tensor = torch.from_numpy(state).float().to(DEVICE)
        baseline_tensor = torch.from_numpy(baseline).float().to(DEVICE)
        gradient_sum = torch.zeros_like(state_tensor)
        alphas = torch.linspace(0, 1, steps, device=DEVICE)

        # 모델의 현재 모드 저장
        was_training = model.training

        # cudNN RNN의 backward는 training 모드에서만 작동하므로 임시로 모드 변경
        model.train()

        for alpha in alphas:
            # 1. 원본 형태로 보간
            interpolated_state_orig = baseline_tensor + alpha * (
                state_tensor - baseline_tensor
            )

            # 2. 모델 입력 형태로 변환 (배치 차원 추가)
            interpolated_state_input = interpolated_state_orig.unsqueeze(
                0
            )  # (1, n_assets, n_features)
            interpolated_state_input.requires_grad_(True)

            # 3. 모델 순전파 및 타겟 설정
            # 중요: detach()나 torch.no_grad() 없이 순전파 진행
            concentration, value = model(interpolated_state_input)

            # value에 대해 requires_grad 확인
            if not value.requires_grad:
                value = value.detach().requires_grad_(True)

            # 상태 가치(value)는 기대 반환과 직접적으로 연결되어 있으므로 IG 대상 스칼라로 사용
            target_output = value.squeeze()

            # 4. 그래디언트 계산
            model.zero_grad()
            target_output.backward()

            # 5. 입력에 대한 그래디언트 추출 및 누적
            if interpolated_state_input.grad is not None:
                # 그래디언트는 입력 텐서와 동일한 형태 (1, n_assets, n_features) -> 배치 차원 제거
                gradient = interpolated_state_input.grad.squeeze(0)
                if gradient.shape == state_tensor.shape:
                    gradient_sum += gradient
                else:
                    # 형태가 예상과 다를 경우 경고 로깅
                    logger.warning(
                        f"IG: 그래디언트 형태 불일치 발생. grad shape: {gradient.shape}, expected: {state_tensor.shape}. 해당 스텝 건너<0xEB><0x9A><0x8D>."
                    )
            # else: # grad가 None인 경우, backward 실패 가능성
            # logger.warning(f"IG: Alpha {alpha:.2f}에서 그래디언트가 None입니다.")

        # 원래의 모델 모드로 복원
        if not was_training:
            model.eval()

        # 6. 최종 IG 계산
        integrated_grads_tensor = (state_tensor - baseline_tensor) * (
            gradient_sum / steps
        )
        return integrated_grads_tensor.cpu().numpy()

    except Exception as e:
        logger.error(f"Integrated Gradients 계산 중 오류: {e}")
        logger.error(traceback.format_exc())
        return np.zeros_like(state)


def linear_model_hindsight(features, returns):
    """
    사후 분석(Hindsight)을 위한 간단한 선형 회귀 모델을 학습하고 계수를 반환합니다.

    Args:
        features (np.ndarray): 입력 특성 데이터 (n_steps, n_assets, n_features).
        returns (np.ndarray): 해당 기간의 실제 수익률 데이터 (n_steps,).

    Returns:
        np.ndarray or None: 학습된 선형 모델의 계수 (n_assets, n_features 형태).
                          오류 발생 시 None 반환.
    """
    logger = logging.getLogger("PortfolioRL")
    try:
        if not isinstance(features, np.ndarray) or not isinstance(returns, np.ndarray):
            raise TypeError("입력 데이터는 NumPy 배열이어야 합니다.")

        # 입력 형태 로깅
        logger.info(
            f"선형 모델 입력 형태 - features: {features.shape}, returns: {returns.shape}"
        )

        # features가 3차원인지 확인, 아닐 경우 변환 시도
        if features.ndim == 2:
            # 2차원 배열이면 (n_steps, n_assets*n_features) 형태로 가정
            n_steps = features.shape[0]
            # n_assets와 n_features 추정 (FEATURE_NAMES 길이로)
            n_features_ = len(FEATURE_NAMES)
            n_assets = features.shape[1] // n_features_
            if n_assets * n_features_ != features.shape[1]:
                logger.warning(
                    f"특성 차원 불일치: {features.shape[1]} vs {n_assets}*{n_features_}"
                )
                # 강제로 변환 (데이터 손실 가능)
                features = features[:, : n_assets * n_features_]
            features = features.reshape(n_steps, n_assets, n_features_)
            logger.info(f"특성 데이터 형태 변환: {features.shape}")

        if features.shape[0] != len(returns):
            raise ValueError(
                f"features({features.shape[0]})와 returns({len(returns)})의 샘플 수가 일치하지 않습니다."
            )
        if features.ndim != 3 or returns.ndim != 1:
            raise ValueError(
                f"입력 데이터 차원 오류. Features: {features.ndim}D, Returns: {returns.ndim}D"
            )

        n_steps, n_assets, n_features_ = features.shape

        # 데이터 플랫 버전
        X_flat = features.reshape(n_steps, -1)
        y = returns

        # 피처 스케일링: 각 피처의 표준편차 계산
        feature_std = np.std(X_flat, axis=0)
        # 0에 가까운 표준편차 확인 (제로 방지)
        feature_std[feature_std < 1e-10] = 1.0

        # 데이터 스케일링 (정규화)
        X_scaled = X_flat / feature_std

        # NaN 값 체크 및 처리
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            logger.warning("X 데이터에 NaN/Inf 값 발견, 0으로 대체")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        if np.isnan(y).any() or np.isinf(y).any():
            logger.warning("y 데이터에 NaN/Inf 값 발견, 0으로 대체")
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # 회귀 모델 학습
        model = LinearRegression(fit_intercept=True, normalize=False)
        model.fit(X_scaled, y)

        # 원본 스케일로 계수 변환 (중요)
        scaled_coef = model.coef_ / feature_std

        # 회귀 계수 (자산, 피처)
        beta = scaled_coef.reshape(n_assets, n_features_)

        # 계수 크기 확인 및 로깅
        logger.info(
            f"계수 크기 통계 - 평균: {np.mean(np.abs(beta)):.6f}, 최대: {np.max(np.abs(beta)):.6f}, 최소: {np.min(np.abs(beta)):.6f}"
        )

        # 크기가 너무 작은 경우 스케일링 (가독성 향상)
        max_abs_beta = np.max(np.abs(beta))
        if max_abs_beta < 0.01:
            # 계수가 너무 작을 때 스케일링 (최대 절대값을 0.1로)
            scaling_factor = 0.1 / max_abs_beta
            beta *= scaling_factor
            logger.info(
                f"계수가 너무 작아 {scaling_factor:.2f}배 스케일링 적용. 새 최대값: {np.max(np.abs(beta)):.6f}"
            )

        # 피처 중요도 계산 방법 개선: 절대값 평균을 사용하여 방향 무시
        feature_weights = np.mean(np.abs(beta), axis=0)

        # 결과 정규화: 합이 1이 되도록
        if np.sum(feature_weights) > 1e-10:
            feature_weights = feature_weights / np.sum(feature_weights)

        logger.info(f"선형 모델 학습 완료. 결과 형태: {feature_weights.shape}")
        return feature_weights

    except Exception as e:
        logger.error(f"Hindsight 선형 모델 학습 중 오류: {e}")
        logger.error(traceback.format_exc())
        return None


def compute_feature_weights_drl(ppo_agent, states):
    """
    통합 그래디언트를 사용하여 DRL 에이전트의 각 특성에 대한 중요도(가중치)를 계산합니다.

    Args:
        ppo_agent (PPO): 학습된 PPO 에이전트.
        states (np.ndarray): 분석할 상태 데이터 (n_steps, n_assets, n_features).

    Returns:
        np.ndarray: 각 스텝별 특성 가중치.
                    오류 발생 시 빈 배열 반환.
    """
    logger = logging.getLogger("PortfolioRL")
    all_feature_weights = []

    try:
        if not isinstance(states, np.ndarray) or states.ndim != 3:
            raise ValueError(
                "입력 states는 (n_steps, n_assets, n_features) 형태의 NumPy 배열이어야 합니다."
            )

        # 메모리 효율성을 위한 배치 처리
        batch_size = 32  # 메모리 사용량과 속도의 균형을 위한 적절한 배치 크기
        n_states = len(states)
        n_assets = states.shape[1]
        n_features = states.shape[2]
        n_batches = (n_states + batch_size - 1) // batch_size  # 올림 나눗셈

        logger.info(f"DRL 특성 가중치 계산 중: {n_states}개 샘플, 형태 {states.shape}")

        # 프로그레스 바 설정
        pbar = tqdm(
            range(n_batches),
            desc="Calculating DRL Feature Weights",
            leave=False,
            ncols=100,
        )

        for batch_idx in pbar:
            # 배치 범위 계산
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_states)
            batch_states = states[start_idx:end_idx]

            # 배치 내 각 상태에 대해 IG 계산
            batch_weights = []
            for state in batch_states:
                try:
                    # integrated_gradients 함수는 내부적으로 그래디언트를 계산해야 하므로 no_grad 제거
                    ig = integrated_gradients(ppo_agent.policy, state)

                    # NaN 또는 Inf 값 확인
                    if np.isnan(ig).any() or np.isinf(ig).any():
                        logger.warning("특성 가중치에 NaN/Inf 값 발견. 0으로 대체")
                        ig = np.nan_to_num(ig, nan=0.0, posinf=0.0, neginf=0.0)

                    # 여기서 중요: 각 특성별 중요도 계산
                    # 통합 그래디언트 결과는 (n_assets, n_features) 형태
                    # 이를 특성별로 평균내어 전체 특성 중요도를 구함
                    feature_importance = np.abs(ig).mean(
                        axis=0
                    )  # 자산에 대해 평균, 결과: (n_features,)

                    batch_weights.append(feature_importance)
                except Exception as e:
                    logger.warning(f"샘플에 대한 통합 그래디언트 계산 중 오류: {e}")
                    # 오류 발생 시 0으로 채운 배열 사용
                    batch_weights.append(np.zeros(n_features))

                # 메모리 관리를 위해 주기적으로 캐시 비우기
                if torch.cuda.is_available() and (len(batch_weights) % 10 == 0):
                    torch.cuda.empty_cache()

            # 배치 결과 누적
            all_feature_weights.extend(batch_weights)

            # 배치 처리 후 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not all_feature_weights:
            logger.warning("DRL 특성 가중치 계산 결과가 비어있습니다.")
            return np.zeros((1, n_features))  # 비어있는 경우 0으로 채운 배열 반환

        # 결과 형태 확인 및 로깅 - 각 샘플별 특성 중요도 (n_samples, n_features)
        result = np.stack(all_feature_weights, axis=0)
        logger.info(f"DRL 특성 가중치 계산 완료. 결과 형태: {result.shape}")

        return result

    except Exception as e:
        logger.error(f"DRL 특성 가중치 계산 중 오류: {e}")
        logger.error(traceback.format_exc())
        # 오류 발생 시 에이전트가 있다면 샘플 크기에 맞는 0 배열 반환
        if (
            isinstance(ppo_agent, PPO)
            and isinstance(states, np.ndarray)
            and states.ndim == 3
        ):
            return np.zeros((1, states.shape[2]))
        return np.zeros((1, 10))  # 기본 형태로 0 배열 반환


def compute_correlation(arr1, arr2):
    """
    두 NumPy 배열 간의 피어슨 상관계수를 계산합니다.

    Args:
        arr1 (np.ndarray): 첫 번째 배열.
        arr2 (np.ndarray): 두 번째 배열.

    Returns:
        float: 계산된 피어슨 상관계수. 오류 시 0.0 반환.
    """
    logger = logging.getLogger("PortfolioRL")
    try:
        if (
            not isinstance(arr1, np.ndarray)
            or not isinstance(arr2, np.ndarray)
            or arr1.shape != arr2.shape
            or arr1.size < 2
        ):
            logger.warning(
                f"상관관계 계산 입력 오류: arr1={arr1.shape}, arr2={arr2.shape}"
            )
            return 0.0

        flat1 = arr1.flatten()
        flat2 = arr2.flatten()

        with np.errstate(divide="ignore", invalid="ignore"):
            correlation_matrix = np.corrcoef(flat1, flat2)

        if not isinstance(
            correlation_matrix, np.ndarray
        ) or correlation_matrix.shape != (2, 2):
            logger.warning("상관관계 계산 결과가 유효하지 않음 (표준편차 0 등).")
            return 0.0

        correlation = correlation_matrix[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    except Exception as e:
        logger.error(f"상관관계 계산 중 오류: {e}")
        return 0.0


# --- 데이터 처리 함수 ---
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
    logger = logging.getLogger("PortfolioRL")
    os.makedirs(save_path, exist_ok=True)
    tickers_str = "_".join(sorted(tickers))
    data_file = os.path.join(
        save_path, f"portfolio_data_{tickers_str}_{start_date}_{end_date}.pkl"
    )

    if os.path.exists(data_file):
        logger.debug(f"캐시 로드 시도: {data_file}")
        try:
            with open(data_file, "rb") as f:
                data_array, common_dates = pickle.load(f)
            logger.info(f"캐시 로드 완료. Shape: {data_array.shape}")
            if not isinstance(data_array, np.ndarray) or not isinstance(
                common_dates, pd.DatetimeIndex
            ):
                raise TypeError("캐시된 데이터 타입 오류")
            if np.isnan(data_array).any():
                logger.warning("캐시 데이터에 NaN 포함. nan_to_num 처리.")
                data_array = np.nan_to_num(data_array, nan=0.0)
            return data_array, common_dates
        except Exception as e:
            logger.warning(f"캐시 로드 오류 ({e}). 새로 다운로드.")

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
                logger.warning(f"{ticker}: 필요한 컬럼 부족. 건너<0xEB><0x9A><0x8D>.")
                error_count += 1
                continue
            stock_data = stock_data_ticker[cols_to_use].copy()
            if isinstance(raw_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)

            # if stock_data.isnull().values.any(): stock_data.ffill(inplace=True).bfill(inplace=True)
            # 수정: inplace=True 호출 분리
            if stock_data.isnull().values.any():
                stock_data.ffill(inplace=True)
                stock_data.bfill(inplace=True)

            # NaN이 여전히 남아있을 경우 (ffill/bfill로 채워지지 않는 경우) 0으로 채움
            if stock_data.isnull().values.any():
                stock_data.fillna(0, inplace=True)

            # 데이터 전체가 NaN이었는지 다시 확인 (fillna 후에도 가능성 있음)
            if stock_data.isnull().values.all():  # all() 체크 위치 이동 및 조건 강화
                logger.warning(
                    f"{ticker}: 데이터 처리 후에도 전체 NaN. 건너<0xEB><0x9A><0x8D>."
                )
                error_count += 1
                continue

            stock_data["MACD"] = compute_macd(stock_data["Close"])
            stock_data["RSI"] = compute_rsi(stock_data["Close"])
            for window in [14, 21, 100]:
                stock_data[f"MA{window}"] = (
                    stock_data["Close"].rolling(window=window, min_periods=1).mean()
                )

            # stock_data.bfill(inplace=True).ffill(inplace=True).fillna(0, inplace=True)
            # 수정: 지표 계산 후 NaN 처리 강화
            stock_data.bfill(inplace=True)
            stock_data.ffill(inplace=True)
            stock_data.fillna(0, inplace=True)  # 최종적으로 0으로 채움

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

    common_dates = pd.to_datetime(
        sorted(
            list(set.intersection(*[set(df.index) for df in processed_dfs.values()]))
        )
    ).tz_localize(None)
    if common_dates.empty:
        logger.error("모든 유효 티커 공통 거래일 없음.")
        return None, None

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

    try:
        with open(data_file, "wb") as f:
            pickle.dump((data_array, common_dates), f)
        logger.info(f"전처리 데이터 저장 완료: {data_file}")
    except Exception as e:
        logger.error(f"데이터 캐싱 오류: {e}")

    return data_array, common_dates


# --- 성능 지표 계산 및 시각화 함수 ---
def plot_performance(
    portfolio_values,
    plot_dir,
    dates=None,
    benchmark_values=None,
    title="Portfolio Performance",
    filename=None,
):
    """
    포트폴리오 가치 변화 시각화 및 지정된 디렉토리에 저장.
    Args:
        portfolio_values (list or np.ndarray): 포트폴리오 가치 리스트.
        plot_dir (str): 그래프 이미지를 저장할 디렉토리 경로.
        dates (list or pd.DatetimeIndex, optional): 각 가치에 해당하는 날짜 리스트. X축 레이블용.
        benchmark_values (list or np.ndarray, optional): 벤치마크 가치 리스트 (비교용).
        title (str, optional): 그래프 제목.
        filename (str, optional): 저장할 파일 이름 (확장자 포함). 미지정 시 자동 생성.
    """
    logger = logging.getLogger("PortfolioRL")
    if len(portfolio_values) == 0:
        logger.warning("그래프 생성 실패: 데이터 없음.")
        return

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(14, 7))
    x_axis = (
        dates
        if dates is not None and len(dates) == len(portfolio_values)
        else range(len(portfolio_values))
    )
    xlabel = (
        "Date"
        if dates is not None and len(dates) == len(portfolio_values)
        else "Trading Days"
    )

    plt.plot(x_axis, portfolio_values, label="PPO Portfolio", linewidth=2)
    if benchmark_values is not None and len(benchmark_values) == len(portfolio_values):
        plt.plot(
            x_axis,
            benchmark_values,
            label="Benchmark",
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
        )

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Portfolio Value", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # plot_dir = create_plot_directory() # 제거됨
    if filename is None:
        filename = (
            f'portfolio_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
    save_path = os.path.join(plot_dir, filename)
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"성능 그래프 저장 완료: {save_path}")  # INFO 유지
    except Exception as e:
        logger.error(f"그래프 저장 오류: {e}")
    finally:
        plt.close()


def calculate_performance_metrics(returns):
    """
    일련의 일일 수익률(daily returns)로부터 주요 성능 지표를 계산합니다.

    Args:
        returns (list or np.ndarray): 일일 수익률 리스트 또는 배열.

    Returns:
        dict: 계산된 성능 지표 딕셔너리.
              {'annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio',
               'total_return', 'daily_std'}
    """
    if not isinstance(returns, np.ndarray):
        daily_returns = np.array(returns)
    else:
        daily_returns = returns

    # 유효한 수익률 데이터가 없는 경우 기본값 반환
    if daily_returns.size == 0:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "total_return": 0.0,
            "daily_std": 0.0,
        }

    # NaN/Inf 값 처리
    if np.isnan(daily_returns).any() or np.isinf(daily_returns).any():
        daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)

    # 연간 수익률 (산술 평균 * 252)
    annual_return = np.mean(daily_returns) * 252

    # 연간 변동성 (일간 표준편차 * sqrt(252))
    daily_std = np.std(daily_returns)
    annual_volatility = daily_std * np.sqrt(252)

    # 샤프 비율 (무위험 이자율 0 가정)
    # 변동성이 0에 가까우면 샤프 비율은 정의되지 않거나 0으로 처리
    if annual_volatility > 1e-8:
        sharpe_ratio = annual_return / annual_volatility
    else:
        sharpe_ratio = 0.0

    # 최대 낙폭 (Max Drawdown)
    cumulative_returns = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(cumulative_returns)  # 각 시점까지의 최고점
    drawdown = (
        (peak - cumulative_returns) / peak
        if peak.all() != 0
        else np.zeros_like(cumulative_returns)
    )  # 0으로 나누기 방지
    max_drawdown = np.max(drawdown) if drawdown.size > 0 else 0.0

    # 칼마 비율 (연간 수익률 / 최대 낙폭)
    # 최대 낙폭이 0에 가까우면 칼마 비율은 정의되지 않거나 0으로 처리
    if max_drawdown > 1e-8:
        calmar_ratio = annual_return / max_drawdown
    else:
        calmar_ratio = 0.0

    # 총 수익률 계산
    total_return = (
        (cumulative_returns[-1] - 1) * 100 if len(cumulative_returns) > 0 else 0.0
    )

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "total_return": total_return,
        "daily_std": daily_std,
    }


def plot_feature_importance(
    drl_weights_mean,
    ref_weights_mean,
    plot_dir,
    feature_names=FEATURE_NAMES,
    filename=None,
):
    """
    DRL 에이전트와 참조 모델의 평균 특성 중요도를 막대 그래프로 비교 시각화하고 지정된 디렉토리에 저장.

    Args:
        drl_weights_mean (np.ndarray): DRL 에이전트의 평균 특성 가중치 (n_features,).
        ref_weights_mean (np.ndarray): 참조 모델의 평균 특성 가중치 (n_features,).
        plot_dir (str): 그래프 이미지를 저장할 디렉토리 경로.
        feature_names (list): 특성 이름 리스트.
        filename (str, optional): 저장할 파일 이름. 미지정 시 자동 생성.
    """
    if (
        not isinstance(drl_weights_mean, np.ndarray)
        or not isinstance(ref_weights_mean, np.ndarray)
        or drl_weights_mean.shape != ref_weights_mean.shape
        or len(drl_weights_mean) != len(feature_names)
    ):
        return

    # 값 범위 정규화 - outlier 필터링 및 MinMax 스케일링 적용
    def normalize_weights(weights):
        # Volume 컬럼을 위한 이상치 처리 로직
        weights_abs = np.abs(weights)
        median_abs = np.median(weights_abs)
        mad = np.median(np.abs(weights_abs - median_abs))  # Median Absolute Deviation
        threshold = median_abs + 10 * mad  # 보수적인 임계값

        # 이상치 제한
        clipped_weights = np.clip(weights, -threshold, threshold)

        # 크기가 있는 데이터에 MinMax 스케일링
        if np.max(np.abs(clipped_weights)) > 1e-10:
            scaled_weights = clipped_weights / np.max(np.abs(clipped_weights))
        else:
            scaled_weights = clipped_weights

        return scaled_weights, threshold

    # 공통 정규화: 두 모델 가중치의 절대적 크기 비교가 가능하도록 글로벌 최대값 사용
    # 각 모델별 이상치 제거 후 동일 스케일 적용
    drl_weights_clipped, drl_threshold = normalize_weights(drl_weights_mean)
    ref_weights_clipped, ref_threshold = normalize_weights(ref_weights_mean)

    # 두 모델 가중치의 공통 최대값으로 정규화
    global_max = np.max(
        [np.max(np.abs(drl_weights_clipped)), np.max(np.abs(ref_weights_clipped))]
    )
    if global_max > 1e-10:
        drl_weights_normalized = drl_weights_clipped / global_max
        ref_weights_normalized = ref_weights_clipped / global_max
    else:
        drl_weights_normalized = drl_weights_clipped
        ref_weights_normalized = ref_weights_clipped

    plt.figure(figsize=(15, 7))  # 너비 증가
    num_features = len(feature_names)
    x = np.arange(num_features)

    # DRL 에이전트 중요도
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(x, drl_weights_normalized, color="skyblue")
    plt.ylabel("Normalized Importance Score")
    plt.title(
        f"DRL Agent Feature Importance\n(Clipped at {drl_threshold:.2e}, Common Scale)"
    )
    plt.xticks(x, feature_names, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--")
    plt.ylim(-1.1, 1.1)  # 정규화된 값의 y 축 범위 고정
    # 막대 위에 값 표시 (소수점 2자리)
    for bar in bars1:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.2f}",
            va="bottom" if yval >= 0 else "top",
            ha="center",
            fontsize=8,
        )

    # 참조 모델 중요도
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(x, ref_weights_normalized, color="lightcoral")
    plt.ylabel("Normalized Importance Score")
    plt.title(
        f"Reference Model Feature Importance\n(Clipped at {ref_threshold:.2e}, Common Scale)"
    )
    plt.xticks(x, feature_names, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--")
    plt.ylim(-1.1, 1.1)  # 정규화된 값의 y 축 범위 고정
    # 막대 위에 값 표시 (소수점 2자리)
    for bar in bars2:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.2f}",
            va="bottom" if yval >= 0 else "top",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()

    # 파일 저장
    if filename is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feature_importance_comparison_{current_time}.png"
    save_path = os.path.join(plot_dir, filename)

    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # bbox_inches 추가
    except Exception as e:
        pass  # 오류 로깅은 주석 처리됨
    finally:
        plt.close()


def plot_integrated_gradients(
    ig_values_mean,
    plot_dir,
    feature_names=FEATURE_NAMES,
    title="Mean Integrated Gradients",
    filename=None,
):
    """
    평균 통합 그래디언트 값을 막대 그래프로 시각화하고 지정된 디렉토리에 저장.

    Args:
        ig_values_mean (np.ndarray): 평균 통합 그래디언트 값 배열 (n_features,).
        plot_dir (str): 그래프 이미지를 저장할 디렉토리 경로.
        feature_names (list): 특성 이름 리스트.
        title (str, optional): 그래프 제목.
        filename (str, optional): 저장할 파일 이름. 미지정 시 자동 생성.
    """
    if not isinstance(ig_values_mean, np.ndarray) or len(ig_values_mean) != len(
        feature_names
    ):
        return  # 데이터 오류 시 함수 종료

    # 극단값을 제한하고 정규화하기
    values_abs = np.abs(ig_values_mean)
    median_abs = np.median(values_abs)
    mad = np.median(np.abs(values_abs - median_abs))
    threshold = median_abs + 10 * mad

    # 원본 값 보존 및 clipping
    original_values = ig_values_mean.copy()
    normalized_values = np.clip(ig_values_mean, -threshold, threshold)

    # 크기가 있는 데이터에 MinMax 스케일링 (-1~1 범위로)
    if np.max(np.abs(normalized_values)) > 1e-10:
        normalized_values = normalized_values / np.max(np.abs(normalized_values))

    # 파일 경로 생성
    if filename is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        norm_filename = f"integrated_gradients_mean_{current_time}.png"
        raw_filename = f"integrated_gradients_raw_{current_time}.png"
    else:
        file_base, file_ext = os.path.splitext(filename)
        norm_filename = filename
        raw_filename = f"{file_base}_raw{file_ext}"

    norm_save_path = os.path.join(plot_dir, norm_filename)
    raw_save_path = os.path.join(plot_dir, raw_filename)

    num_features = len(feature_names)
    x = np.arange(num_features)

    # 1. 원본 값 그래프 생성 및 저장
    plt.figure(figsize=(12, 6))
    plt.bar(x, original_values, color="lightsteelblue", alpha=0.7)
    plt.ylabel("Raw Attribution Score")
    plt.title(f"Original {title} (Unclipped)")
    plt.xticks(x, feature_names, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()

    try:
        plt.savefig(raw_save_path, dpi=300, bbox_inches="tight")
    except Exception as e:
        pass
    finally:
        plt.close()

    # 2. 정규화 값 그래프 생성 및 저장
    plt.figure(figsize=(12, 6))
    bars = plt.bar(x, normalized_values, color="mediumpurple")
    plt.ylabel("Normalized Attribution Score")
    plt.title(f"{title}\n(Normalized, clipped at {threshold:.2e})")
    plt.xticks(x, feature_names, rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--")
    plt.ylim(-1.1, 1.1)

    # 막대 위에 값 표시
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.2f}",
            va="bottom" if yval >= 0 else "top",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()

    try:
        plt.savefig(norm_save_path, dpi=300, bbox_inches="tight")
    except Exception as e:
        pass  # 오류 로깅은 주석 처리됨
    finally:
        plt.close()


# --- 학습 및 평가 함수 ---
def print_memory_stats(logger):
    """현재 GPU 메모리 사용량 및 캐시 상태를 로깅합니다."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(
            f"GPU 메모리 사용량: {allocated:.2f} MB / 예약됨: {reserved:.2f} MB"
        )


def train_ppo_agent(
    env: StockPortfolioEnv,
    ppo_agent: PPO,
    max_episodes: int,
    max_timesteps: int,
    update_timestep: int,
    logger: logging.Logger,
    validation_env=None,
):
    """
    주어진 환경에서 PPO 에이전트를 학습시킵니다.
    주기적 상세 로그는 DEBUG 레벨로 변경, 최종 결과는 INFO 유지.

    Args:
        env (StockPortfolioEnv): 학습에 사용할 환경
        ppo_agent (PPO): 학습할 PPO 에이전트
        max_episodes (int): 최대 학습 에피소드 수
        max_timesteps (int): 각 에피소드의 최대 스텝 수
        update_timestep (int): 정책 업데이트 주기 (스텝 단위)
        logger (logging.Logger): 로깅에 사용할 로거
        validation_env (StockPortfolioEnv, optional): 검증에 사용할 환경. None이면 학습 환경 사용.
    """
    logger.info(
        f"PPO 학습 시작: {max_episodes} 에피소드, 에피소드당 최대 {max_timesteps} 스텝"
    )
    logger.info(f"정책 업데이트 주기: {update_timestep} 스텝")

    # 검증 환경이 제공되지 않으면 학습 환경 사용
    if validation_env is None:
        validation_env = env

    memory = Memory()
    episode_raw_rewards = []
    training_start_time = time.time()
    total_steps = 0
    update_count = 0

    # Early Stopping 설정 로깅
    if ppo_agent.use_early_stopping:
        logger.info(
            f"Early Stopping 활성화: 인내심 {ppo_agent.early_stopping_patience} 에피소드, "
            f"검증 간격 {VALIDATION_INTERVAL} 에피소드"
        )

    pbar = tqdm(
        range(max_episodes), desc="Training Episodes", file=sys.stdout, ncols=100
    )

    for episode in pbar:
        # GPU 메모리 정리
        if episode % 20 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # 환경 초기화
        state, _ = env.reset()
        episode_raw_reward_sum = 0.0
        current_episode_steps = 0

        logger.debug(f"--- 에피소드 {episode+1} 시작 ---")  # DEBUG로 변경

        terminated = False
        truncated = False
        while not terminated and not truncated:
            total_steps += 1
            current_episode_steps += 1
            action, log_prob, value = ppo_agent.policy_old.act(state)
            next_state, reward_norm, terminated, truncated_env, info = env.step(action)
            is_terminal_step = terminated or (current_episode_steps >= max_timesteps)
            memory.add_experience(
                state, action, log_prob, reward_norm, is_terminal_step, value
            )
            state = next_state
            episode_raw_reward_sum += info.get("raw_reward", 0.0)

            # 정책 업데이트
            if (
                total_steps % update_timestep == 0
                and len(memory.states) >= ppo_agent.k_epochs
            ):
                update_loss = ppo_agent.update(memory)
                update_count += 1
                memory.clear_memory()

                # 학습률 스케줄러 업데이트
                if ppo_agent.use_lr_scheduler:
                    current_lr = ppo_agent.update_lr_scheduler()
                    logger.debug(
                        f" 정책 업데이트 {update_count}, 현재 학습률: {current_lr:.7f}"
                    )
                else:
                    logger.debug(
                        f" 정책 업데이트 {update_count} (총 {total_steps} 스텝). Loss: {update_loss:.4f}"
                    )

            if current_episode_steps >= max_timesteps:
                truncated = True

        # 에피소드 완료
        episode_raw_rewards.append(episode_raw_reward_sum)
        ppo_agent.save_model(episode, episode_raw_reward_sum)

        # 주기적 로깅
        if (episode + 1) % 5 == 0 or episode == max_episodes - 1:
            lookback = max(10, len(episode_raw_rewards) // 10)
            avg_raw_reward = np.mean(episode_raw_rewards[-lookback:])
            final_value = info.get("portfolio_value", env.portfolio_value)
            pbar.set_postfix(
                {
                    f"AvgRew(raw, L{lookback})": f"{avg_raw_reward:.2f}",
                    "LastValue": f"{final_value:,.0f}",
                    "Steps": f"{total_steps:,}",
                },
                refresh=True,
            )

            # 주기적 상세 로그는 DEBUG
            if (episode + 1) % 50 == 0:
                logger.debug(
                    f" 에피소드 {episode+1} 완료. 최근 {lookback} 평균 Raw 보상: {avg_raw_reward:.4f}, 최종 가치: {final_value:.2f}"
                )

        # 주기적 검증 및 Early Stopping 확인
        if ppo_agent.use_early_stopping and (episode + 1) % VALIDATION_INTERVAL == 0:
            validation_reward = ppo_agent.validate(validation_env)
            logger.info(
                f"검증 에피소드 {episode+1}: 평균 보상 {validation_reward:.4f} (최고: {ppo_agent.best_validation_reward:.4f})"
            )

            # Early Stopping 조건 확인
            should_stop = ppo_agent.check_early_stopping(validation_reward)
            if should_stop:
                logger.info(
                    f"Early Stopping으로 학습 중단 (에피소드 {episode+1}/{max_episodes})"
                )
                break

    pbar.close()

    total_training_time = time.time() - training_start_time
    avg_step_time = total_training_time / total_steps if total_steps > 0 else 0
    logger.info(
        f"\n총 학습 시간: {total_training_time:.2f}초 ({avg_step_time:.4f}초/스텝)"
    )  # 최종 결과는 INFO

    if hasattr(env, "obs_rms") and env.normalize_states and env.obs_rms is not None:
        ppo_agent.obs_rms = env.obs_rms
        logger.info(
            "학습 환경의 상태 정규화(obs_rms) 통계를 에이전트에 저장했습니다."
        )  # INFO 유지
    else:
        logger.warning(
            "학습 환경의 obs_rms 통계를 에이전트에 저장하지 못했습니다."
        )  # 경고 유지

    return episode_raw_rewards


def evaluate_ppo_agent(
    env: StockPortfolioEnv,
    ppo_agent: PPO,
    max_test_timesteps: int,
    load_best_model=True,
):
    """
    학습된 PPO 에이전트를 평가합니다.
    평가 시작/종료 메시지만 INFO로 유지.
    """
    logger = ppo_agent.logger

    if load_best_model:
        if not ppo_agent.load_model():
            logger.error("모델 로드 실패, 평가 중단.")
            return None

    # 평가 환경 리셋 시 시작 인덱스를 0으로 고정
    state, info_init = env.reset(start_index=0)  # start_index=0 추가
    total_raw_reward = 0.0
    portfolio_values = [info_init["portfolio_value"]]
    daily_returns = []
    asset_weights = [info_init["weights"]]
    chosen_actions = []

    terminated, truncated = False, False
    step_count = 0

    logger.info("PPO 에이전트 평가 시작 (결정론적 행동)...")  # INFO 유지
    pbar_eval = tqdm(
        total=max_test_timesteps, desc="Evaluating Agent", file=sys.stdout, ncols=100
    )

    while not terminated and not truncated and step_count < max_test_timesteps:
        normalized_state = state
        if ppo_agent.obs_rms is not None and ppo_agent.obs_rms.count > RMS_EPSILON:
            normalized_state = np.clip(
                (state - ppo_agent.obs_rms.mean)
                / np.sqrt(ppo_agent.obs_rms.var + RMS_EPSILON),
                -CLIP_OBS,
                CLIP_OBS,
            )

        # EMA 모델 사용하여 액션 선택 (결정론적)
        action = ppo_agent.select_action(normalized_state, use_ema=True)

        chosen_actions.append(action)
        next_state, _, terminated, truncated_env, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        daily_returns.append(info["return"])
        asset_weights.append(info.get("weights", np.zeros(env.action_space.shape[0])))
        total_raw_reward += info.get("raw_reward", 0.0)
        state = next_state
        step_count += 1
        pbar_eval.update(1)
        if step_count >= max_test_timesteps:
            truncated = True

    pbar_eval.close()
    logger.info(f"평가 종료. 총 스텝: {step_count}")  # INFO 유지

    return {
        "episode_reward": total_raw_reward,
        "portfolio_values": portfolio_values,
        "returns": daily_returns,
        "weights": asset_weights,
        "actions": chosen_actions,
    }


def evaluate_ensemble(
    env: StockPortfolioEnv, agents: list, max_test_timesteps: int, load_best_model=True
):
    """
    앙상블 에이전트 평가 - 여러 에이전트의 평균 행동을 사용

    Args:
        env: 평가 환경
        agents: PPO 에이전트 리스트
        max_test_timesteps: 최대 평가 스텝 수
        load_best_model: 각 에이전트의 베스트 모델 로드 여부

    Returns:
        dict: 평가 결과
    """
    logger = agents[0].logger if agents else logging.getLogger("PortfolioRL")

    if not agents:
        logger.error("앙상블 평가를 위한 에이전트가 없음.")
        return None

    # 각 에이전트의 모델 로드
    if load_best_model:
        for i, agent in enumerate(agents):
            if not agent.load_model():
                logger.warning(f"앙상블 에이전트 {i+1} 모델 로드 실패. 스킵됨.")
                return None

    # 평가 환경 리셋
    state, info_init = env.reset(start_index=0)
    total_raw_reward = 0.0
    portfolio_values = [info_init["portfolio_value"]]
    daily_returns = []
    asset_weights = [info_init["weights"]]
    chosen_actions = []

    terminated, truncated = False, False
    step_count = 0

    logger.info(f"앙상블 평가 시작 ({len(agents)}개 에이전트)...")
    pbar_eval = tqdm(
        total=max_test_timesteps, desc="Evaluating Ensemble", file=sys.stdout, ncols=100
    )

    while not terminated and not truncated and step_count < max_test_timesteps:
        # 각 에이전트별 정규화된 상태 준비
        agent_actions = []
        for agent in agents:
            normalized_state = state
            if agent.obs_rms is not None and agent.obs_rms.count > RMS_EPSILON:
                normalized_state = np.clip(
                    (state - agent.obs_rms.mean)
                    / np.sqrt(agent.obs_rms.var + RMS_EPSILON),
                    -CLIP_OBS,
                    CLIP_OBS,
                )

            # 각 에이전트의 결정론적 액션 선택 (EMA 사용)
            action = agent.select_action(normalized_state, use_ema=True)
            agent_actions.append(action)

        # 앙상블 액션 - 단순 평균
        ensemble_action = np.mean(agent_actions, axis=0)

        # 정규화 (합이 1이 되도록)
        if ensemble_action.sum() > 1e-6:
            ensemble_action = ensemble_action / ensemble_action.sum()
        else:
            ensemble_action = np.ones_like(ensemble_action) / len(ensemble_action)

        chosen_actions.append(ensemble_action)
        next_state, _, terminated, truncated_env, info = env.step(ensemble_action)
        portfolio_values.append(info["portfolio_value"])
        daily_returns.append(info["return"])
        asset_weights.append(info.get("weights", np.zeros(env.action_space.shape[0])))
        total_raw_reward += info.get("raw_reward", 0.0)
        state = next_state
        step_count += 1
        pbar_eval.update(1)
        if step_count >= max_test_timesteps:
            truncated = True

    pbar_eval.close()
    logger.info(f"앙상블 평가 종료. 총 스텝: {step_count}")

    return {
        "episode_reward": total_raw_reward,
        "portfolio_values": portfolio_values,
        "returns": daily_returns,
        "weights": asset_weights,
        "actions": chosen_actions,
    }


# --- 벤치마크 및 향상된 백테스팅 함수 추가 ---
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
            logger.info(f"벤치마크 데이터 캐시 로드 완료")
            return benchmark_data
        except Exception as e:
            logger.warning(f"벤치마크 데이터 캐시 로드 실패: {e}")

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
        with open(cache_file, "wb") as f:
            pickle.dump(benchmark_data, f)

        logger.info(
            f"벤치마크 데이터 처리 완료 및 캐시 저장 (총 {len(benchmark_data)} 종목)"
        )
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
        metrics = calculate_performance_metrics(daily_returns.values)

        benchmark_performance[ticker] = {
            "values": normalized_values.values
            * INITIAL_CASH,  # 포트폴리오와 동일한 초기 금액으로 스케일링
            "returns": daily_returns.values,
            "metrics": metrics,
        }

    return benchmark_performance


def plot_performance_comparison(
    portfolio_values,
    benchmark_performance,
    plot_dir,
    dates=None,
    title="Portfolio vs Benchmark",
    filename=None,
):
    """
    포트폴리오와 벤치마크 성과를 비교하는 그래프를 생성합니다.

    Args:
        portfolio_values (list): 포트폴리오 가치 리스트
        benchmark_performance (dict): 벤치마크 성능 딕셔너리
        plot_dir (str): 그래프 저장 디렉토리
        dates (pd.DatetimeIndex): 날짜 인덱스
        title (str): 그래프 제목
        filename (str): 저장할 파일명
    """
    logger = logging.getLogger("PortfolioRL")
    if len(portfolio_values) == 0:
        logger.warning("그래프 생성 실패: 포트폴리오 데이터 없음")
        return

    if not benchmark_performance:
        logger.warning("그래프 생성 실패: 벤치마크 데이터 없음")
        return

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(14, 10))

    # 1. 가치 비교 그래프
    plt.subplot(2, 1, 1)

    # 날짜 축 설정
    x_axis = (
        dates
        if dates is not None and len(dates) == len(portfolio_values)
        else range(len(portfolio_values))
    )
    xlabel = (
        "Date"
        if dates is not None and len(dates) == len(portfolio_values)
        else "Trading Days"
    )

    # 포트폴리오 가치 플롯
    plt.plot(x_axis, portfolio_values, label="PPO Portfolio", linewidth=2, color="blue")

    # 벤치마크 가치 플롯
    colors = ["red", "green", "orange", "purple"]
    for i, (ticker, data) in enumerate(benchmark_performance.items()):
        # 길이 맞추기
        benchmark_values = data["values"]
        if len(benchmark_values) > len(portfolio_values):
            benchmark_values = benchmark_values[: len(portfolio_values)]
        elif len(benchmark_values) < len(portfolio_values):
            # 부족한 길이는 마지막 값으로 채우기
            padding = np.full(
                len(portfolio_values) - len(benchmark_values), benchmark_values[-1]
            )
            benchmark_values = np.concatenate([benchmark_values, padding])

        plt.plot(
            x_axis,
            benchmark_values,
            label=f"{ticker} Benchmark",
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
            color=colors[i % len(colors)],
        )

    plt.title(f"{title} - Value Comparison", fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # 2. 누적 수익률 비교 그래프
    plt.subplot(2, 1, 2)

    # 포트폴리오 누적 수익률 계산
    portfolio_returns = np.array(
        [
            (portfolio_values[i] / portfolio_values[i - 1]) - 1
            for i in range(1, len(portfolio_values))
        ]
    )
    portfolio_cumulative_returns = (
        np.cumprod(1 + portfolio_returns) - 1
    ) * 100  # 퍼센트로 변환

    # 포트폴리오 누적 수익률 플롯
    plt.plot(
        x_axis[1:],
        portfolio_cumulative_returns,
        label="PPO Portfolio",
        linewidth=2,
        color="blue",
    )

    # 벤치마크 누적 수익률 플롯
    for i, (ticker, data) in enumerate(benchmark_performance.items()):
        benchmark_returns = data["returns"]
        if len(benchmark_returns) > len(portfolio_returns):
            benchmark_returns = benchmark_returns[: len(portfolio_returns)]
        elif len(benchmark_returns) < len(portfolio_returns):
            # 부족한 길이는 0으로 채우기
            padding = np.zeros(len(portfolio_returns) - len(benchmark_returns))
            benchmark_returns = np.concatenate([benchmark_returns, padding])

        benchmark_cumulative_returns = (
            np.cumprod(1 + benchmark_returns) - 1
        ) * 100  # 퍼센트로 변환

        plt.plot(
            x_axis[1:],
            benchmark_cumulative_returns,
            label=f"{ticker} Benchmark",
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
            color=colors[i % len(colors)],
        )

    plt.title(f"{title} - Cumulative Return Comparison (%)", fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    # 파일 저장
    if filename is None:
        filename = (
            f'portfolio_vs_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
    save_path = os.path.join(plot_dir, filename)

    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"비교 그래프 저장 완료: {save_path}")
    except Exception as e:
        logger.error(f"그래프 저장 오류: {e}")
    finally:
        plt.close()


def print_performance_comparison(portfolio_metrics, benchmark_performance, logger):
    """
    포트폴리오와 벤치마크의 성능 지표를 비교하여 로깅합니다.

    Args:
        portfolio_metrics (dict): 포트폴리오 성능 지표
        benchmark_performance (dict): 벤치마크 성능 지표
        logger (logging.Logger): 로깅에 사용할 로거
    """
    logger.info("=" * 50)
    logger.info("성능 비교 결과 (포트폴리오 vs 벤치마크)")
    logger.info("=" * 50)

    # 포트폴리오 성능 출력
    logger.info("포트폴리오 성능:")
    logger.info(f"- 총 수익률: {portfolio_metrics['total_return']:.2f}%")
    logger.info(f"- 연간 수익률: {portfolio_metrics['annual_return']*100:.2f}%")
    logger.info(f"- 연간 변동성: {portfolio_metrics['annual_volatility']*100:.2f}%")
    logger.info(f"- 샤프 비율: {portfolio_metrics['sharpe_ratio']:.2f}")
    logger.info(f"- 최대 낙폭: {portfolio_metrics['max_drawdown']*100:.2f}%")
    logger.info(f"- 칼마 비율: {portfolio_metrics['calmar_ratio']:.2f}")

    # 각 벤치마크 성능 출력
    for ticker, data in benchmark_performance.items():
        metrics = data["metrics"]
        logger.info(f"\n{ticker} 벤치마크 성능:")
        logger.info(f"- 총 수익률: {metrics['total_return']:.2f}%")
        logger.info(f"- 연간 수익률: {metrics['annual_return']*100:.2f}%")
        logger.info(f"- 연간 변동성: {metrics['annual_volatility']*100:.2f}%")
        logger.info(f"- 샤프 비율: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"- 최대 낙폭: {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"- 칼마 비율: {metrics['calmar_ratio']:.2f}")

        # 상대 성능 계산
        relative_return = portfolio_metrics["total_return"] - metrics["total_return"]
        rel_annual_return = (
            portfolio_metrics["annual_return"] - metrics["annual_return"]
        )
        rel_sharpe = portfolio_metrics["sharpe_ratio"] - metrics["sharpe_ratio"]

        logger.info(f"\n포트폴리오 vs {ticker} 상대 성능:")
        logger.info(f"- 초과 총 수익률: {relative_return:.2f}%")
        logger.info(f"- 초과 연간 수익률: {rel_annual_return*100:.2f}%")
        logger.info(f"- 샤프 비율 차이: {rel_sharpe:.2f}")

    logger.info("=" * 50)


# --- 메인 실행 함수 ---
def main():
    """메인 실행 함수: 데이터 로드, 학습, 평가, 결과 분석 및 시각화 수행"""
    # 다중 시드 학습 설정
    n_seeds = ENSEMBLE_SIZE  # 학습할 시드 수
    seeds = [int(time.time()) + i * 1000 for i in range(n_seeds)]

    # 결과 저장 디렉토리 생성
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_BASE_PATH, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # 로거 설정
    logger = setup_logger(run_dir)
    logger.info(f"결과 저장 폴더: {run_dir}")
    logger.info(f"다중 시드 학습 설정: {n_seeds}개 모델, 시드: {seeds}")

    # --- 시스템 환경 확인 ---
    logger.info("\n" + "=" * 15 + " 시스템 환경 확인 " + "=" * 15)
    logger.info(f" 사용 디바이스: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f" CUDA 버전: {torch.version.cuda}")
        logger.info(f" GPU: {torch.cuda.get_device_name(0)}")
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            logger.info(f" GPU 사용률: {util.gpu}% / 메모리 사용률: {util.memory}%")
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"pynvml 사용 불가 ({e}). GPU 상세 사용률 확인 생략.")
    else:
        logger.warning(" CUDA 사용 불가능. CPU로 실행됩니다.")
    logger.info("=" * 48)

    # --- 데이터 준비 ---
    logger.info("\n" + "=" * 18 + " 데이터 준비 " + "=" * 18)
    data_array, common_dates = fetch_and_preprocess_data(
        TRAIN_START_DATE, TEST_END_DATE, STOCK_TICKERS
    )
    if data_array is None:
        logger.error("데이터 준비 실패. 종료.")
        return

    # 데이터 분할
    split_date = pd.Timestamp(TEST_START_DATE).tz_localize(None)
    if not isinstance(common_dates, pd.DatetimeIndex):
        common_dates = pd.to_datetime(common_dates)
    common_dates_naive = common_dates.tz_localize(None)
    try:
        split_idx = np.searchsorted(common_dates_naive, split_date)
    except Exception as e:
        logger.error(f"데이터 분할 인덱스 검색 오류: {e}. 대체 검색 사용.")
        split_idx_arr = np.where(common_dates_naive >= split_date)[0]
        if len(split_idx_arr) == 0:
            logger.error(f"분할 날짜({TEST_START_DATE}) 이후 데이터 없음. 종료.")
            return
        split_idx = split_idx_arr[0]
    if not (0 < split_idx < len(common_dates)):
        logger.error(f"데이터 분할 오류: 분할 인덱스({split_idx}) 유효하지 않음. 종료.")
        return

    train_data = data_array[:split_idx]
    test_data = data_array[split_idx:]
    test_dates = common_dates[split_idx:]
    logger.info(
        f" 훈련 데이터: {train_data.shape} ({common_dates[0].date()} ~ {common_dates[split_idx-1].date()})"
    )
    logger.info(
        f" 테스트 데이터: {test_data.shape} ({test_dates[0].date()} ~ {test_dates[-1].date()})"
    )

    # 벤치마크 데이터 가져오기
    if USE_BENCHMARK:
        logger.info(f" 벤치마크 설정: {BENCHMARK_TICKERS}")
        benchmark_data = fetch_benchmark_data(
            BENCHMARK_TICKERS, TEST_START_DATE, TEST_END_DATE
        )
    else:
        logger.info(" 벤치마크 비교 사용하지 않음")
        benchmark_data = {}

    logger.info("=" * 48)

    # --- 피처 스케일링 (z-score) 적용 ---
    # 기술 지표(Volume 제외) 인덱스만 스케일링하고 가격 관련 지표는 원래 스케일 유지
    n_features_data = data_array.shape[2]
    tech_start_idx = 5  # FEATURE_NAMES 기준 MACD부터
    if n_features_data > tech_start_idx:
        idx_to_scale = np.arange(tech_start_idx, n_features_data)

        scaler = StandardScaler()
        scaler.fit(train_data[:, :, idx_to_scale].reshape(-1, len(idx_to_scale)))

        # 변환 적용
        train_scaled_slice = scaler.transform(
            train_data[:, :, idx_to_scale].reshape(-1, len(idx_to_scale))
        )
        test_scaled_slice = scaler.transform(
            test_data[:, :, idx_to_scale].reshape(-1, len(idx_to_scale))
        )

        train_data[:, :, idx_to_scale] = train_scaled_slice.reshape(
            train_data.shape[0], train_data.shape[1], len(idx_to_scale)
        )
        test_data[:, :, idx_to_scale] = test_scaled_slice.reshape(
            test_data.shape[0], test_data.shape[1], len(idx_to_scale)
        )

    # --- 다중 시드 기반 학습 및 앙상블 에이전트 생성 ---
    logger.info("\n" + "=" * 14 + " 다중 시드 학습 시작 " + "=" * 14)

    train_env = StockPortfolioEnv(train_data, normalize_states=True)
    n_assets, n_features = train_env.n_assets, train_env.n_features

    # 검증 환경 생성 (Early Stopping용)
    validation_env = StockPortfolioEnv(
        train_data[-500:], normalize_states=False
    )  # 학습 데이터 마지막 500일 사용

    # EMA 가중치 사용 설정
    use_ema = True
    ema_decay = 0.99

    # 에피소드 및 스텝 설정
    max_episodes_train = NUM_EPISODES
    max_timesteps_train = train_env.max_episode_length

    # 앙상블 구성을 위한 에이전트 리스트
    ensemble_agents = []

    # 각 시드별 모델 학습
    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n--- 시드 {seed_idx+1}/{n_seeds} (seed={seed}) 학습 시작 ---")

        # 시드 설정
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 에이전트 생성 (시드별 모델 저장 경로 분리)
        seed_model_path = os.path.join(run_dir, f"model_seed_{seed}")
        os.makedirs(seed_model_path, exist_ok=True)

        ppo_agent = PPO(
            n_assets,
            n_features,
            logger=logger,
            lr=DEFAULT_LR,
            use_ema=use_ema,
            ema_decay=ema_decay,
            model_path=seed_model_path,
            use_lr_scheduler=True,
            use_early_stopping=True,
        )

        logger.info(
            f"PPO 에이전트 생성 완료 (lr={DEFAULT_LR}, use_ema={use_ema}, seed={seed})"
        )

        # 환경 리셋 (시드 설정)
        train_env.reset(seed=seed)

        # 모델 학습
        training_rewards = train_ppo_agent(
            train_env,
            ppo_agent,
            max_episodes_train,
            max_timesteps_train,
            PPO_UPDATE_TIMESTEP,
            logger,
            validation_env=validation_env,
        )

        # 앙상블을 위해 에이전트 저장
        ensemble_agents.append(ppo_agent)

        logger.info(f"시드 {seed_idx+1}/{n_seeds} 학습 완료!")

    logger.info("\n" + "=" * 16 + " 앙상블 에이전트 평가 " + "=" * 16)

    # 테스트 환경 생성
    test_env = StockPortfolioEnv(test_data, normalize_states=False)
    max_test_timesteps = len(test_data) - 1

    # 1. 개별 에이전트 평가
    individual_results = []
    for i, agent in enumerate(ensemble_agents):
        logger.info(f"\n--- 개별 에이전트 {i+1}/{n_seeds} 평가 ---")
        result = evaluate_ppo_agent(
            test_env, agent, max_test_timesteps, load_best_model=True
        )
        if result:
            individual_results.append(result)
            metrics = calculate_performance_metrics(result["returns"])
            logger.info(
                f"에이전트 {i+1} - 최종 가치: {result['portfolio_values'][-1]:.2f}, 총 수익률: {metrics['total_return']:.2f}%"
            )

    # 2. 앙상블 에이전트 평가
    logger.info("\n--- 앙상블 에이전트 평가 ---")
    ensemble_result = evaluate_ensemble(
        test_env, ensemble_agents, max_test_timesteps, load_best_model=True
    )

    if ensemble_result is None:
        logger.error("앙상블 평가 실패")
        return

    # --- 성능 분석 및 시각화 ---
    logger.info("\n" + "=" * 13 + " 성능 분석 및 시각화 " + "=" * 13)

    # 앙상블 성능 지표 계산
    ens_metrics = calculate_performance_metrics(ensemble_result["returns"])
    logger.info("--- 앙상블 포트폴리오 성능 지표 ---")
    logger.info(f" 최종 가치: {ensemble_result['portfolio_values'][-1]:.2f}")
    logger.info(f" 총 수익률: {ens_metrics['total_return']:.2f}%")
    logger.info(f" 일간 표준편차: {ens_metrics['daily_std']:.4f}")
    logger.info(f" 연간 수익률: {ens_metrics['annual_return']:.2%}")
    logger.info(f" 연간 변동성: {ens_metrics['annual_volatility']:.2%}")
    logger.info(f" 샤프 비율: {ens_metrics['sharpe_ratio']:.2f}")
    logger.info(f" 최대 낙폭: {ens_metrics['max_drawdown']:.2%}")
    logger.info(f" 칼마 비율: {ens_metrics['calmar_ratio']:.2f}")

    # --- 벤치마크와 비교 ---
    if USE_BENCHMARK and benchmark_data:
        logger.info("\n--- 벤치마크 성능 분석 ---")
        # 벤치마크 성능 계산
        benchmark_performance = calculate_benchmark_performance(
            benchmark_data, test_dates
        )

        # 성능 비교 출력
        print_performance_comparison(ens_metrics, benchmark_performance, logger)

        # 벤치마크 비교 그래프 생성
        plot_performance_comparison(
            ensemble_result["portfolio_values"],
            benchmark_performance,
            plot_dir=run_dir,
            dates=test_dates,
            title="PPO Ensemble vs Market Benchmarks",
            filename="portfolio_vs_benchmark.png",
        )

    # 앙상블 vs 개별 모델 성능 비교
    if individual_results:
        ind_returns = [
            np.mean(
                [
                    calculate_performance_metrics(r["returns"])["total_return"]
                    for r in individual_results
                ]
            )
        ]
        ind_sharpes = [
            np.mean(
                [
                    calculate_performance_metrics(r["returns"])["sharpe_ratio"]
                    for r in individual_results
                ]
            )
        ]
        ind_volatility = [
            np.mean(
                [
                    calculate_performance_metrics(r["returns"])["annual_volatility"]
                    for r in individual_results
                ]
            )
        ]

        logger.info("\n--- 앙상블 vs 개별 모델 평균 성능 ---")
        logger.info(f" 개별 모델 평균 수익률: {ind_returns[0]:.2f}%")
        logger.info(f" 앙상블 수익률: {ens_metrics['total_return']:.2f}%")
        logger.info(f" 개별 모델 평균 샤프 비율: {ind_sharpes[0]:.2f}")
        logger.info(f" 앙상블 샤프 비율: {ens_metrics['sharpe_ratio']:.2f}")
        logger.info(f" 개별 모델 평균 변동성: {ind_volatility[0]:.2%}")
        logger.info(f" 앙상블 변동성: {ens_metrics['annual_volatility']:.2%}")

        # 변동성 감소율 계산
        vol_reduction = 1 - (ens_metrics["annual_volatility"] / ind_volatility[0])
        logger.info(f" 앙상블의 변동성 감소율: {vol_reduction:.2%}")

    # 성능 그래프 생성
    plot_performance(
        ensemble_result["portfolio_values"],
        dates=test_dates,
        title="Ensemble Portfolio Performance (Evaluation)",
        filename=f"ensemble_performance.png",
        plot_dir=run_dir,
    )

    # 샘플 개별 에이전트 그래프 (첫 번째만)
    if individual_results:
        plot_performance(
            individual_results[0]["portfolio_values"],
            dates=test_dates,
            title="Individual Agent Performance (First Agent)",
            filename=f"individual_performance.png",
            plot_dir=run_dir,
        )

    logger.info("=" * 48)
    logger.info("\n===== 프로그램 종료 =====")

    # --- XAI 분석: 특성 중요도 및 통합 그래디언트 분석 ---
    logger.info("\n" + "=" * 15 + " XAI 분석 및 시각화 " + "=" * 15)

    # 테스트 데이터 샘플 추출 (통합 그래디언트 및 특성 중요도 분석용)
    test_sample_indices = np.linspace(
        0, len(test_data) - 1, XAI_SAMPLE_COUNT, dtype=int
    )
    test_samples = test_data[test_sample_indices]

    # 1. DRL 모델 특성 중요도 분석
    logger.info("DRL 모델 특성 중요도 분석 중...")
    drl_weights = []

    for agent_idx, agent in enumerate(ensemble_agents):
        agent_weights = compute_feature_weights_drl(agent, test_samples)
        drl_weights.append(agent_weights)
        logger.info(f"에이전트 {agent_idx+1} 특성 중요도 분석 완료")

    # 모든 에이전트의 평균 특성 중요도
    drl_weights_mean = np.mean(drl_weights, axis=0)

    # 2. 선형 참조 모델 분석 (사후 분석)
    logger.info("선형 참조 모델 분석 중...")

    # 테스트 데이터에서 보상 계산 (일간 수익률)
    ref_features = test_data[:-1]  # 마지막 날은 포함하지 않음 (returns가 n-1개이므로)
    ref_returns = np.array(ensemble_result["returns"])

    # 데이터 형태 확인
    logger.info(
        f"참조 모델 입력 데이터 형태 - features: {ref_features.shape}, returns: {len(ref_returns)}"
    )

    # 데이터 차원 확인 및 조정
    if ref_features.shape[1] != len(STOCK_TICKERS) or ref_features.shape[2] != len(
        FEATURE_NAMES
    ):
        logger.warning(
            f"참조 모델 데이터 차원 불일치: shape={ref_features.shape}, 티커={len(STOCK_TICKERS)}, 특성={len(FEATURE_NAMES)}"
        )

    # 샘플 수 확인 후 조정
    if ref_features.shape[0] != len(ref_returns):
        logger.warning(
            f"특성({ref_features.shape[0]})과 반환({len(ref_returns)}) 길이 불일치. 특성 데이터 조정."
        )
        # 더 짧은 길이에 맞춤
        min_len = min(ref_features.shape[0], len(ref_returns))
        ref_features = ref_features[:min_len]
        ref_returns = ref_returns[:min_len]

    # 선형 회귀 모델로 중요도 계산
    try:
        ref_weights = linear_model_hindsight(ref_features, ref_returns)
        if ref_weights is None:
            logger.warning("참조 모델 가중치 계산 실패, 더미 데이터 사용")
            # 더미 데이터 생성
            ref_weights = np.zeros(len(FEATURE_NAMES))
        else:
            # 가중치 로깅
            logger.info("참조 모델 특성 중요도:")
            for i, feature_name in enumerate(FEATURE_NAMES):
                if i < len(ref_weights):
                    logger.info(f"  {feature_name}: {ref_weights[i]:.4f}")
    except Exception as e:
        logger.error(f"참조 모델 분석 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
        ref_weights = np.zeros(len(FEATURE_NAMES))

    # 3. 통합 그래디언트 분석
    logger.info("통합 그래디언트 분석 중...")
    ig_values = []

    # 첫 번째 에이전트만 사용 (계산 비용 절감)
    primary_agent = ensemble_agents[0]
    logger.info(
        f"통합 그래디언트 분석에 앙상블 중 첫 번째 에이전트(seed: {primary_agent.seed if hasattr(primary_agent, 'seed') else 'unknown'}) 사용"
    )

    # EMA 모델 사용 여부 확인 및 로깅
    is_using_ema = primary_agent.use_ema if hasattr(primary_agent, "use_ema") else False
    logger.info(
        f"통합 그래디언트 분석에 {'EMA 모델' if is_using_ema else '기본 모델'} 사용"
    )

    # 각 샘플에 대한 통합 그래디언트 계산
    for i, sample in enumerate(test_samples):
        normalized_sample = sample.copy()
        if (
            primary_agent.obs_rms is not None
            and primary_agent.obs_rms.count > RMS_EPSILON
        ):
            normalized_sample = np.clip(
                (sample - primary_agent.obs_rms.mean)
                / np.sqrt(primary_agent.obs_rms.var + RMS_EPSILON),
                -CLIP_OBS,
                CLIP_OBS,
            )

        # 기본 샘플을 0으로 설정
        baseline = np.zeros_like(normalized_sample)

        # 통합 그래디언트 계산
        ig_result = integrated_gradients(
            primary_agent.policy_ema if primary_agent.use_ema else primary_agent.policy,
            normalized_sample,
        )
        ig_values.append(ig_result)
        logger.info(f"샘플 {i+1}/{len(test_samples)} 통합 그래디언트 분석 완료")

    # 모든 샘플의 평균 통합 그래디언트
    ig_values_mean = np.mean(ig_values, axis=0)

    # 차원 확인 및 처리 (필요한 경우 평탄화)
    logger.info(f"통합 그래디언트 원본 형태: {ig_values_mean.shape}")

    # 차원 변환 처리
    if ig_values_mean.shape == (len(STOCK_TICKERS), len(FEATURE_NAMES)):
        # 올바른 형태: (n_assets, n_features) -> 자산별 평균 계산
        logger.info(
            "통합 그래디언트: 각 자산별 특성 기여도를 평균하여 전체 특성 중요도 계산"
        )
        # 자산 차원에 대해 평균 계산하여 특성당 하나의 값으로 만듦
        ig_feature_importance = np.mean(ig_values_mean, axis=0)

        # 통합 그래디언트 특성 중요도 로깅
        for i, feature_name in enumerate(FEATURE_NAMES):
            logger.info(f"  {feature_name}: {ig_feature_importance[i]:.4f}")

        ig_values_mean = ig_feature_importance
    else:
        logger.warning(f"예상치 못한 통합 그래디언트 형태: {ig_values_mean.shape}")
        # 강제 재구성 시도
        if ig_values_mean.ndim > 1:
            n_features = len(FEATURE_NAMES)
            if ig_values_mean.size >= n_features:
                # 일부 데이터 손실을 감수하고 첫 n_features 개 요소 사용
                ig_values_mean = ig_values_mean.flatten()[:n_features]
                logger.info(f"통합 그래디언트 강제 변환: {ig_values_mean.shape}")
            else:
                # 데이터가 부족하면 0으로 패딩
                temp = np.zeros(n_features)
                temp[: ig_values_mean.size] = ig_values_mean.flatten()
                ig_values_mean = temp
                logger.info(f"통합 그래디언트 패딩 추가: {ig_values_mean.shape}")

    # --- 결과 시각화 및 저장 ---
    # 1. 특성 중요도 비교 (DRL vs 선형 참조 모델)
    if ref_weights is not None:
        # ref_weights가 올바른 형태인지 확인
        if isinstance(ref_weights, np.ndarray):
            # ref_weights 형태 조정
            if ref_weights.ndim > 1:
                ref_weights = np.mean(
                    ref_weights, axis=0
                )  # 첫 번째 차원에 대해 평균 계산
                logger.info(f"참조 모델 가중치 형태 변환: {ref_weights.shape}")

            # drl_weights_mean 형태 조정 (필요한 경우)
            drl_weights_for_plot = drl_weights_mean
            if drl_weights_mean.ndim > 1:
                drl_weights_for_plot = np.mean(drl_weights_mean, axis=0)
                logger.info(
                    f"DRL 가중치 형태 변환 (비교용): {drl_weights_for_plot.shape}"
                )

            plot_feature_importance(
                drl_weights_for_plot,
                ref_weights,
                plot_dir=run_dir,
                feature_names=FEATURE_NAMES,
                filename="feature_importance_comparison.png",
            )
            logger.info("특성 중요도 비교 시각화 완료")
        else:
            logger.warning("참조 모델 가중치가 None이거나 예상치 않은 형태입니다.")

    # 1-1. DRL 에이전트 특성 중요도 단독 시각화 (추가)
    if isinstance(drl_weights_mean, np.ndarray):
        # DRL 에이전트 특성 중요도 별도 시각화
        plt.figure(figsize=(12, 6))

        # 차원 변환 필요 시 처리 - compute_feature_weights_drl에서 이미 처리되었으므로 간소화
        drl_feature_weights = drl_weights_mean
        if drl_weights_mean.ndim > 1:
            # 로그 추가
            logger.info(f"DRL 특성 가중치 원본 형태: {drl_weights_mean.shape}")
            # 샘플 차원에 대해 평균 계산
            drl_feature_weights = np.mean(drl_weights_mean, axis=0)
            logger.info(f"샘플 차원에 대한 평균 계산: {drl_feature_weights.shape}")

        # 길이 확인 및 자르기/패딩
        if len(drl_feature_weights) > len(FEATURE_NAMES):
            logger.warning(
                f"특성 가중치({len(drl_feature_weights)})가 이름({len(FEATURE_NAMES)})보다 많음. 자르기 실행"
            )
            drl_feature_weights = drl_feature_weights[: len(FEATURE_NAMES)]
        elif len(drl_feature_weights) < len(FEATURE_NAMES):
            logger.warning(
                f"특성 가중치({len(drl_feature_weights)})가 이름({len(FEATURE_NAMES)})보다 적음. 패딩 실행"
            )
            temp = np.zeros(len(FEATURE_NAMES))
            temp[: len(drl_feature_weights)] = drl_feature_weights
            drl_feature_weights = temp

        # NaN 값 확인 및 처리
        if np.isnan(drl_feature_weights).any():
            logger.warning("특성 가중치에 NaN 값 발견. 0으로 대체")
            drl_feature_weights = np.nan_to_num(drl_feature_weights, nan=0.0)

        # plot_feature_importance 함수와 동일한 정규화 방식 적용
        weights_abs = np.abs(drl_feature_weights)
        median_abs = np.median(weights_abs)
        mad = np.median(np.abs(weights_abs - median_abs))  # Median Absolute Deviation
        threshold = median_abs + 10 * mad  # 보수적인 임계값

        # 이상치 제한
        clipped_weights = np.clip(drl_feature_weights, -threshold, threshold)

        # 크기가 있는 데이터에 MinMax 스케일링
        if np.max(np.abs(clipped_weights)) > 1e-10:
            normalized_weights = clipped_weights / np.max(np.abs(clipped_weights))
        else:
            normalized_weights = clipped_weights

        # 데이터 최종 검증
        normalized_weights = np.array(normalized_weights, dtype=np.float64)

        # 그래프 생성
        try:
            x = np.arange(len(FEATURE_NAMES))
            bars = plt.bar(x, normalized_weights, color="skyblue")
            plt.ylabel("Normalized Importance Score")
            plt.title(f"DRL Agent Feature Importance\n(Clipped at {threshold:.2e})")
            plt.xticks(x, FEATURE_NAMES, rotation=45, ha="right")
            plt.grid(axis="y", linestyle="--")
            plt.ylim(-1.1, 1.1)

            # 값 표시
            for bar in bars:
                yval = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    yval,
                    f"{yval:.2f}",
                    va="bottom" if yval >= 0 else "top",
                    ha="center",
                    fontsize=8,
                )

            plt.tight_layout()

            # 저장
            save_path = os.path.join(run_dir, "feature_importance.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(
                "DRL 에이전트 특성 중요도 단독 시각화 완료: feature_importance.png"
            )
        except Exception as e:
            logger.error(f"DRL 에이전트 특성 중요도 시각화 오류: {e}")
            logger.error(traceback.format_exc())
        finally:
            plt.close()

    # 2. 통합 그래디언트 시각화
    if isinstance(ig_values_mean, np.ndarray) and len(ig_values_mean) == len(
        FEATURE_NAMES
    ):
        # 올바른 형태를 가진 경우에만 시각화
        model_info = f"{'EMA' if is_using_ema else 'Base'} Model, Agent 1"
        plot_integrated_gradients(
            ig_values_mean,
            plot_dir=run_dir,
            feature_names=FEATURE_NAMES,
            title=f"Mean Integrated Gradients ({model_info})",
            filename="integrated_gradients.png",
        )
        logger.info("통합 그래디언트 시각화 완료")
    else:
        logger.warning(
            f"통합 그래디언트 형태 불일치로 시각화 불가: {ig_values_mean.shape if isinstance(ig_values_mean, np.ndarray) else 'None'}"
        )

    logger.info("XAI 분석 및 시각화 완료")

    logger.info("=" * 48)
    logger.info("\n===== 프로그램 종료 =====")


if __name__ == "__main__":
    main()
