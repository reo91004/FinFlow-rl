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
import yfinance as yf
from datetime import datetime
import os
import pickle
import logging
import sys
import gc
import time
from tqdm import tqdm
import traceback # 오류 로깅을 위해 추가

# --- 상수 정의 ---
# GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 분석 대상 주식 티커 목록
STOCK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "JPM", "JNJ", "PG", "V"
]

# 학습/테스트 데이터 기간 설정
TRAIN_START_DATE = '2008-01-02'
TRAIN_END_DATE = '2020-12-31'
TEST_START_DATE = '2021-01-01'
TEST_END_DATE = '2024-12-31'

# 포트폴리오 초기 설정
INITIAL_CASH = 1e6
COMMISSION_RATE = 0.005

# PPO 하이퍼파라미터 (기본값)
DEFAULT_HIDDEN_DIM = 128
DEFAULT_LR = 1e-4  # 학습률 조정됨
DEFAULT_GAMMA = 0.99
DEFAULT_K_EPOCHS = 10
DEFAULT_EPS_CLIP = 0.2
PPO_UPDATE_TIMESTEP = 2000 # PPO 업데이트 주기 (스텝 수)

# 환경 설정
MAX_EPISODE_LENGTH = 200 # 환경의 기본 최대 에피소드 길이

# 상태/보상 정규화 설정
NORMALIZE_STATES = True
CLIP_OBS = 10.0
CLIP_REWARD = 10.0
RMS_EPSILON = 1e-8

# GAE 설정
LAMBDA_GAE = 0.95

# 모델 및 로그 저장 경로
MODEL_SAVE_PATH = 'models'
LOG_SAVE_PATH = 'logs'
PLOT_SAVE_PATH = 'plots'
DATA_SAVE_PATH = 'data'

# 설명 가능한 AI (XAI) 관련 설정
INTEGRATED_GRADIENTS_STEPS = 50
XAI_SAMPLE_COUNT = 5 # 통합 그래디언트 분석 샘플 수

# 피처 이름 정의 (데이터 처리 순서와 일치)
FEATURE_NAMES = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'RSI', 'MA14', 'MA21', 'MA100']

# --- 로깅 설정 ---
def setup_logger(log_dir=LOG_SAVE_PATH):
    """
    로깅 시스템을 설정합니다.
    파일 핸들러는 INFO 레벨 이상, 콘솔 핸들러는 WARNING 레벨 이상만 출력하도록 변경.
    특정 필터는 제거하고 레벨로 제어합니다.
    """
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_log_{current_time}.txt')

    logger = logging.getLogger('PortfolioRL')
    # 기본 로거 레벨을 DEBUG로 설정하여 모든 메시지 처리 가능하도록 함
    logger.setLevel(logging.DEBUG)

    # 기존 핸들러 제거
    if logger.hasHandlers():
        logger.handlers.clear()

    # 파일 핸들러 설정 (INFO 레벨 이상, 모든 정보 기록)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO) # 파일에는 INFO 레벨 이상 기록
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러 설정 (WARNING 레벨 이상, 간략 정보)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s') # 레벨 이름 포함
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING) # 콘솔에는 WARNING 레벨 이상만 출력
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
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        """ 배치 데이터로 평균과 분산을 업데이트합니다. """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """ 계산된 평균과 분산으로 내부 상태를 업데이트합니다. """
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
        """ 메모리에 저장된 모든 경험을 삭제합니다. """
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]

    def add_experience(self, state, action, logprob, reward, is_terminal, value):
        """ 새로운 경험을 메모리에 추가합니다. """
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
    - 보상(Reward): 포트폴리오 가치의 로그 변화율
    - 상태 정규화(State Normalization): RunningMeanStd를 이용한 관측값 및 보상 정규화 기능 포함
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, data: np.ndarray, initial_cash=INITIAL_CASH, commission_rate=COMMISSION_RATE,
                 max_episode_length=MAX_EPISODE_LENGTH, normalize_states=NORMALIZE_STATES, gamma=DEFAULT_GAMMA):
        super(StockPortfolioEnv, self).__init__()
        self.data = data  # (n_steps, n_assets, n_features)
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.max_episode_length = max_episode_length
        self.normalize_states = normalize_states
        self.gamma = gamma # 보상 정규화 시 사용
        
        self.n_steps, self.n_assets, self.n_features = data.shape
        
        # 상태 공간 정의
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_assets, self.n_features), dtype=np.float32
        )
        
        # 행동 공간 정의 (Dirichlet 분포 사용하므로, 각 자산 비중)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        
        # 상태/보상 정규화 객체 초기화
        if self.normalize_states:
            self.obs_rms = RunningMeanStd(shape=(self.n_assets, self.n_features))
            self.ret_rms = RunningMeanStd(shape=())
            self.returns_norm = np.zeros(1) # 정규화된 누적 보상 추적
        else:
            self.obs_rms = None
            self.ret_rms = None

        # 내부 상태 변수 초기화 (reset에서 수행)
        self.current_step = 0
        self.cash = 0.0
        self.holdings = np.zeros(self.n_assets, dtype=np.float32) # 보유 주식 수
        self.portfolio_value = 0.0 # 현재 포트폴리오 가치
        self.weights = np.ones(self.n_assets) / self.n_assets # 현재 자산 비중
    
    def _normalize_obs(self, obs):
        """ 관측값을 정규화합니다. """
        if not self.normalize_states or self.obs_rms is None: return obs
        # RunningMeanStd 업데이트 (차원 맞추기)
        self.obs_rms.update(obs.reshape(1, self.n_assets, self.n_features))
        # 정규화 및 클리핑
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + RMS_EPSILON),
                     -CLIP_OBS, CLIP_OBS)
    
    def _normalize_reward(self, reward):
        """ 보상을 정규화합니다. """
        if not self.normalize_states or self.ret_rms is None: return reward
        # 누적 할인 보상 업데이트
        self.returns_norm = self.gamma * self.returns_norm + reward
        # RunningMeanStd 업데이트
        self.ret_rms.update(self.returns_norm)
        # 정규화 및 클리핑
        return np.clip(reward / np.sqrt(self.ret_rms.var + RMS_EPSILON),
                     -CLIP_REWARD, CLIP_REWARD)

    def reset(self, *, seed=None, options=None, start_index=None):
        """ 환경을 초기 상태로 리셋합니다. """
        super().reset(seed=seed)
        logger = logging.getLogger('PortfolioRL')
        
        # 에피소드 시작 인덱스 설정 (데이터 길이 내 무작위 또는 0)
        if start_index is None:
            max_start_index = max(0, self.n_steps - self.max_episode_length)
            if max_start_index == 0:
                 start_index = 0
            else:
                 start_index = np.random.randint(0, max_start_index + 1) # 0부터 시작 가능하도록 +1
        elif start_index >= self.n_steps:
             logger.warning(f"제공된 시작 인덱스({start_index})가 데이터 범위({self.n_steps})를 벗어남. 0으로 설정.")
             start_index = 0

        self.current_step = start_index
        
        # 내부 상태 초기화
        self.cash = self.initial_cash
        self.holdings.fill(0)
        self.portfolio_value = self.cash
        self.weights = np.ones(self.n_assets) / self.n_assets
        
        # 보상 정규화 관련 변수 초기화
        if self.normalize_states:
            self.returns_norm = np.zeros(1)

        # 초기 관측값 반환
        observation = self._get_observation()
        normalized_observation = self._normalize_obs(observation)
        info = self._get_info() # 초기 정보 생성
        
        return normalized_observation.astype(np.float32), info
    
    def _get_observation(self):
        """ 현재 스텝의 원본 관측 데이터를 반환합니다. """
        # 데이터 인덱스 범위 확인 (방어 코드)
        step = min(self.current_step, self.n_steps - 1) # 범위를 벗어나지 않도록 조정
        return self.data[step]

    def _get_info(self):
        """ 현재 환경 상태 정보를 담은 딕셔너리를 반환합니다. """
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "weights": self.weights.copy(),
            "return": 0.0, # 초기 상태에서는 수익률 0
            "raw_reward": 0.0 # 초기 상태에서는 보상 0
        }

    def step(self, action):
        """
        환경을 한 스텝 진행시킵니다.

        Args:
            action (np.ndarray): 에이전트가 선택한 행동 (자산별 목표 비중).

        Returns:
            tuple: (next_observation, reward, terminated, truncated, info)
                   - next_observation (np.ndarray): 정규화된 다음 상태 관측값.
                   - reward (float): 정규화된 보상.
                   - terminated (bool): 에피소드 종료 여부 (파산 또는 데이터 끝 도달).
                   - truncated (bool): 에피소드 중단 여부 (최대 길이 도달).
                   - info (dict): 추가 정보 (포트폴리오 가치, 현금, 수익률 등).
        """
        logger = logging.getLogger('PortfolioRL')
        # 행동 정규화 (비중 합 1)
        action = np.clip(action, 0, 1)
        action_sum = action.sum()
        if action_sum > 1e-6:
            action = action / action_sum
        else: # 비중 합이 0에 가까우면 균등 분배
            action = np.ones(self.n_assets) / self.n_assets
            
        # 현재 가격 정보 (원본 데이터 사용)
        current_obs = self._get_observation()
        current_prices = np.maximum(current_obs[:, 3], 1e-6) # 종가, 0 방지

        # 이전 포트폴리오 가치
        prev_portfolio_value = self.cash + np.dot(self.holdings, current_prices)

        # 파산 조건 확인
        if prev_portfolio_value <= 1e-6:
            terminated = True
            truncated = False
            raw_reward = -10.0 # 파산 시 큰 음수 보상
            info = {
                "portfolio_value": 0.0, "cash": 0.0, "holdings": self.holdings.copy(),
                "weights": np.zeros_like(self.weights), "return": -1.0, "raw_reward": raw_reward
            }
            # 마지막 관측값은 현재 관측값 사용 (정규화)
            last_obs_norm = self._normalize_obs(current_obs)
            reward_norm = self._normalize_reward(raw_reward)
            return last_obs_norm.astype(np.float32), float(reward_norm), terminated, truncated, info

        # 목표 자산 가치 계산
        target_value_allocation = action * prev_portfolio_value

        # 실제 거래 실행 (매수/매도)
        self._execute_trades(target_value_allocation, current_prices)
                
        # 다음 스텝으로 이동
        self.current_step += 1
        terminated = self.current_step >= self.n_steps # 종료 조건: 마지막 스텝 이후
        truncated = False # Truncated는 학습 루프에서 제어
        
        # 다음 스텝 가격 및 새 포트폴리오 가치 계산
        next_obs_raw = self._get_observation() # 다음 스텝 관측값 가져오기
        next_prices = np.maximum(next_obs_raw[:, 3], 1e-6) # 다음 날 종가, 0 방지
        if terminated:
            next_obs_raw = current_obs # 마지막 스텝이면 현재 관측값 사용
        else:
            next_obs_raw = self._get_observation() # _get_observation 사용

        next_prices = np.maximum(next_obs_raw[:, 3], 1e-6) # 다음 날 종가, 0 방지
        self.portfolio_value = self.cash + np.dot(self.holdings, next_prices)

        # 가중치 업데이트 (0으로 나누기 방지)
        if self.portfolio_value > 1e-8:
            self.weights = (self.holdings * next_prices) / self.portfolio_value
        else:
            self.weights.fill(0)

        # 원본 보상 계산 (로그 수익률)
        prev_value_safe = max(prev_portfolio_value, 1e-8) # 이전 가치가 0에 가까울 때 대비
        current_value_safe = max(self.portfolio_value, 0.0) # 현재 가치는 0이 될 수 있음
        raw_reward = np.log(current_value_safe / prev_value_safe + 1e-8) # log(0) 방지

        if np.isnan(raw_reward) or np.isinf(raw_reward):
            # logger.warning(f"보상 계산 중 NaN/Inf 발생. 이전 가치: {prev_portfolio_value}, 현재 가치: {self.portfolio_value}. 보상 -1.0으로 설정.")
            raw_reward = -1.0 # NaN/Inf 발생 시 페널티

        # 다음 상태 및 보상 정규화
        next_obs_norm = self._normalize_obs(next_obs_raw)
        reward_norm = self._normalize_reward(raw_reward)

        # 정보 업데이트
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "weights": self.weights.copy(),
            "return": self.portfolio_value / prev_value_safe - 1 if prev_value_safe > 1e-8 else 0.0,
            "raw_reward": raw_reward
        }

        return next_obs_norm.astype(np.float32), float(reward_norm), terminated, truncated, info

    def _execute_trades(self, target_value_allocation, current_prices):
        """ 목표 가치 배분에 따라 실제 주식 거래를 실행하고 수수료를 계산합니다. """
        current_value_allocation = self.holdings * current_prices
        trade_value = target_value_allocation - current_value_allocation
        # 0 가격으로 나누는 것 방지됨 (current_prices >= 1e-6)
        shares_to_trade = trade_value / current_prices

        # 거래 순서는 중요하지 않음 (모든 계산은 현재 가격 기준)
        for i in range(self.n_assets):
            shares = shares_to_trade[i]
            price = current_prices[i]
            commission_multiplier = 1 + self.commission_rate

            if shares > 1e-6: # 매수
                cost = shares * price
                # commission = cost * self.commission_rate # 아래 total_cost 계산에 포함됨
                total_cost = cost * commission_multiplier # 비용 + 수수료

                # 현금 부족 시 구매 가능 수량 조정
                if total_cost > self.cash + 1e-9: # 부동 소수점 오차 고려
                    # 가격 * (1+수수료율) 이 0에 가까운 경우 처리
                    if price * commission_multiplier < 1e-8:
                        continue # 살 수 없음
                    affordable_shares = self.cash / (price * commission_multiplier)
                    if affordable_shares < 1e-6:
                        continue # 살 수 없음
                    # 실제 구매 가능량으로 조정
                    shares = affordable_shares
                    # total_cost = shares * price * commission_multiplier # 다시 계산
                    total_cost = self.cash # 가진 현금 전부 사용 (근사)

                self.holdings[i] += shares
                self.cash -= total_cost

            elif shares < -1e-6: # 매도
                # 실제 팔 수 있는 주식 수는 보유량 이하
                shares_to_sell = min(abs(shares), self.holdings[i])
                if shares_to_sell < 1e-6:
                    continue # 팔 주식 없음

                revenue = shares_to_sell * price
                commission = revenue * self.commission_rate
                total_revenue = revenue - commission

                self.holdings[i] -= shares_to_sell
                self.cash += total_revenue

        # 거래 후 현금이 음수가 되는 경우 방지 (매우 작은 음수값 처리)
        self.cash = max(self.cash, 0.0)

    def render(self, mode="human"):
        """ (선택적) 환경 상태를 간단히 출력합니다. """
        obs = self._get_observation()
        current_prices = obs[:, 3]
        print(f"스텝: {self.current_step}")
        print(f"현금: {self.cash:.2f}")
        print(f"주식 평가액: {np.dot(self.holdings, current_prices):.2f}")
        print(f"총 포트폴리오 가치: {self.portfolio_value:.2f}")

    def close(self):
        """ 환경 관련 리소스를 정리합니다 (현재는 불필요). """
        pass

# --- 신경망 모델 ---
class ActorCritic(nn.Module):
    """
    PPO를 위한 액터-크리틱(Actor-Critic) 네트워크입니다.

    - 입력: 평탄화된 상태 (batch_size, n_assets * n_features)
    - 액터 출력: Dirichlet 분포의 concentration 파라미터 (자산 비중 결정)
    - 크리틱 출력: 상태 가치 (State Value)
    """
    def __init__(self, n_assets, n_features, hidden_dim=DEFAULT_HIDDEN_DIM):
        super(ActorCritic, self).__init__()
        self.input_dim = n_assets * n_features
        
        # 공통 특징 추출 레이어
        self.actor_critic_base = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        ).to(DEVICE)
        
        # 액터 헤드 (Dirichlet 분포 파라미터)
        self.actor_head = nn.Linear(hidden_dim // 2, n_assets).to(DEVICE)
        
        # 크리틱 헤드 (상태 가치)
        self.critic_head = nn.Linear(hidden_dim // 2, 1).to(DEVICE)
        
        # 가중치 초기화 적용
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """ 신경망 가중치를 초기화합니다 (Kaiming He 초기화 사용). """
        if isinstance(module, nn.Linear):
            # ReLU 활성화 함수에 적합한 Kaiming 초기화
            nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0) # 편향은 0으로 초기화
    
    def forward(self, states):
        """
        네트워크의 순전파를 수행합니다.

        Args:
            states (torch.Tensor): 입력 상태 텐서.
                                   (batch_size, n_assets, n_features) 또는 (n_assets, n_features) 형태.

        Returns:
            tuple: (concentration, value)
                   - concentration (torch.Tensor): 액터 헤드의 출력 (Dirichlet 파라미터).
                   - value (torch.Tensor): 크리틱 헤드의 출력 (상태 가치).
        """
        # 상태 텐서 평탄화 (batch_size, n_assets * n_features)
        original_shape = states.shape
        if states.dim() == 3: # 배치 처리
            states = states.view(states.size(0), -1)
        elif states.dim() == 2: # 단일 상태 처리 (추론 시)
            states = states.view(1, -1)
        else:
            raise ValueError(f"지원하지 않는 입력 상태 차원: {original_shape}")

        # NaN/Inf 입력 방지 (안정성 강화)
        if torch.isnan(states).any() or torch.isinf(states).any():
            # logger.warning(f"ActorCritic 입력에 NaN/Inf 발견. 0으로 대체합니다. Shape: {original_shape}")
            states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

        # 공통 베이스 네트워크 통과
        base_output = self.actor_critic_base(states)
        
        # 액터 출력: Concentration 계산 (Softplus + 안정성 위한 조정)
        actor_output = self.actor_head(base_output)
        # Softplus는 양수 값을 보장, 작은 값(1e-4)을 더해 0 방지
        concentration = F.softplus(actor_output) + 1e-4
        # 매우 크거나 작은 값 제한 (수치적 안정성)
        concentration = torch.clamp(concentration, min=1e-4, max=1e4)

        # Concentration 값의 NaN/Inf 확인 (디버깅 및 안정성)
        if torch.isnan(concentration).any() or torch.isinf(concentration).any():
            # logger.warning("Concentration 계산 중 NaN/Inf 발생. 기본값(1)으로 대체합니다.")
            concentration = torch.ones_like(concentration) # 문제 발생 시 기본값 사용

        # 크리틱 출력: 상태 가치
        value = self.critic_head(base_output)
        
        return concentration, value

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
            # 모델 forward는 평탄화하므로 (n_assets, n_features) -> (1, n_assets * n_features) 로 변환 필요
            if state.ndim == 2: # (n_assets, n_features)
                 state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            elif state.ndim == 1: # 이미 평탄화된 경우? (호환성 위해)
                 state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            else:
                 raise ValueError(f"act 메서드: 예상치 못한 NumPy 상태 차원: {state.shape}")
        elif torch.is_tensor(state):
             if state.dim() == 2:
                  state_tensor = state.float().unsqueeze(0).to(DEVICE)
             elif state.dim() == 1:
                  state_tensor = state.float().unsqueeze(0).to(DEVICE)
             else:
                  raise ValueError(f"act 메서드: 예상치 못한 Tensor 상태 차원: {state.shape}")
        else:
            raise TypeError(f"act 메서드: 지원하지 않는 상태 타입: {type(state)}")

        # 그래디언트 계산 비활성화 (추론 모드)
        with torch.no_grad():
            concentration, value = self.forward(state_tensor)
            
            # Dirichlet 분포 생성 및 행동 샘플링
            dist = torch.distributions.Dirichlet(concentration)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        # 결과를 CPU NumPy 배열 및 스칼라 값으로 변환하여 반환
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def evaluate(self, states, actions):
        """
        주어진 상태(states)와 행동(actions)에 대한 로그 확률(log_prob),
        분포 엔트로피(entropy), 상태 가치(value)를 계산합니다.
        PPO 업데이트 시 사용됩니다.

        Args:
            states (torch.Tensor): 상태 배치.
            actions (torch.Tensor): 행동 배치.

        Returns:
            tuple: (log_prob, entropy, value)
                   - log_prob (torch.Tensor): 각 행동의 로그 확률.
                   - entropy (torch.Tensor): 분포의 엔트로피.
                   - value (torch.Tensor): 각 상태의 예측된 가치 (1D Tensor).
        """
        concentration, value = self.forward(states)
        dist = torch.distributions.Dirichlet(concentration)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        # value 텐서의 형태를 (batch_size,)로 일관성 있게 조정
        return log_prob, entropy, value.view(-1)


# --- PPO 알고리즘 구현 ---
class PPO:
    """
    Proximal Policy Optimization (PPO) 알고리즘 클래스입니다.
    Actor-Critic 모델을 사용하여 포트폴리오 관리 문제를 학습합니다.
    """
    def __init__(self, n_assets, n_features,
                 hidden_dim=DEFAULT_HIDDEN_DIM, lr=DEFAULT_LR, gamma=DEFAULT_GAMMA,
                 k_epochs=DEFAULT_K_EPOCHS, eps_clip=DEFAULT_EPS_CLIP,
                 model_path=MODEL_SAVE_PATH, logger=None):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.model_path = model_path
        self.logger = logger or setup_logger() # 로거 없으면 기본 설정 사용
        self.n_assets = n_assets
        self.n_features = n_features # 추가
        
        os.makedirs(model_path, exist_ok=True)
        
        # 정책 네트워크 (현재 정책, 이전 정책)
        self.policy = ActorCritic(n_assets, n_features, hidden_dim).to(DEVICE)
        self.policy_old = ActorCritic(n_assets, n_features, hidden_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict()) # 가중치 복사

        # 옵티마이저 및 손실 함수
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss() # 크리틱 손실용
        # 학습률 스케줄러 (Cosine Annealing) - 선택적
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)

        self.best_reward = -float('inf') # 최고 성능 모델 저장을 위한 변수
        self.obs_rms = None # 학습된 상태 정규화 통계 저장용

        # GPU 설정 (성능 향상 옵션)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 행렬 곱셈 연산 정밀도 설정 (A100/H100 등 TensorFloat32 지원 시 유리)
            # torch.set_float32_matmul_precision('high') # 또는 'medium'
            
    def save_model(self, episode, reward):
        """ 최고 성능 모델의 가중치와 옵티마이저 상태, obs_rms 통계를 저장합니다. """
        if reward > self.best_reward:
            self.best_reward = reward
            save_file = os.path.join(self.model_path, 'best_model.pth')
            try:
                # 저장할 데이터 구성
                checkpoint = {
                'episode': episode,
                'model_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_reward': self.best_reward,
                }
                # obs_rms가 있으면 통계량 추가
                if self.obs_rms is not None:
                    checkpoint.update({
                        'obs_rms_mean': self.obs_rms.mean,
                        'obs_rms_var': self.obs_rms.var,
                        'obs_rms_count': self.obs_rms.count,
                    })

                torch.save(checkpoint, save_file)
                self.logger.info(f"새로운 최고 성능 모델 저장! 에피소드: {episode}, 보상: {reward:.4f} -> {save_file}")
            except Exception as e:
                 self.logger.error(f"모델 저장 중 오류 발생: {e}")

    def load_model(self, model_file=None):
        """ 저장된 모델 가중치와 옵티마이저 상태, 상태 정규화 통계를 불러옵니다. """
        if model_file is None:
            model_file = os.path.join(self.model_path, 'best_model.pth')

        if not os.path.exists(model_file):
            self.logger.warning(f"저장된 모델 파일 없음: {model_file}")
            return False

        try:
            checkpoint = torch.load(model_file, map_location=DEVICE, weights_only=False)

            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.policy_old.load_state_dict(checkpoint['model_state_dict'])

            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.best_reward = checkpoint.get('best_reward', -float('inf'))

            if 'obs_rms_mean' in checkpoint and checkpoint['obs_rms_mean'] is not None:
                if self.obs_rms is None:
                    self.obs_rms = RunningMeanStd(shape=(self.n_assets, self.n_features))
                self.obs_rms.mean = checkpoint['obs_rms_mean']
                self.obs_rms.var = checkpoint['obs_rms_var']
                self.obs_rms.count = checkpoint['obs_rms_count']
                self.logger.info("저장된 상태 정규화(obs_rms) 통계 로드 완료.")
            else:
                self.obs_rms = None

            self.logger.info(f"모델 로드 성공! ({model_file}), 최고 보상: {self.best_reward:.4f}")
            return True

        except (KeyError, TypeError) as load_err:
            self.logger.warning(f"모델 파일 로드 중 오류 ({model_file}): {load_err}. 가중치만 로드 시도합니다.")
            try:
                weights = torch.load(model_file, map_location=DEVICE, weights_only=True)
                self.policy.load_state_dict(weights)
                self.policy_old.load_state_dict(weights)
                self.logger.info(f"모델 가중치 로드 성공 (weights_only=True)! ({model_file})")
                self.best_reward = -float('inf')
                self.obs_rms = None
                return True
            except Exception as e_inner:
                self.logger.error(f"weights_only=True 로도 모델 로드 실패 ({model_file}): {e_inner}")
                return False
        except Exception as e:
            self.logger.error(f"모델 로드 중 예상치 못한 오류 발생 ({model_file}): {e}")
            return False

    def select_action(self, state):
        """ 추론 시 이전 정책(policy_old)을 사용하여 행동을 결정합니다. """
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
        """ 메모리에 저장된 경험을 사용하여 정책(policy)을 업데이트합니다. """
        if not memory.states:
            self.logger.warning("업데이트 시도: 메모리가 비어있습니다.")
            return 0.0

        total_loss_val = 0.0

        try:
            old_states = torch.stack([torch.from_numpy(s).float() for s in memory.states]).to(DEVICE)
            old_actions = torch.stack([torch.from_numpy(a).float() for a in memory.actions]).to(DEVICE)
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
                logprobs, entropy, state_values = self.policy.evaluate(old_states, old_actions)
                ratios = torch.exp(logprobs - old_logprobs.detach())
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values, returns)
                entropy_loss = entropy.mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error(f"손실 계산 중 NaN/Inf 발생! Actor: {actor_loss.item()}, Critic: {critic_loss.item()}, Entropy: {entropy_loss.item()}. 해당 배치 업데이트 건너<0xEB><0x9A><0x8D>니다.")
                    total_loss_val = 0.0
                    break

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_loss_val += loss.item()

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
    logger = logging.getLogger('PortfolioRL')
    if baseline is None:
        baseline = np.zeros_like(state)
    
    try:
        if state.shape != baseline.shape:
            raise ValueError(f"State shape {state.shape}와 Baseline shape {baseline.shape} 불일치")

        state_tensor = torch.from_numpy(state).float().to(DEVICE)
        baseline_tensor = torch.from_numpy(baseline).float().to(DEVICE)
        gradient_sum = torch.zeros_like(state_tensor)
        alphas = torch.linspace(0, 1, steps, device=DEVICE)
    
        for alpha in alphas:
            # 1. 원본 형태로 보간
            interpolated_state_orig = baseline_tensor + alpha * (state_tensor - baseline_tensor)

            # 2. 모델 입력 형태로 변환 (배치 차원 추가)
            interpolated_state_input = interpolated_state_orig.unsqueeze(0) # (1, n_assets, n_features)
            interpolated_state_input.requires_grad_(True)

            # 3. 모델 순전파 및 타겟 설정
            concentration, _ = model.forward(interpolated_state_input)
            target_output = concentration.mean()

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
                     logger.warning(f"IG: 그래디언트 형태 불일치 발생. grad shape: {gradient.shape}, expected: {state_tensor.shape}. 해당 스텝 건너<0xEB><0x9A><0x8D>.")
            # else: # grad가 None인 경우, backward 실패 가능성
                # logger.warning(f"IG: Alpha {alpha:.2f}에서 그래디언트가 None입니다.")

        # 6. 최종 IG 계산
        integrated_grads_tensor = (state_tensor - baseline_tensor) * (gradient_sum / steps)
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
    logger = logging.getLogger('PortfolioRL')
    try:
        if not isinstance(features, np.ndarray) or not isinstance(returns, np.ndarray):
             raise TypeError("입력 데이터는 NumPy 배열이어야 합니다.")
        if features.shape[0] != len(returns):
            raise ValueError(f"features({features.shape[0]})와 returns({len(returns)})의 샘플 수가 일치하지 않습니다.")
        if features.ndim != 3 or returns.ndim != 1:
             raise ValueError(f"입력 데이터 차원 오류. Features: {features.ndim}D, Returns: {returns.ndim}D")

        n_steps, n_assets, n_features_ = features.shape
        X = features.reshape(n_steps, -1)
        y = returns

        model = LinearRegression()
        model.fit(X, y)
        coefficients = model.coef_.reshape(n_assets, n_features_)
        return coefficients

    except Exception as e:
        logger.error(f"Hindsight 선형 모델 학습 중 오류: {e}")
        return None

def compute_feature_weights_drl(ppo_agent, states):
    """
    통합 그래디언트를 사용하여 DRL 에이전트의 각 특성에 대한 중요도(가중치)를 계산합니다.

    Args:
        ppo_agent (PPO): 학습된 PPO 에이전트.
        states (np.ndarray): 분석할 상태 데이터 (n_steps, n_assets, n_features).

    Returns:
        np.ndarray: 각 스텝별 특성 가중치 (n_steps, n_assets, n_features).
                    오류 발생 시 빈 배열 반환.
    """
    logger = logging.getLogger('PortfolioRL')
    all_feature_weights = []

    try:
        if not isinstance(states, np.ndarray) or states.ndim != 3:
             raise ValueError("입력 states는 (n_steps, n_assets, n_features) 형태의 NumPy 배열이어야 합니다.")

        for state in tqdm(states, desc="Calculating DRL Feature Weights", leave=False, ncols=100):
            ig = integrated_gradients(ppo_agent.policy, state)
            all_feature_weights.append(ig)

        if not all_feature_weights:
             logger.warning("DRL 특성 가중치 계산 결과가 비어있습니다.")
             return np.array([])

        return np.stack(all_feature_weights, axis=0)

    except Exception as e:
        logger.error(f"DRL 특성 가중치 계산 중 오류: {e}")
        logger.error(traceback.format_exc())
        return np.array([])

def compute_correlation(arr1, arr2):
    """
    두 NumPy 배열 간의 피어슨 상관계수를 계산합니다.

    Args:
        arr1 (np.ndarray): 첫 번째 배열.
        arr2 (np.ndarray): 두 번째 배열.

    Returns:
        float: 계산된 피어슨 상관계수. 오류 시 0.0 반환.
    """
    logger = logging.getLogger('PortfolioRL')
    try:
        if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray) \
           or arr1.shape != arr2.shape or arr1.size < 2:
            logger.warning(f"상관관계 계산 입력 오류: arr1={arr1.shape}, arr2={arr2.shape}")
            return 0.0

        flat1 = arr1.flatten()
        flat2 = arr2.flatten()

        with np.errstate(divide='ignore', invalid='ignore'):
             correlation_matrix = np.corrcoef(flat1, flat2)

        if not isinstance(correlation_matrix, np.ndarray) or correlation_matrix.shape != (2, 2):
            logger.warning("상관관계 계산 결과가 유효하지 않음 (표준편차 0 등).")
            return 0.0

        correlation = correlation_matrix[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    except Exception as e:
        logger.error(f"상관관계 계산 중 오류: {e}")
        return 0.0

# --- 데이터 처리 함수 ---
def compute_macd(close_series, span_fast=12, span_slow=26):
    """ MACD 지표 계산 (Pandas EWM 사용) """
    ema_fast = close_series.ewm(span=span_fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=span_slow, adjust=False).mean()
    return ema_fast - ema_slow

def compute_rsi(close_series, period=14):
    """ RSI 지표 계산 (Pandas Rolling 사용) """
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # 초기 NaN 방지를 위해 min_periods=period 설정 고려 가능
    avg_gain = gain.rolling(window=period, min_periods=1).mean() # min_periods=1 추가
    avg_loss = loss.rolling(window=period, min_periods=1).mean() # min_periods=1 추가
    rs = avg_gain / (avg_loss + 1e-8) # 0으로 나누기 방지
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50) # 초기 NaN은 중립값 50으로 채움

def fetch_and_preprocess_data(start_date, end_date, tickers, save_path=DATA_SAVE_PATH):
    logger = logging.getLogger('PortfolioRL')
    os.makedirs(save_path, exist_ok=True)
    tickers_str = "_".join(sorted(tickers))
    data_file = os.path.join(save_path, f'portfolio_data_{tickers_str}_{start_date}_{end_date}.pkl')

    if os.path.exists(data_file):
        logger.debug(f"캐시 로드 시도: {data_file}")
        try:
            with open(data_file, 'rb') as f:
                data_array, common_dates = pickle.load(f)
            logger.info(f"캐시 로드 완료. Shape: {data_array.shape}")
            if not isinstance(data_array, np.ndarray) or not isinstance(common_dates, pd.DatetimeIndex):
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
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for ticker in tickers:
        try:
            stock_data_ticker = raw_data.loc[:, pd.IndexSlice[:, ticker]] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data
            cols_to_use = [col for col in required_columns if col in stock_data_ticker.columns]
            if len(cols_to_use) != len(required_columns):
                logger.warning(f"{ticker}: 필요한 컬럼 부족. 건너<0xEB><0x9A><0x8D>.")
                error_count += 1
                continue
            stock_data = stock_data_ticker[cols_to_use].copy()
            if isinstance(raw_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)

            if stock_data.isnull().values.any(): stock_data.ffill(inplace=True).bfill(inplace=True)
            if stock_data.isnull().values.all():
                logger.warning(f"{ticker}: 데이터 전체 NaN. 건너<0xEB><0x9A><0x8D>.")
                error_count += 1; continue

            stock_data['MACD'] = compute_macd(stock_data['Close'])
            stock_data['RSI'] = compute_rsi(stock_data['Close'])
            for window in [14, 21, 100]:
                stock_data[f'MA{window}'] = stock_data['Close'].rolling(window=window, min_periods=1).mean()

            stock_data.bfill(inplace=True).ffill(inplace=True).fillna(0, inplace=True)
            processed_dfs[ticker] = stock_data[FEATURE_NAMES]

        except Exception as e:
            logger.warning(f"{ticker}: 처리 중 오류 - {e}")
            error_count += 1

    valid_tickers = list(processed_dfs.keys())
    if not valid_tickers: logger.error("처리 가능한 유효 종목 없음."); return None, None
    if error_count > 0: logger.warning(f"처리 중 {error_count}개 종목 오류/경고 발생.")

    common_dates = pd.to_datetime(sorted(list(set.intersection(*[set(df.index) for df in processed_dfs.values()])))).tz_localize(None)
    if common_dates.empty: logger.error("모든 유효 티커 공통 거래일 없음."); return None, None

    asset_data = [processed_dfs[ticker].loc[common_dates].astype(np.float32).values for ticker in valid_tickers]
    data_array = np.stack(asset_data, axis=1)
    if np.isnan(data_array).any(): data_array = np.nan_to_num(data_array, nan=0.0)
    logger.info(f"데이터 전처리 완료. Shape: {data_array.shape} ({len(valid_tickers)} 종목)")

    try:
        with open(data_file, 'wb') as f:
            pickle.dump((data_array, common_dates), f)
        logger.info(f"전처리 데이터 저장 완료: {data_file}")
    except Exception as e: logger.error(f"데이터 캐싱 오류: {e}")

    return data_array, common_dates

# --- 성능 지표 계산 및 시각화 함수 ---
def plot_performance(portfolio_values, plot_dir, dates=None, benchmark_values=None, title="Portfolio Performance", filename=None):
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
    logger = logging.getLogger('PortfolioRL')
    if len(portfolio_values) == 0: logger.warning("그래프 생성 실패: 데이터 없음."); return

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 7))
    x_axis = dates if dates is not None and len(dates) == len(portfolio_values) else range(len(portfolio_values))
    xlabel = 'Date' if dates is not None and len(dates) == len(portfolio_values) else 'Trading Days'

    plt.plot(x_axis, portfolio_values, label='PPO Portfolio', linewidth=2)
    if benchmark_values is not None and len(benchmark_values) == len(portfolio_values):
        plt.plot(x_axis, benchmark_values, label='Benchmark', linestyle='--', alpha=0.8, linewidth=1.5)

    plt.title(title, fontsize=16); plt.xlabel(xlabel, fontsize=12); plt.ylabel('Portfolio Value', fontsize=12)
    plt.legend(fontsize=10); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()

    # plot_dir = create_plot_directory() # 제거됨
    if filename is None: filename = f'portfolio_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    save_path = os.path.join(plot_dir, filename)
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"성능 그래프 저장 완료: {save_path}") # INFO 유지
    except Exception as e: logger.error(f"그래프 저장 오류: {e}")
    finally: plt.close()

def calculate_performance_metrics(returns):
    """
    일련의 일일 수익률(daily returns)로부터 주요 성능 지표를 계산합니다.

    Args:
        returns (list or np.ndarray): 일일 수익률 리스트 또는 배열.

    Returns:
        dict: 계산된 성능 지표 딕셔너리.
              {'annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio'}
    """
    if not isinstance(returns, np.ndarray):
        daily_returns = np.array(returns)
    else:
        daily_returns = returns

    # 유효한 수익률 데이터가 없는 경우 기본값 반환
    if daily_returns.size == 0:
        return {'annual_return': 0.0, 'annual_volatility': 0.0, 'sharpe_ratio': 0.0,
                'max_drawdown': 0.0, 'calmar_ratio': 0.0}

    # NaN/Inf 값 처리
    if np.isnan(daily_returns).any() or np.isinf(daily_returns).any():
        daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
    # 연간 수익률 (산술 평균 * 252)
    annual_return = np.mean(daily_returns) * 252
    
    # 연간 변동성 (일간 표준편차 * sqrt(252))
    annual_volatility = np.std(daily_returns) * np.sqrt(252)
    
    # 샤프 비율 (무위험 이자율 0 가정)
    # 변동성이 0에 가까우면 샤프 비율은 정의되지 않거나 0으로 처리
    if annual_volatility > 1e-8: 
        sharpe_ratio = annual_return / annual_volatility
    else:
        sharpe_ratio = 0.0
        
    # 최대 낙폭 (Max Drawdown)
    cumulative_returns = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(cumulative_returns) # 각 시점까지의 최고점
    drawdown = (peak - cumulative_returns) / peak if peak.all() != 0 else np.zeros_like(cumulative_returns) # 0으로 나누기 방지
    max_drawdown = np.max(drawdown) if drawdown.size > 0 else 0.0

    # 칼마 비율 (연간 수익률 / 최대 낙폭)
    # 최대 낙폭이 0에 가까우면 칼마 비율은 정의되지 않거나 0으로 처리
    if max_drawdown > 1e-8:
        calmar_ratio = annual_return / max_drawdown
    else:
        calmar_ratio = 0.0
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio
    }
    
def plot_feature_importance(drl_weights_mean, ref_weights_mean, plot_dir, feature_names=FEATURE_NAMES, filename=None):
    """
    DRL 에이전트와 참조 모델의 평균 특성 중요도를 막대 그래프로 비교 시각화하고 지정된 디렉토리에 저장.

    Args:
        drl_weights_mean (np.ndarray): DRL 에이전트의 평균 특성 가중치 (n_features,).
        ref_weights_mean (np.ndarray): 참조 모델의 평균 특성 가중치 (n_features,).
        plot_dir (str): 그래프 이미지를 저장할 디렉토리 경로.
        feature_names (list): 특성 이름 리스트.
        filename (str, optional): 저장할 파일 이름. 미지정 시 자동 생성.
    """
    if not isinstance(drl_weights_mean, np.ndarray) or not isinstance(ref_weights_mean, np.ndarray) \
            or drl_weights_mean.shape != ref_weights_mean.shape \
            or len(drl_weights_mean) != len(feature_names):
        return
        
    plt.figure(figsize=(15, 7)) # 너비 증가
    num_features = len(feature_names)
    x = np.arange(num_features)

    # DRL 에이전트 중요도
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(x, drl_weights_mean, color='skyblue')
    plt.ylabel('Importance Score')
    plt.title('DRL Agent Mean Feature Importance')
    plt.xticks(x, feature_names, rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--')
    # 막대 위에 값 표시 (소수점 2자리)
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom' if yval >= 0 else 'top', ha='center')

    # 참조 모델 중요도
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(x, ref_weights_mean, color='lightcoral')
    plt.ylabel('Importance Score')
    plt.title('Reference Model Mean Feature Importance')
    plt.xticks(x, feature_names, rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--')
    # 막대 위에 값 표시 (소수점 2자리)
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom' if yval >= 0 else 'top', ha='center')

    
    plt.tight_layout()
    
    # 파일 저장
    if filename is None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'feature_importance_comparison_{current_time}.png'
    save_path = os.path.join(plot_dir, filename)

    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight') # bbox_inches 추가
    except Exception as e:
        pass # 오류 로깅은 주석 처리됨
    finally:
        plt.close()
    
def plot_integrated_gradients(ig_values_mean, plot_dir, feature_names=FEATURE_NAMES, title="Mean Integrated Gradients", filename=None):
    """
    평균 통합 그래디언트 값을 막대 그래프로 시각화하고 지정된 디렉토리에 저장.

    Args:
        ig_values_mean (np.ndarray): 평균 통합 그래디언트 값 배열 (n_features,).
        plot_dir (str): 그래프 이미지를 저장할 디렉토리 경로.
        feature_names (list): 특성 이름 리스트.
        title (str, optional): 그래프 제목.
        filename (str, optional): 저장할 파일 이름. 미지정 시 자동 생성.
    """
    if not isinstance(ig_values_mean, np.ndarray) or len(ig_values_mean) != len(feature_names):
        return # 데이터 오류 시 함수 종료

    plt.figure(figsize=(12, 6))
    num_features = len(feature_names)
    x = np.arange(num_features)

    bars = plt.bar(x, ig_values_mean, color='mediumpurple')
    plt.ylabel('Mean Attribution Score')
    plt.title(title)
    plt.xticks(x, feature_names, rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--')

    # 막대 위에 값 표시
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom' if yval >= 0 else 'top', ha='center')

    plt.tight_layout()

    if filename is None:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'integrated_gradients_mean_{current_time}.png'
    save_path = os.path.join(plot_dir, filename)

    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        pass # 오류 로깅은 주석 처리됨
    finally:
        plt.close()
    
# --- 학습 및 평가 함수 ---
def print_memory_stats(logger):
    """ 현재 GPU 메모리 사용량 및 캐시 상태를 로깅합니다. """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU 메모리 사용량: {allocated:.2f} MB / 예약됨: {reserved:.2f} MB")

def train_ppo_agent(env: StockPortfolioEnv, ppo_agent: PPO, max_episodes: int, max_timesteps: int,
                      update_timestep: int, logger: logging.Logger):
    """
    주어진 환경에서 PPO 에이전트를 학습시킵니다.
    주기적 상세 로그는 DEBUG 레벨로 변경, 최종 결과는 INFO 유지.
    """
    logger.info(f"PPO 학습 시작: {max_episodes} 에피소드, 에피소드당 최대 {max_timesteps} 스텝")
    logger.info(f"정책 업데이트 주기: {update_timestep} 스텝")

    memory = Memory()
    episode_raw_rewards = []
    training_start_time = time.time()
    total_steps = 0
    update_count = 0

    pbar = tqdm(range(max_episodes), desc="Training Episodes", file=sys.stdout, ncols=100)

    for episode in pbar:
        if episode % 20 == 0:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        state, _ = env.reset()
        episode_raw_reward_sum = 0.0
        current_episode_steps = 0

        logger.debug(f"--- 에피소드 {episode+1} 시작 ---") # DEBUG로 변경

        terminated = False
        truncated = False
        while not terminated and not truncated:
            total_steps += 1
            current_episode_steps += 1
            action, log_prob, value = ppo_agent.policy_old.act(state)
            next_state, reward_norm, terminated, truncated_env, info = env.step(action)
            is_terminal_step = terminated or (current_episode_steps >= max_timesteps)
            memory.add_experience(state, action, log_prob, reward_norm, is_terminal_step, value)
            state = next_state
            episode_raw_reward_sum += info.get('raw_reward', 0.0)

            if total_steps % update_timestep == 0 and len(memory.states) >= ppo_agent.k_epochs:
                update_loss = ppo_agent.update(memory)
                update_count += 1
                memory.clear_memory()
                if update_loss != 0.0:
                    logger.debug(f" 정책 업데이트 {update_count} (총 {total_steps} 스텝). Loss: {update_loss:.4f}")

            if current_episode_steps >= max_timesteps:
                truncated = True

        episode_raw_rewards.append(episode_raw_reward_sum)
        ppo_agent.save_model(episode, episode_raw_reward_sum)

        if (episode + 1) % 5 == 0 or episode == max_episodes - 1:
            lookback = max(10, len(episode_raw_rewards) // 10)
            avg_raw_reward = np.mean(episode_raw_rewards[-lookback:])
            final_value = info.get('portfolio_value', env.portfolio_value)
            pbar.set_postfix({
                f'AvgRew(raw, L{lookback})': f'{avg_raw_reward:.2f}',
                'LastValue': f'{final_value:,.0f}',
                'Steps': f'{total_steps:,}'
            }, refresh=True)

            # 주기적 상세 로그는 DEBUG
            if (episode + 1) % 50 == 0:
                logger.debug(f" 에피소드 {episode+1} 완료. 최근 {lookback} 평균 Raw 보상: {avg_raw_reward:.4f}, 최종 가치: {final_value:.2f}")

    pbar.close()

    total_training_time = time.time() - training_start_time
    avg_step_time = total_training_time / total_steps if total_steps > 0 else 0
    logger.info(f"\n총 학습 시간: {total_training_time:.2f}초 ({avg_step_time:.4f}초/스텝)") # 최종 결과는 INFO

    if hasattr(env, 'obs_rms') and env.normalize_states and env.obs_rms is not None:
        ppo_agent.obs_rms = env.obs_rms
        logger.info("학습 환경의 상태 정규화(obs_rms) 통계를 에이전트에 저장했습니다.") # INFO 유지
    else:
        logger.warning("학습 환경의 obs_rms 통계를 에이전트에 저장하지 못했습니다.") # 경고 유지

    return episode_raw_rewards

def evaluate_ppo_agent(env: StockPortfolioEnv, ppo_agent: PPO, max_test_timesteps: int, load_best_model=True):
    """
    학습된 PPO 에이전트를 평가합니다.
    평가 시작/종료 메시지만 INFO로 유지.
    """
    logger = ppo_agent.logger

    if load_best_model:
        if not ppo_agent.load_model():
            logger.error("모델 로드 실패, 평가 중단.")
            return None

    state, info_init = env.reset()
    total_raw_reward = 0.0
    portfolio_values = [info_init['portfolio_value']]
    daily_returns = []
    asset_weights = [info_init['weights']]
    chosen_actions = []

    terminated, truncated = False, False
    step_count = 0

    logger.info("PPO 에이전트 평가 시작 (결정론적 행동)...") # INFO 유지
    pbar_eval = tqdm(total=max_test_timesteps, desc="Evaluating Agent", file=sys.stdout, ncols=100)

    while not terminated and not truncated and step_count < max_test_timesteps:
        normalized_state = state
        if ppo_agent.obs_rms is not None and ppo_agent.obs_rms.count > RMS_EPSILON:
            normalized_state = np.clip(
                (state - ppo_agent.obs_rms.mean) / np.sqrt(ppo_agent.obs_rms.var + RMS_EPSILON),
                -CLIP_OBS, CLIP_OBS)

        with torch.no_grad():
            state_tensor = torch.from_numpy(normalized_state).float().unsqueeze(0).to(DEVICE)
            if state_tensor.dim() == 2: state_tensor = state_tensor.unsqueeze(0)
            action = np.ones(env.n_assets) / env.n_assets # 기본값
            try:
                 if state_tensor.dim() != 3:
                      raise ValueError(f"예상치 못한 상태 텐서 형태: {state_tensor.shape}")
                 concentration, _ = ppo_agent.policy.forward(state_tensor)
                 action = torch.distributions.Dirichlet(concentration).mean.squeeze(0).cpu().numpy()
            except Exception as forward_err:
                 logger.error(f"평가 중 모델 forward 오류: {forward_err}")

        chosen_actions.append(action)
        next_state, _, terminated, truncated_env, info = env.step(action)
        portfolio_values.append(info['portfolio_value'])
        daily_returns.append(info['return'])
        asset_weights.append(info.get('weights', np.zeros(env.n_assets)))
        total_raw_reward += info.get('raw_reward', 0.0)
        state = next_state
        step_count += 1
        pbar_eval.update(1)
        if step_count >= max_test_timesteps: truncated = True

    pbar_eval.close()
    logger.info(f"평가 종료. 총 스텝: {step_count}") # INFO 유지

    return {
        'episode_reward': total_raw_reward,
        'portfolio_values': portfolio_values,
        'returns': daily_returns,
        'weights': asset_weights,
        'actions': chosen_actions
    }

# --- 메인 실행 함수 ---
def main():
    """ 메인 실행 함수: 데이터 로드, 학습, 평가, 결과 분석 및 시각화 수행 """
    current_time_seed = int(time.time())
    np.random.seed(current_time_seed)
    torch.manual_seed(current_time_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(current_time_seed)

    logger = setup_logger()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_plot_dir = os.path.join(PLOT_SAVE_PATH, run_timestamp)
    os.makedirs(run_plot_dir, exist_ok=True)
    logger.info(f"결과 저장 폴더: {run_plot_dir}")

    # --- 시스템 환경 확인 (INFO 레벨 유지) ---
    logger.info("\n" + "="*15 + " 시스템 환경 확인 " + "="*15)
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
    logger.info("="*48)

    # --- 데이터 준비 (INFO 레벨 유지) ---
    logger.info("\n" + "="*18 + " 데이터 준비 " + "="*18)
    data_array, common_dates = fetch_and_preprocess_data(
        TRAIN_START_DATE, TEST_END_DATE, STOCK_TICKERS
    )
    if data_array is None: logger.error("데이터 준비 실패. 종료."); return
    split_date = pd.Timestamp(TEST_START_DATE).tz_localize(None)
    if not isinstance(common_dates, pd.DatetimeIndex): common_dates = pd.to_datetime(common_dates)
    common_dates_naive = common_dates.tz_localize(None)
    try:
        split_idx = np.searchsorted(common_dates_naive, split_date)
    except Exception as e:
        logger.error(f"데이터 분할 인덱스 검색 오류: {e}. 대체 검색 사용.")
        split_idx_arr = np.where(common_dates_naive >= split_date)[0]
        if len(split_idx_arr) == 0: logger.error(f"분할 날짜({TEST_START_DATE}) 이후 데이터 없음. 종료."); return
        split_idx = split_idx_arr[0]
    if not (0 < split_idx < len(common_dates)):
        logger.error(f"데이터 분할 오류: 분할 인덱스({split_idx}) 유효하지 않음. 종료.")
        return # return 들여쓰기 수정
    train_data = data_array[:split_idx]
    test_data = data_array[split_idx:]
    test_dates = common_dates[split_idx:]
    logger.info(f" 훈련 데이터: {train_data.shape} ({common_dates[0].date()} ~ {common_dates[split_idx-1].date()})")
    logger.info(f" 테스트 데이터: {test_data.shape} ({test_dates[0].date()} ~ {test_dates[-1].date()})")
    logger.info("="*48)

    # --- 환경 및 에이전트 설정 (INFO 레벨 유지) ---
    logger.info("\n" + "="*14 + " 환경 및 에이전트 설정 " + "="*14)
    train_env = StockPortfolioEnv(train_data, normalize_states=True)
    n_assets, n_features = train_env.n_assets, train_env.n_features
    logger.info(f" 환경 설정: 자산 수={n_assets}, 피처 수={n_features}")
    ppo_agent = PPO(n_assets, n_features, logger=logger, lr=DEFAULT_LR)
    logger.info(f" PPO 에이전트 생성 완료 (lr={DEFAULT_LR})")
    logger.info("="*48)

    logger.info("\n" + "="*16 + " PPO 에이전트 학습 " + "="*16)
    max_episodes_train = 500
    max_timesteps_train = train_env.max_episode_length
    training_rewards = train_ppo_agent(
        train_env, ppo_agent, max_episodes_train, max_timesteps_train, PPO_UPDATE_TIMESTEP, logger
    )
    logger.info(" PPO 학습 완료!")
    logger.info("="*48)

    logger.info("\n" + "="*16 + " PPO 에이전트 평가 " + "="*16)
    test_env = StockPortfolioEnv(test_data, normalize_states=False)
    max_test_timesteps = len(test_data) - 1
    test_results = evaluate_ppo_agent(test_env, ppo_agent, max_test_timesteps, load_best_model=True)
    if test_results is None: logger.error("테스트 실패. 성능 분석 생략."); return
    logger.info(" 테스트 완료!")
    logger.info("="*48)

    logger.info("\n" + "="*13 + " 성능 분석 및 시각화 " + "="*13)
    metrics = calculate_performance_metrics(test_results['returns'])
    logger.info("--- 포트폴리오 성능 지표 ---")
    logger.info(f" 연간 수익률: {metrics['annual_return']:.2%}")
    logger.info(f" 연간 변동성: {metrics['annual_volatility']:.2%}")
    logger.info(f" 샤프 비율: {metrics['sharpe_ratio']:.2f}")
    logger.info(f" 최대 낙폭: {metrics['max_drawdown']:.2%}")
    logger.info(f" 칼마 비율: {metrics['calmar_ratio']:.2f}")
    logger.info(f" 테스트 기간 총 Raw 보상: {test_results['episode_reward']:.4f}")

    # 그래프 저장 폴더 생성
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_plot_dir = os.path.join(PLOT_SAVE_PATH, run_timestamp)
    os.makedirs(run_plot_dir, exist_ok=True)

    plot_performance(
        test_results['portfolio_values'], dates=test_dates,
        title="PPO Portfolio Performance (Evaluation)", filename=f"PPO_performance_{run_timestamp}.png",
        plot_dir=run_plot_dir
    )
    logger.info("="*48)

    # --- 설명 가능한 AI (XAI) 분석 (INFO 레벨 유지) ---
    logger.info("\n" + "="*12 + " 설명 가능한 AI (XAI) 분석 " + "="*12)
    try:
        num_test_steps = len(test_results['returns'])
        if num_test_steps == 0: raise ValueError("테스트 결과 수익률 없음")
        test_data_aligned = test_data[:num_test_steps]
        returns_aligned = np.array(test_results['returns'])

        logger.info(" DRL 에이전트 특성 가중치 계산 중...")
        drl_weights_ts = compute_feature_weights_drl(ppo_agent, test_data_aligned)
        if drl_weights_ts.size == 0:
            logger.error("DRL 가중치 계산 실패. XAI 분석 일부 생략.")
        else:
            drl_weights_mean = drl_weights_ts.mean(axis=(0, 1))
            logger.info(" 참조 모델(선형 회귀) 특성 가중치 계산 중...")
            ref_weights = linear_model_hindsight(test_data_aligned, returns_aligned)
            if ref_weights is None:
                logger.error("참조 모델 가중치 계산 실패. 비교 분석 생략.")
            else:
                ref_weights_mean = ref_weights.mean(axis=0)
                plot_feature_importance(drl_weights_mean, ref_weights_mean, plot_dir=run_plot_dir,
                                        filename=f"feature_importance_{run_timestamp}.png")
                correlation = compute_correlation(drl_weights_mean, ref_weights_mean)
                logger.info(f" DRL과 참조 모델 평균 특성 중요도 상관계수: {correlation:.4f}")

            plot_integrated_gradients(
                drl_weights_mean, plot_dir=run_plot_dir,
                title="DRL Agent Mean Integrated Gradients", filename=f"integrated_gradients_mean_{run_timestamp}.png"
            )
        logger.info(" XAI 분석 완료!")

    except Exception as e:
        logger.error(f" XAI 분석 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
    logger.info("="*48)

    logger.info("\n===== 프로그램 종료 =====")

if __name__ == "__main__":
    main()