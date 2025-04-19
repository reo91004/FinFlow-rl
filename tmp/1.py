import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import ta
from tqdm import tqdm
import os
import pickle
import logging
import sys
import gc
import time
import torch.nn.functional as F

# GPU 사용 가능 여부 확인
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 랜덤 시드 설정
# np.random.seed(42)
# torch.manual_seed(42)

STOCK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "JPM", "JNJ", "PG", "V"
]

def setup_logger(log_dir='logs'):
    """로깅 설정"""
    # 로그 디렉토리 생성
    os.makedirs(log_dir, exist_ok=True)
    
    # 현재 시간으로 로그 파일명 생성
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_log_{current_time}.txt')
    
    # 로거 설정
    logger = logging.getLogger('PortfolioRL')
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러 - 모든 로그 기록
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # 콘솔 핸들러 - 중요한 정보만 출력
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 포맷터
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
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

class StockPortfolioEnv(gym.Env):
    """
    현금 기반 포트폴리오 거래 환경 (PortfolioEnv 로직 기반)
    관측: 각 자산의 10개 피처 (OHLCV, MACD, RSI, MA14, MA21, MA100)
    행동: 각 자산의 투자 비중 (0 ~ 1, 총합 1)
    보상: 하루 단위 로그 수익률
    (상태 정규화 추가)
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, data: np.ndarray, initial_cash=1e6, commission_rate=0.005, max_episode_length=200, normalize_states=True, gamma=0.99):
        super(StockPortfolioEnv, self).__init__()
        self.data = data # (n_steps, n_assets, n_features)
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.max_episode_length = max_episode_length # 에피소드 길이 제한
        self.normalize_states = normalize_states
        self.gamma = gamma # 할인 계수 (보상 정규화용)
        
        self.n_steps = data.shape[0]
        self.n_assets = data.shape[1]
        self.n_features = data.shape[2]
        
        # 상태 공간: (자산 수, 피처 수)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.n_assets, self.n_features), 
            dtype=np.float32
        )
        
        # 행동 공간: 각 자산의 목표 비중 (합계 1)
        self.action_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # 상태 정규화용 RunningMeanStd 초기화
        if self.normalize_states:
            self.obs_rms = RunningMeanStd(shape=(self.n_assets, self.n_features))
            self.ret_rms = RunningMeanStd(shape=())
            self.clip_obs = 10.0
            self.clip_reward = 10.0
            self.epsilon = 1e-8
            self.returns = np.zeros(1) # 보상 정규화용

        # 내부 변수 초기화 (reset에서 호출)
        self.current_step = 0
        self.cash = 0.0
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = 0.0
    
    def _normalize_obs(self, obs):
        if not self.normalize_states: return obs
        self.obs_rms.update(obs.reshape(1, self.n_assets, self.n_features)) # 배치 차원 추가하여 업데이트
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                     -self.clip_obs, self.clip_obs)
    
    def _normalize_reward(self, reward):
        if not self.normalize_states: return reward
        self.returns = self.gamma * self.returns + reward
        self.ret_rms.update(self.returns)
        return np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), 
                     -self.clip_reward, self.clip_reward)

    def reset(self, *, seed=None, options=None, start_index=None):
        super().reset(seed=seed)
        
        # 에피소드 길이 제한 내에서 무작위 시작점 설정
        if start_index is None:
            if self.n_steps > self.max_episode_length:
                start_index = np.random.randint(0, self.n_steps - self.max_episode_length)
            else:
                start_index = 0 # 데이터가 짧으면 처음부터 시작
        self.current_step = start_index
        
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = self.cash # 초기 가치는 현금
        
        # 보상 정규화용 returns 초기화
        if self.normalize_states:
            self.returns = np.zeros(1) 
        
        # 기타 히스토리 초기화 (필요 시)
        self.feature_history = []
        self.return_history = []
        self.weights = np.ones(self.n_assets) / self.n_assets # 초기 가중치
        
        observation = self._get_observation()
        normalized_observation = self._normalize_obs(observation)
        info = {} # 초기 info는 비어있음
        
        return normalized_observation.astype(np.float32), info
    
    def _get_observation(self):
        # 현재 스텝의 데이터 반환 (자산 수, 피처 수)
        # 정규화는 reset과 step에서 처리
        return self.data[self.current_step]

    def step(self, action):
        # 행동 정규화 (비중 합 1)
        action = np.clip(action, 0, 1)
        if action.sum() > 1e-6: # 합이 0에 가까우면 균등 분배
            action = action / action.sum()
        else:
            action = np.ones(self.n_assets) / self.n_assets
            
        # 현재 상태 및 가격 정보 (정규화 안된 원래 데이터 사용)
        obs = self.data[self.current_step]
        current_prices = obs[:, 3] # 종가 (인덱스 3)
        current_prices = np.maximum(current_prices, 1e-6) # 0 가격 방지
        
        # 이전 포트폴리오 가치 계산
        prev_portfolio_value = self.cash + np.dot(self.holdings, current_prices)
        if prev_portfolio_value <= 1e-6: # 가치가 0에 가까우면 에피소드 종료 (파산)
            terminated = True
            raw_reward = -10.0 # 큰 음수 보상 (raw)
            info = {
                "portfolio_value": 0.0, 
                "cash": 0.0, 
                "holdings": self.holdings.copy(),
                "return": -1.0, # 파산 시 수익률 -100%
                "raw_reward": raw_reward # raw_reward 추가
            }
            
            # 마지막 관측 얻기 및 정규화
            last_obs = self._get_observation()
            normalized_last_obs = self._normalize_obs(last_obs)
            normalized_reward = self._normalize_reward(raw_reward)
            return normalized_last_obs.astype(np.float32), float(normalized_reward), terminated, False, info
             
        # 목표 자산 가치 및 현재 자산 가치
        target_value_allocation = action * prev_portfolio_value
        current_value_allocation = self.holdings * current_prices
        
        # 거래량 계산 (목표 가치 - 현재 가치) / 가격
        trade_value = target_value_allocation - current_value_allocation
        # 0 가격 방지 후 나누기
        shares_to_trade = trade_value / current_prices 
        
        # 거래 실행 및 수수료 계산
        for i in range(self.n_assets):
            if shares_to_trade[i] > 1e-6: # 매수
                cost = shares_to_trade[i] * current_prices[i]
                commission = cost * self.commission_rate
                total_cost = cost + commission
                
                # 구매 가능 수량 재계산 (현금 부족 시)
                if total_cost > self.cash:
                    # Check for zero price before division
                    if current_prices[i] * (1 + self.commission_rate) > 1e-8:
                         affordable_shares = self.cash / (current_prices[i] * (1 + self.commission_rate))
                    else:
                         affordable_shares = 0.0
                    if affordable_shares < 1e-6: continue # 살 수 없으면 통과
                    shares_to_trade[i] = affordable_shares # 살 수 있는 만큼만 수정
                    cost = shares_to_trade[i] * current_prices[i]
                    commission = cost * self.commission_rate
                    total_cost = cost + commission
                    
                self.holdings[i] += shares_to_trade[i]
                self.cash -= total_cost
                
            elif shares_to_trade[i] < -1e-6: # 매도
                shares_to_sell = min(abs(shares_to_trade[i]), self.holdings[i])
                if shares_to_sell < 1e-6: continue # 팔 주식 없으면 통과
                
                revenue = shares_to_sell * current_prices[i]
                commission = revenue * self.commission_rate
                total_revenue = revenue - commission
                
                self.holdings[i] -= shares_to_sell
                self.cash += total_revenue
                
        # 다음 스텝으로 이동
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False # Truncated는 학습 루프에서 max_episode_length 도달 시 설정
        
        # 다음 스텝 관측 및 새 포트폴리오 가치 계산
        if terminated:
            next_obs_raw = obs # 마지막 관측값 사용 (정규화 안된 것)
        else:
            next_obs_raw = self.data[self.current_step]
            
        new_prices = next_obs_raw[:, 3] # 다음 날 종가
        new_prices = np.maximum(new_prices, 1e-6) # 0 가격 방지
        new_portfolio_value = self.cash + np.dot(self.holdings, new_prices)
        
        # 보상 계산: 로그 수익률
        raw_reward = np.log(new_portfolio_value / prev_portfolio_value + 1e-8) # 1e-8 더해서 log(0) 방지
        
        # NaN/inf 보상 처리
        if np.isnan(raw_reward) or np.isinf(raw_reward):
            raw_reward = -1.0 # 문제가 생기면 페널티
        
        self.portfolio_value = new_portfolio_value
        # 가중치 계산 시 0으로 나누기 방지
        if new_portfolio_value > 1e-8:
             self.weights = (self.holdings * new_prices) / new_portfolio_value
        else:
             self.weights = np.zeros_like(self.holdings)
        
        # 정보 업데이트 (정규화 안된 값 사용)
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "weights": self.weights.copy(),
            "return": new_portfolio_value / prev_portfolio_value - 1 if prev_portfolio_value > 1e-8 else 0.0,
            "raw_reward": raw_reward # raw_reward 추가
        }
        
        # 다음 상태 정규화 및 보상 정규화
        normalized_next_obs = self._normalize_obs(next_obs_raw)
        normalized_reward = self._normalize_reward(raw_reward)
        
        # 다음 상태 반환 (정규화된 값)
        return normalized_next_obs.astype(np.float32), float(normalized_reward), terminated, truncated, info

    def render(self, mode="human"):
        obs = self.data[self.current_step]
        current_prices = obs[:, 3]
        print(f"Step: {self.current_step}")
        print(f"Cash: {self.cash:.2f}")
        print(f"Holdings Value: {np.dot(self.holdings, current_prices):.2f}")
        print(f"Portfolio Value: {self.portfolio_value:.2f}")
        # print(f"Holdings (shares): {self.holdings}")
        # print(f"Current Weights: {self.weights}")

    def close(self):
        pass

class Memory:
    """
    경험 저장을 위한 간단한 메모리 버퍼 (NumPy 기반)
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = [] # 상태 가치 추가
        
    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]
    
    def add_experience(self, state, action, logprob, reward, is_terminal, value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.values.append(value)
        
class ActorCritic(nn.Module):
    """
    PPO 알고리즘을 위한 Actor-Critic 네트워크 - PortfolioEnv 호환
    """
    
    def __init__(self, n_assets, n_features, hidden_dim=128): # hidden_dim을 PortfolioEnv와 맞춤 (512 -> 128)
        super(ActorCritic, self).__init__()
        self.input_dim = n_assets * n_features # 입력 차원 수정
        
        # 단순화된 네트워크 구조
        self.actor_critic_base = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        ).to(device)
        
        # 액터 헤드: Dirichlet 분포의 concentration 파라미터 출력
        self.actor_head = nn.Linear(hidden_dim // 2, n_assets).to(device)
        
        # 크리틱 헤드: 상태 가치 출력
        self.critic_head = nn.Linear(hidden_dim // 2, 1).to(device)
        
        # 모델 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, states):
        # 상태 평탄화 (batch_size, n_assets, n_features) -> (batch_size, n_assets * n_features)
        original_shape = states.shape
        if states.dim() == 3:
            states = states.reshape(states.size(0), -1)
        elif states.dim() == 2: # 단일 상태 입력 처리 (act 메서드용)
             states = states.reshape(1, -1)
        
        # 입력 값 확인 (디버깅용)
        if torch.isnan(states).any() or torch.isinf(states).any():
            print(f"Warning: Input state contains NaN or Inf in ActorCritic forward. Shape: {original_shape}")
            # NaN/Inf를 0으로 대체하거나 다른 처리 방식 고려
            states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

        # 공통 베이스 네트워크 통과
        base_output = self.actor_critic_base(states)
        
        # 액터 출력: Concentration (Softplus + 클리핑)
        actor_output = self.actor_head(base_output)
        concentration = F.softplus(actor_output)
        
        # NaN 방지를 위한 클램핑 및 작은 값 추가 조정
        # min 값을 약간 높이고, 더해주는 값도 조정
        concentration = torch.clamp(concentration, min=1e-4, max=1e4) + 1e-4 
        
        # Concentration 값 확인 (디버깅용)
        if torch.isnan(concentration).any() or torch.isinf(concentration).any():
             print(f"Warning: Concentration became NaN/Inf after clamp/add. Actor output before softplus: {actor_output.detach().cpu().numpy()}")
             # 문제 발생 시 농도를 기본값(예: 1)으로 설정하는 등의 예외 처리 고려
             concentration = torch.ones_like(concentration) # 임시방편

        # 크리틱 출력: 상태 가치
        value = self.critic_head(base_output)
        
        return concentration, value

    def act(self, state):
        # 상태를 텐서로 변환 (NumPy 입력 가정)
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(device)
            if state.dim() == 2: # 단일 상태면 배치 차원 추가
                 state = state.unsqueeze(0)
                 
        with torch.no_grad():
            concentration, value = self.forward(state)
            
            # Dirichlet 분포로부터 액션 샘플링
            dist = torch.distributions.Dirichlet(concentration)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        # 결과를 NumPy 배열과 스칼라로 변환하여 반환
        return action.squeeze(0).cpu().numpy(), log_prob.squeeze(0).item(), value.squeeze(0).item()

    def evaluate(self, states, actions):
        concentration, value = self.forward(states)
        dist = torch.distributions.Dirichlet(concentration)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value

class PPO:
    """
    PPO 알고리즘 구현 - NumPy 입력 처리
    """
    
    def __init__(self, n_assets, n_features, hidden_dim=128, # 파라미터 이름 변경 및 기본값 설정
                 lr=3e-4, gamma=0.99, k_epochs=10, eps_clip=0.2,
                 model_path='models', logger=None):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.model_path = model_path
        self.logger = logger or logging.getLogger('PortfolioRL')
        self.n_assets = n_assets # 추가
        
        os.makedirs(model_path, exist_ok=True)
        
        # 정책 네트워크 (PortfolioEnv 호환)
        self.policy = ActorCritic(n_assets, n_features, hidden_dim).to(device)
        self.policy_old = ActorCritic(n_assets, n_features, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        self.best_reward = -float('inf')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision('high')
            
    def save_model(self, episode, reward):
        """
        모델 저장 - 최고 성능 모델만 저장
        """
        if reward > self.best_reward:
            self.best_reward = reward
            torch.save({
                'episode': episode,
                'model_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_reward': self.best_reward
            }, os.path.join(self.model_path, 'best_model.pth'))
            print(f"\n새로운 최고 성능 모델 저장! 에피소드: {episode}, 보상: {reward:.4f}")
    
    def load_model(self, model_path=None):
        """
        저장된 모델 불러오기
        """
        if model_path is None:
            model_path = os.path.join(self.model_path, 'best_model.pth')
            
        if os.path.exists(model_path):
            # --- 수정: weights_only=True 추가 --- 
            try:
                 checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                 self.policy.load_state_dict(checkpoint['model_state_dict'])
                 self.policy_old.load_state_dict(checkpoint['model_state_dict'])
                 # Optimizer state 와 best_reward 는 weights_only=True 에서 로드되지 않음
                 # 필요하다면 별도로 저장/로드하거나 False로 로드 (보안 위험 감수)
                 # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 # self.best_reward = checkpoint['best_reward']
                 print(f"\n모델 가중치 불러오기 성공! ({model_path})")
                 # best_reward 는 다시 초기화하거나 다른 방법으로 추적 필요
                 self.best_reward = -float('inf') 
                 return True
            except Exception as e:
                 print(f"\n모델 로드 중 오류 발생 ({model_path}): {e}")
                 print("weights_only=False로 다시 시도합니다 (보안 위험 참고)...")
                 try:
                      # Fallback to weights_only=False if True fails (and log warning)
                      checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                      self.policy.load_state_dict(checkpoint['model_state_dict'])
                      self.policy_old.load_state_dict(checkpoint['model_state_dict'])
                      # 옵티마이저와 보상 로드 시도
                      if 'optimizer_state_dict' in checkpoint:
                           self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                      if 'best_reward' in checkpoint:
                           self.best_reward = checkpoint['best_reward']
                      print(f"\n모델 불러오기 성공 (weights_only=False)! 최고 보상: {self.best_reward:.4f}")
                      return True
                 except Exception as e2:
                      print(f"\nweights_only=False 로도 모델 로드 실패 ({model_path}): {e2}")
                      return False
        else:
            print("\n저장된 모델이 없습니다.")
            return False
    
    def select_action(self, state):
        # act 메서드는 NumPy 배열 반환
        action, _, _ = self.policy_old.act(state)
        return action
    
    def compute_returns_and_advantages(self, rewards, is_terminals, values):
        """ Generalized Advantage Estimation (GAE) 계산 """
        # --- Debugging Start ---
        # print(f"[DEBUG GAE] Input shapes - rewards: {len(rewards)}, is_terminals: {len(is_terminals)}, values: {getattr(values, 'shape', 'N/A')}")
        if not isinstance(values, np.ndarray) or values.ndim < 1:
             # print(f"[DEBUG GAE] ERROR: values is not a >= 1D numpy array! Type: {type(values)}, Shape: {getattr(values, 'shape', 'N/A')}")
             # Handle error appropriately, maybe return default tensors
             default_tensor = torch.zeros(len(rewards), device=device) if rewards else torch.tensor(0.0, device=device) # Handle empty rewards case
             return default_tensor, default_tensor
        # --- Debugging End ---

        returns = []
        advantages = []
        last_value = 0.0 # Initialize last_value
        
        # Check if values is empty before accessing index -1
        if values.size > 0:
             try: # Add try-except block for robust indexing
                 last_value = values[-1] # 마지막 상태의 가치
             except IndexError:
                  # print(f"[DEBUG GAE] IndexError accessing values[-1]. values shape: {values.shape}")
                  last_value = 0.0 # Default value if index error occurs
        else:
            # print("[DEBUG GAE] Warning: values array is empty.")
            # If values is empty, rewards should ideally also be empty. 
            # If not, it indicates a logic error elsewhere.
            # We return empty tensors or handle based on expected behavior.
            return torch.tensor([], device=device), torch.tensor([], device=device)


        last_gae_lam = 0
        lambda_gae = 0.95 # GAE lambda
        
        for i in reversed(range(len(rewards))):
            try: # Add try-except block for robust indexing
                # Check if index i is valid for values array
                if i >= len(values):
                    # print(f"[DEBUG GAE] Index {i} out of bounds for values array (shape: {values.shape}). Skipping step.")
                    continue
                current_value = values[i]
                
                if is_terminals[i]:
                    delta = rewards[i] - current_value
                    last_gae_lam = delta # 에피소드 끝이면 다음 스텝 가치 0
                else:
                    # Check index i+1 validity carefully
                    next_value = last_value # Default to last_value
                    if i < len(rewards) - 1:
                        if i + 1 < len(values):
                             next_value = values[i+1]
                        else:
                             # print(f"[DEBUG GAE] Index {i+1} out of bounds for values array (shape: {values.shape}) when accessing next_value. Using last_value.")
                             pass # next_value remains last_value
                             
                    # Ensure next_value is scalar before calculation
                    if not isinstance(next_value, (float, np.float32, np.float64)):
                         # print(f"[DEBUG GAE] Warning: next_value is not a scalar ({type(next_value)}). Using 0.0.")
                         next_value = 0.0

                    delta = rewards[i] + self.gamma * next_value * (1 - float(is_terminals[i])) - current_value # Ensure is_terminals is float for math
                    last_gae_lam = delta + self.gamma * lambda_gae * (1 - float(is_terminals[i])) * last_gae_lam
                
                advantages.insert(0, last_gae_lam)
                returns.insert(0, last_gae_lam + current_value) # Return = Advantage + Value

            except IndexError as ie:
                 # print(f"[DEBUG GAE] IndexError during GAE calculation at step i={i}. values shape: {values.shape}, len(rewards): {len(rewards)}. Error: {ie}")
                 continue # Skip this problematic step
            except TypeError as te:
                 # print(f"[DEBUG GAE] TypeError during GAE calculation at step i={i}. values shape: {values.shape}, current_value type: {type(current_value)}, next_value type: {type(next_value)}. Error: {te}")
                 continue
            except Exception as e: # Catch other potential errors like the 0-dim index error
                 # if "0-dimensional" in str(e):
                 #      print(f"[DEBUG GAE] Caught potential 0-dim indexing error at step i={i}. values shape: {values.shape}. Trying to access values[{i}] or values[{i+1}]. Error: {e}")
                 continue # Skip this problematic step

        # Ensure lists are not empty before creating tensors if steps were skipped
        if not returns:
            returns = [0.0] # Avoid error with empty tensor creation
        if not advantages:
             advantages = [0.0]

        # NaN/Inf 확인 추가
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)

        # Check for NaNs/Infs after tensor creation
        if torch.isnan(returns_tensor).any() or torch.isinf(returns_tensor).any():
            # print(f"Warning: NaNs/Infs detected in returns tensor: {returns_tensor}")
            returns_tensor = torch.nan_to_num(returns_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(advantages_tensor).any() or torch.isinf(advantages_tensor).any():
            # print(f"Warning: NaNs/Infs detected in advantages tensor before normalization: {advantages_tensor}")
            advantages_tensor = torch.nan_to_num(advantages_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
        return returns_tensor, advantages_tensor

    def update(self, memory):
        if not memory.states: return 0.0 # Return 0 loss if memory is empty
        
        loss_val = 0.0 # Initialize loss value
        try:
            # NumPy 리스트 -> Torch 텐서 (GPU)
            old_states = torch.from_numpy(np.array(memory.states)).float().to(device)
            old_actions = torch.from_numpy(np.array(memory.actions)).float().to(device)
            old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(device)
            old_values_tensor = torch.tensor(memory.values, dtype=torch.float32).to(device).squeeze() # Squeeze if needed

            rewards = memory.rewards
            is_terminals = memory.is_terminals

            # old_values를 NumPy 배열로 변환하고 최소 1차원 배열인지 확인
            old_values_np = old_values_tensor.cpu().numpy()
            if old_values_np.ndim == 0:
                old_values_np = np.array([old_values_np]) # 스칼라 값을 1차원 배열로 변환
            
            # GAE 및 리턴 계산 (수정된 old_values_np 사용)
            returns, advantages = self.compute_returns_and_advantages(rewards, is_terminals, old_values_np)

            # --- 수정 시작: std() 경고 및 NaN 방지 --- 
            adv_mean = advantages.mean()
            adv_std = 0.0
            # 요소가 2개 이상일 때만 표준편차 계산
            if advantages.numel() > 1:
                 adv_std = advantages.std()
                 
            # 표준편차가 0에 가까우면 NaN 방지
            if adv_std > 1e-8: 
                 advantages = (advantages - adv_mean) / (adv_std + 1e-8) 
            else:
                 advantages = advantages - adv_mean # 평균만 빼줌 (표준편차가 0이거나 요소가 1개인 경우)
            # --- 수정 끝 --- 

            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                 print(f"Warning: NaNs/Infs detected in normalized advantages AFTER fix: {advantages}")
                 advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)


            for _ in range(self.k_epochs):
                logprobs, entropy, state_values = self.policy.evaluate(old_states, old_actions)
                
                # --- MSELoss 크기 일치 --- 
                state_values = state_values.view(-1)
                returns = returns.view(-1)
                # --- 수정 끝 ---

                # NaN/Inf 확인 (Shape 확인 후)
                if torch.isnan(logprobs).any() or torch.isinf(logprobs).any(): print("Warning: NaN/Inf in logprobs")
                if torch.isnan(entropy).any() or torch.isinf(entropy).any(): print("Warning: NaN/Inf in entropy")
                if torch.isnan(state_values).any() or torch.isinf(state_values).any(): print("Warning: NaN/Inf in state_values")

                ratios = torch.exp(logprobs - old_logprobs.detach())
                if torch.isnan(ratios).any() or torch.isinf(ratios).any(): print("Warning: NaN/Inf in ratios")

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values, returns) # 크기가 맞춰진 텐서 사용
                entropy_loss = entropy.mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss
                
                # 손실 값 확인
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Warning: NaN/Inf detected in loss. Actor: {actor_loss.item()}, Critic: {critic_loss.item()}, Entropy: {entropy_loss.item()}")
                    loss = torch.tensor(0.0, device=device, requires_grad=True) 

                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
            
            self.policy_old.load_state_dict(self.policy.state_dict())
            loss_val = loss.item() 

        except ValueError as ve:
             print(f"PPO 업데이트 중 ValueError: {ve}")
             # Check shapes before printing
             if 'old_states' in locals(): print(f"Shapes - old_states: {old_states.shape}")
             if 'old_actions' in locals(): print(f"Shapes - old_actions: {old_actions.shape}")
             if 'old_logprobs' in locals(): print(f"Shapes - old_logprobs: {old_logprobs.shape}")
             if 'old_values_tensor' in locals(): print(f"Shapes - old_values_tensor: {old_values_tensor.shape}")
             if 'returns' in locals(): print(f"Shapes - returns: {returns.shape}")
             if 'advantages' in locals(): print(f"Shapes - advantages: {advantages.shape}")
             loss_val = 0.0 
        except Exception as e:
            print(f"PPO 업데이트 오류: {e}") 
            loss_val = 0.0 

        return loss_val 

def integrated_gradients(model, state, baseline=None, steps=50):
    """
    통합 그래디언트 계산 
    (논문에서 제시한 DRL 에이전트 설명을 위한 방법)
    """
    if baseline is None:
        # 기준점으로 전체 0인 상태 사용
        baseline = np.zeros_like(state)
    
    # 보간 경로
    alphas = np.linspace(0, 1, steps)
    gradient_sum = np.zeros_like(state)
    
    for alpha in alphas:
        # 보간된 상태
        interpolated_state = baseline + alpha * (state - baseline)
        
        # 모델 출력에 대한 그래디언트 계산
        interpolated_tensor = torch.FloatTensor(interpolated_state.reshape(-1)).to(device)
        interpolated_tensor.requires_grad_(True)
        
        # 평탄화된 상태에 대해 모델 실행
        action_probs = model.policy.actor_critic_base(interpolated_tensor)
        
        # 그래디언트 계산 - 평균 액션 확률에 대해
        action_mean = torch.mean(action_probs)
        action_mean.backward()
        
        gradients = interpolated_tensor.grad.cpu().numpy()
        
        # 그래디언트 누적 - shape 맞춰주기
        gradient_sum += gradients.reshape(state.shape)
    
    # 통합 그래디언트 계산
    integrated_grads = (state.reshape(-1) - baseline.reshape(-1)) * (gradient_sum / steps).reshape(-1) # gradient_sum도 평균 계산 후 reshape
    
    # 원래 상태 형태로 재구성
    return integrated_grads.reshape(state.shape)

def linear_model_hindsight(features, returns):
    """
    사후 확인 선형 모델 구현 (참조 모델)
    """
    # 특성 형태 재구성
    X = features.reshape(features.shape[0], -1)
    y = returns
    
    # 선형 회귀 모델 훈련
    model = LinearRegression()
    model.fit(X, y)
    
    # 모델 계수 추출
    coefficients = model.coef_
    
    # 원래 형태로 재구성
    return coefficients.reshape(features.shape[1:])

def compute_macd(close_series, span_fast=12, span_slow=26):
    ema_fast = close_series.ewm(span=span_fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=span_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd

def compute_rsi(close_series, period=14):
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-8) # Prevent division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50) # Fill initial NaNs with neutral 50

def fetch_and_preprocess_data(start_date, end_date, tickers, save_path='data'):
    """
    주식 데이터 가져오기 및 전처리 함수 - PortfolioEnv 형식에 맞춤
    Output: data_array (n_steps, n_assets, 10), common_dates
    Features: [Open, High, Low, Close, Volume, MACD, RSI, MA14, MA21, MA100]
    """
    print(f"데이터 가져오기 및 전처리: {start_date}부터 {end_date}까지, {len(tickers)}개 종목")
    
    # 저장 디렉토리 생성
    os.makedirs(save_path, exist_ok=True)
    
    # 저장 파일 경로 (파일명에 티커 포함)
    tickers_str = "_".join(sorted(tickers))
    data_file = os.path.join(save_path, f'portfolio_data_{tickers_str}_{start_date}_{end_date}.pkl')
    
    # 이미 전처리된 데이터가 있는지 확인
    if os.path.exists(data_file):
        print("전처리된 데이터를 불러옵니다...")
        with open(data_file, 'rb') as f:
            data_array, common_dates = pickle.load(f)
        print(f"데이터 로드 완료. Shape: {data_array.shape}")
        return data_array, common_dates
    
    # 주식 데이터 가져오기 (병렬 처리 고려 가능)
    data_dict = {}
    error_count = 0
    print("yf.download 시작...")
    raw_data = yf.download(tickers, start=start_date, end=end_date)
    if raw_data.empty:
        print("경고: yf.download 결과가 비어있습니다.")
        return None, None
    print("yf.download 완료.")

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    processed_dfs = {}

    for ticker in tickers:
        try:
            # MultiIndex에서 종목별 데이터 추출
            stock_data = raw_data.loc[:, (slice(None), ticker)]
            stock_data.columns = stock_data.columns.droplevel(1) # Ticker 레벨 제거
            
            # 필요한 컬럼만 선택 및 결측치 처리 (ffill 후 bfill)
            stock_data = stock_data[required_columns].copy()
            if stock_data.isnull().values.any():
                 stock_data.ffill(inplace=True)
                 stock_data.bfill(inplace=True)
            
            # 모든 값이 NaN인 경우 건너뛰기
            if stock_data.isnull().values.all():
                 print(f"경고: {ticker} 데이터 전체가 NaN입니다. 건너니다.")
                 error_count += 1
                 continue
            
            # 기술 지표 계산
            stock_data['MACD'] = compute_macd(stock_data['Close'])
            stock_data['RSI'] = compute_rsi(stock_data['Close'])
            stock_data['MA14'] = stock_data['Close'].rolling(window=14).mean()
            stock_data['MA21'] = stock_data['Close'].rolling(window=21).mean()
            stock_data['MA100'] = stock_data['Close'].rolling(window=100).mean()
            
            # 초기 NaN 값 처리 (지표 계산 후 발생)
            stock_data.bfill(inplace=True) # 과거 값으로 채우기
            stock_data.ffill(inplace=True) # 그래도 남은 NaN은 미래 값으로
            stock_data.fillna(0, inplace=True) # 모든 NaN을 0으로 (최후의 수단)
            
            # 필요한 최종 피처 순서대로 정렬
            feature_order = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'RSI', 'MA14', 'MA21', 'MA100']
            processed_dfs[ticker] = stock_data[feature_order]
            
        except Exception as e:
            print(f"경고: {ticker} 처리 중 오류 발생 - {str(e)}")
            error_count += 1

    if not processed_dfs:
        print("오류: 처리된 데이터가 없습니다.")
        return None, None

    # 공통 날짜 찾기
    common_dates = set.intersection(*(set(df.index) for df in processed_dfs.values()))
    common_dates = sorted(list(common_dates))
    
    if not common_dates:
        print("오류: 모든 티커에 대한 공통 거래일이 없습니다.")
        return None, None
        
    # NumPy 배열로 변환 (n_steps, n_assets, n_features)
    asset_data = []
    valid_tickers = list(processed_dfs.keys())
    for ticker in valid_tickers:
        # 공통 날짜에 해당하는 데이터만 선택 후 float32로 변환
        df_aligned = processed_dfs[ticker].loc[common_dates].astype(np.float32)
        asset_data.append(df_aligned.values)
        
    data_array = np.stack(asset_data, axis=1)
    
    # 최종 데이터 확인 및 저장
    print(f"전처리 완료. 최종 데이터 Shape: {data_array.shape}")
    if np.isnan(data_array).any() or np.isinf(data_array).any():
         print("경고: 최종 데이터에 NaN 또는 Inf가 포함되어 있습니다. nan_to_num 처리합니다.")
         data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)

    if error_count > 0:
        print(f"경고: 처리 중 {error_count}개의 오류/경고가 발생했습니다.")

    print("전처리된 데이터를 저장합니다...")
    with open(data_file, 'wb') as f:
        pickle.dump((data_array, common_dates), f)
    
    return data_array, common_dates

def print_memory_stats():
    if torch.cuda.is_available():
        print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU 메모리 캐시: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

def train_ppo_agent(env, n_assets, n_features, max_episodes, max_timesteps, logger):
    """
    PPO 에이전트 학습 - PortfolioEnv 호환
    """
    print(f"PPO 에이전트 초기화 (자산: {n_assets}, 피처: {n_features})")
    
    current_time = int(time.time())
    np.random.seed(current_time)
    torch.manual_seed(current_time)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(current_time)
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    # --- 학습률 조정: PPO 에이전트 생성 시 lr 전달 --- 
    learning_rate = 1e-4 # 학습률 조정 (예: 3e-4 -> 1e-4)
    logger.info(f"Learning Rate: {learning_rate}")
    ppo_agent = PPO(n_assets, n_features, logger=logger, lr=learning_rate) 
    memory = Memory()
    
    episode_rewards = []
    training_start_time = time.time()
    total_steps = 0
    update_count = 0
    update_timestep = 2000 # 업데이트 간격을 늘려 더 많은 데이터 수집 (예: 200 -> 2000)
    
    logger.info(f"Update Timestep: {update_timestep}") # 로그 추가

    for episode in tqdm(range(max_episodes)):
        if episode % 20 == 0: torch.cuda.empty_cache(); gc.collect()
        
        state, _ = env.reset() # NumPy 배열 반환
        episode_reward = 0
        episode_start_time = time.time()
        current_episode_steps = 0 # 에피소드 내 스텝 수 추적
            
        # 에피소드 시작 로그는 10번에 한 번만 출력
        if episode % 10 == 0:
            logger.debug(f"=== 에피소드 {episode+1} 시작 ===")
        
        # --- 수정: max_timesteps 루프 제거, total_steps 기준으로 변경 --- 
        done = False
        truncated = False
        while not done and not truncated:
            total_steps += 1
            current_episode_steps += 1
            
            # 액션 선택 (NumPy 배열, 스칼라 값 반환)
            action, log_prob, value = ppo_agent.policy_old.act(state)
            
            # 환경 스텝 (NumPy 배열, 스칼라 값 반환)
            next_state, reward, done, truncated, info = env.step(action)
                    
            # 메모리에 저장 (NumPy, 스칼라)
            # done 또는 truncated 여부와 관계없이 저장
            memory.add_experience(state, action, log_prob, reward, done or truncated, value) # done or truncated를 is_terminal로 저장
                    
            state = next_state
            episode_reward += reward
                    
            # 일정 스텝마다 업데이트 (에피소드 종료와 무관하게)
            if total_steps % update_timestep == 0 and len(memory.states) > 0: # 메모리가 비어있지 않을 때만 업데이트
                update_loss = ppo_agent.update(memory)
                update_count += 1
                memory.clear_memory() # 업데이트 후 메모리 비우기
                if update_loss is not None:
                     logger.debug(f"업데이트 {update_count}, Loss: {update_loss:.4f}") 

            # 에피소드 최대 길이 도달 시 truncated = True 설정
            if current_episode_steps >= max_timesteps:
                 truncated = True 

        episode_time = time.time() - episode_start_time
        steps_per_second = current_episode_steps / episode_time if episode_time > 0 else 0
        episode_rewards.append(episode_reward)
        # --- 수정: 에피소드 종료 시 모델 저장 여부 확인 (필요시 조건 변경) ---
        # 현재는 매 에피소드 보상 기준으로 저장
        ppo_agent.save_model(episode, episode_reward)
            
        # 5번에 한 번만 성능 정보 출력
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-5:])
            # 에피소드 종료 시점의 info 사용 시 주의 (루프 마지막 info)
            final_portfolio_value = info.get('portfolio_value', env.portfolio_value) if 'info' in locals() else env.portfolio_value
            logger.info(f"\n에피소드 {episode+1}/{max_episodes}")
            logger.info(f"최종 보상: {episode_reward:.4f}")
            logger.info(f"최종 포트폴리오 가치: {final_portfolio_value:.2f}") # 소수점 2자리
            logger.info(f"최근 5개 평균 보상: {avg_reward:.4f}")
            logger.info(f"에피소드 스텝 수: {current_episode_steps}") # 실제 스텝 수 출력
            logger.info(f"에피소드 시간: {episode_time:.2f}초 ({steps_per_second:.1f} 스텝/초)")
            logger.info(f"총 스텝 수: {total_steps}")
            logger.info(f"총 업데이트 횟수: {update_count}")
            print_memory_stats()
                
    total_training_time = time.time() - training_start_time
    logger.info(f"\n총 학습 시간: {total_training_time:.2f}초")
    logger.info(f"총 스텝 수: {total_steps}")
    logger.info(f"스텝당 평균 시간: {total_training_time/total_steps if total_steps > 0 else 0:.4f}초")
    
    # --- 학습 종료 후 obs_rms를 에이전트에 저장 --- 
    if hasattr(env, 'obs_rms') and env.normalize_states:
        ppo_agent.obs_rms = env.obs_rms
        logger.info("학습 환경의 obs_rms를 에이전트에 저장했습니다.")
    
    return ppo_agent, episode_rewards
    
def evaluate_ppo_agent(env, ppo_agent, max_test_timesteps, load_best_model=True):
    """
    학습된 PPO 에이전트 평가 - PortfolioEnv 호환
    (환경의 상태 정규화 여부에 따라 입력 상태 처리)
    """
    if load_best_model: 
        loaded = ppo_agent.load_model() 
        if not loaded: 
            print("모델 로드 실패, 평가를 중단합니다.")
            return None # 모델 로드 실패 시 None 반환
    
    # 평가 시에는 환경 내 정규화를 사용하지 않음 (main에서 False로 설정)
    state, _ = env.reset() # 정규화 안된 상태 반환
    episode_reward = 0
    raw_episode_reward = 0 # 정규화 안된 보상 추적용
    portfolio_values = [env.initial_cash] # 초기 자본으로 시작
    returns = []
    weights = []
    actions = []
    terminated, truncated = False, False
    step_count = 0

    while not terminated and not truncated and step_count < max_test_timesteps:
        # --- 중요: 에이전트에 입력하기 전에 상태 정규화 (학습 시 사용한 RMS 사용) --- 
        normalized_state = state # 기본값
        # 에이전트에 저장된 obs_rms 사용 시도
        if hasattr(ppo_agent, 'obs_rms') and ppo_agent.obs_rms is not None: 
             # obs_rms 통계량이 유효한지 간단히 확인 (count > 0)
             if ppo_agent.obs_rms.count > 1e-4: 
                 normalized_state = np.clip((state - ppo_agent.obs_rms.mean) / np.sqrt(ppo_agent.obs_rms.var + 1e-8),
                                        -10.0, 10.0) # clip 값은 학습 시와 동일하게
             else:
                 print("Warning: 에이전트의 obs_rms 통계량이 유효하지 않아 정규화를 건너니다.")
        else:
             print("Warning: 에이전트에 obs_rms가 없어 상태 정규화 없이 진행합니다.")
        
        # 정규화된 상태를 텐서로 변환
        state_tensor = torch.from_numpy(normalized_state).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 액터-크리틱 네트워크 직접 사용 (정책 결정)
            concentration, _ = ppo_agent.policy.forward(state_tensor)
            dist = torch.distributions.Dirichlet(concentration)
            action = dist.mean.squeeze(0).cpu().numpy() # 평균 비중 사용
        
        actions.append(action)
        # 환경 스텝은 정규화 안된 상태 반환
        next_state, reward, terminated, truncated, info = env.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        returns.append(info['return'])
        weights.append(info['weights'])
        
        state = next_state
        # --- 수정: 정규화 안된 raw_reward 사용 --- 
        raw_reward = info.get('raw_reward', 0.0) # info에서 raw_reward 가져오기 (없으면 0)
        raw_episode_reward += raw_reward
        # episode_reward += reward # 정규화된 보상 대신 raw_reward 사용
        step_count += 1
        
        # 테스트 루프에서도 truncated 조건 확인 (max_test_timesteps 도달 시)
        if step_count >= max_test_timesteps:
            truncated = True
    
    # 결과 딕셔너리 반환 전 확인
    if not portfolio_values:
         print("Warning: No portfolio values recorded during evaluation.")
         portfolio_values = [env.initial_cash] # 초기값이라도 포함
         
    return {
        'episode_reward': raw_episode_reward, # 정규화 안된 보상 반환
        'portfolio_values': portfolio_values,
        'returns': returns,
        'weights': weights,
        'actions': actions
    }

def compute_feature_weights_drl(ppo_agent, states):
    """
    DRL 에이전트의 특성 가중치 계산
    (통합 그래디언트 사용)
    
    논문 식(15)에 따른 구현:
    M^π(t)_k = Σ^N_{i=1} IG(f^k(t))_i
             ≈ Σ^N_{i=1} f^k(t)_i · Σ^∞_{l=0} γ^l · ∂E[w^T(t+l)·y(t+l)|s^{k,i}(t),w(t)]/∂f^k(t)_i
    """
    feature_weights = []
    
    for state in tqdm(states):
        # 통합 그래디언트 계산
        ig = integrated_gradients(ppo_agent.policy, state)
        
        # 특성 가중치 추출 (기술 지표 부분만)
        feature_dim = state.shape[1] - state.shape[0]  # 공분산 행렬 제외
        feature_ig = ig[:, :feature_dim]
        
        # 각 특성에 대한 통합 그래디언트 합 (식 15에 해당)
        feature_weight = np.sum(feature_ig, axis=0)
        
        feature_weights.append(feature_weight)
    
    return np.array(feature_weights)

def compute_correlation(drl_weights, reference_weights):
    """
    DRL 에이전트와 참조 모델 간의 상관관계 계산
    """
    correlations = []
    
    for i in range(len(drl_weights)):
        # 선형 상관계수 (피어슨) 계산
        correlation = np.corrcoef(drl_weights[i], reference_weights[i])[0, 1]
        correlations.append(correlation)
    
    return np.array(correlations)

def plot_performance(portfolio_values, benchmark_values=None, title="Portfolio Performance"):
    """
    포트폴리오 성능 시각화
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label='PPO Portfolio')
    
    if benchmark_values is not None:
        plt.plot(benchmark_values, label='Benchmark')
    
    plt.title(title)
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_performance_metrics(portfolio_values, returns):
    """
    포트폴리오 성능 지표 계산
    """
    # 수익률이 없는 경우 기본값 반환
    if not returns:
        return {
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,  # 또는 np.nan
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0  # 또는 np.nan
        }
        
    daily_returns = np.array(returns)
    
    # NaN/Inf 값 처리 (필요시)
    if np.isnan(daily_returns).any() or np.isinf(daily_returns).any():
        print("Warning: NaN/Inf detected in daily returns. Replacing with 0.")
        daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
    # 연간 수익률
    annual_return = np.mean(daily_returns) * 252
    
    # 연간 변동성
    annual_volatility = np.std(daily_returns) * np.sqrt(252)
    
    # 샤프 비율 (무위험 이자율 0% 가정)
    # --- 수정: 0으로 나누기 방지 --- 
    if annual_volatility > 1e-8: 
        sharpe_ratio = annual_return / annual_volatility
    else:
        sharpe_ratio = 0.0 # 변동성이 0이면 샤프비율 0 또는 NaN으로 정의
        print("Warning: Annual volatility is near zero. Sharpe ratio set to 0.")
        
    # 최대 낙폭
    cumulative_returns = np.cumprod(1 + daily_returns)
    if not isinstance(cumulative_returns, np.ndarray):
        cumulative_returns = np.array(cumulative_returns) # Ensure it's an array
        
    max_drawdown = 0
    # 누적 수익률이 스칼라가 아닐 경우에만 peak 계산
    peak = cumulative_returns[0] if cumulative_returns.ndim > 0 and cumulative_returns.size > 0 else 1.0 
    
    for value in cumulative_returns:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak != 0 else 0 # peak가 0일 경우 방지
        max_drawdown = max(max_drawdown, drawdown)
    
    # 칼마 비율
    # --- 수정: 0으로 나누기 방지 --- 
    if max_drawdown > 1e-8:
        calmar_ratio = annual_return / max_drawdown
    else:
        calmar_ratio = 0.0 # 최대 낙폭이 0이면 칼마 비율 0 또는 NaN으로 정의
        print("Warning: Max drawdown is near zero. Calmar ratio set to 0.")
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio
    }
    
def main():
    current_time = int(time.time())
    np.random.seed(current_time)
    torch.manual_seed(current_time)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(current_time)
    
    logger = setup_logger()
    logger.info("\n===== GPU 상태 확인 =====")
    if torch.cuda.is_available():
        logger.info(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 메모리 사용량: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        logger.info(f"GPU 메모리 예약량: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            logger.info(f"GPU 사용률: {info.gpu}%")
            logger.info(f"GPU 메모리 사용률: {info.memory}%")
        except (ModuleNotFoundError, ImportError):
            logger.warning("GPU 사용률 정보를 확인할 수 없습니다. (pynvml 모듈 필요)")
    else:
        logger.warning("CUDA 사용 불가능 - CPU 모드로 실행됩니다.")
    
    train_start_date = '2008-01-02'
    train_end_date = '2020-12-31'
    test_start_date = '2021-01-01'
    test_end_date = '2024-12-31'
    initial_cash = 1e6 # 초기 자본 설정
    commission_rate = 0.005 # 수수료 설정
    
    print("\n데이터 준비 중...")
    # 데이터 로드 (PortfolioEnv 형식)
    data_array, common_dates = fetch_and_preprocess_data(train_start_date, test_end_date, STOCK_TICKERS)
    if data_array is None:
        print("데이터 로드 실패. 종료합니다.")
        return
    
    # 데이터 분할
    split_date = pd.Timestamp(test_start_date)
    # common_dates를 datetime 객체로 변환하여 비교
    common_dates_dt = [pd.to_datetime(d).tz_localize(None) for d in common_dates] # timezone 제거
    split_idx_arr = np.where(np.array(common_dates_dt) >= split_date)[0]
    if len(split_idx_arr) == 0:
        print(f"오류: 분할 날짜 {test_start_date} 이후 데이터가 없습니다.")
        return
    split_idx = split_idx_arr[0]
    
    train_data = data_array[:split_idx]
    test_data = data_array[split_idx:]
    print(f"훈련 데이터 Shape: {train_data.shape}, 테스트 데이터 Shape: {test_data.shape}")

    # 학습 환경 설정 (정규화 활성화)
    train_env = StockPortfolioEnv(train_data, initial_cash=initial_cash, commission_rate=commission_rate, normalize_states=True)
    n_assets = train_env.n_assets
    n_features = train_env.n_features
    
    max_episodes = 500 
    max_timesteps = train_env.max_episode_length 
    
    print(f"\n학습 시작: {max_episodes} 에피소드, 각 에피소드당 최대 {max_timesteps} 타임스텝")
    
    # PPO 에이전트 학습 (학습률 조정됨, obs_rms 저장 로직 추가됨)
    ppo_agent, training_rewards = train_ppo_agent(train_env, n_assets, n_features, max_episodes, max_timesteps, logger)
    logger.info("\n학습 완료!")
    print_memory_stats()
    
    # 테스트 환경 설정 (정규화 비활성화)
    test_env = StockPortfolioEnv(test_data, initial_cash=initial_cash, commission_rate=commission_rate, normalize_states=False)
    max_test_timesteps = len(test_env.data) - 1 
    
    print("\n테스트 시작...")
    # evaluate_ppo_agent는 이제 ppo_agent.obs_rms를 사용하여 정규화 시도
    test_results = evaluate_ppo_agent(test_env, ppo_agent, max_test_timesteps)
    
    # 모델 로드 실패 또는 다른 이유로 test_results가 None일 수 있음
    if test_results is None:
         print("테스트 실패. 성능 지표 계산을 건너니다.")
    else:
         print("테스트 완료!")
         print_memory_stats()
         
         # 성능 지표 계산
         metrics = calculate_performance_metrics(test_results['portfolio_values'], test_results['returns'])
         
         print("\n===== 포트폴리오 성능 지표 =====")
         print(f"연간 수익률: {metrics['annual_return']:.2%}")
         print(f"연간 변동성: {metrics['annual_volatility']:.2%}")
         print(f"샤프 비율: {metrics['sharpe_ratio']:.2f}")
         print(f"최대 낙폭: {metrics['max_drawdown']:.2%}")
         print(f"칼마 비율: {metrics['calmar_ratio']:.2f}")
         print(f"테스트 기간 최종 Raw Reward 합계: {test_results['episode_reward']:.4f}") # Raw reward 합계 출력
         
         # 성능 시각화
         plot_performance(test_results['portfolio_values'], title="PPO Portfolio Performance (Evaluation)")

if __name__ == "__main__":
    main()