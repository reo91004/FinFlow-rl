import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt  # 시각화를 위해 추가
import math

# device 설정 (CUDA 사용 가능하면 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================
# 1. 기술적 지표 계산 함수들 (pandas 사용)
# ==========================================
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
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ==========================================
# 2. yfinance를 통한 데이터 다운로드 및 전처리
# ==========================================
def fetch_and_compute_ta(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    df['MACD'] = compute_macd(df['Close'])
    df['RSI'] = compute_rsi(df['Close'])
    df['MA14'] = df['Close'].rolling(window=14).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()

    df.dropna(inplace=True)
    return df

def load_portfolio_data(tickers, start, end):
    data_dict = {}
    for ticker in tickers:
        df = fetch_and_compute_ta(ticker, start, end)
        data_dict[ticker] = df

    # 공통 날짜: 모든 자산에 데이터가 존재하는 날짜들의 교집합
    common_dates = set.intersection(*(set(df.index) for df in data_dict.values()))
    common_dates = sorted(common_dates)

    asset_data = []
    for ticker in tickers:
        df = data_dict[ticker].loc[common_dates]
        # 피처 순서: [Open, High, Low, Close, Volume, MACD, RSI, MA14, MA21, MA100]
        asset_data.append(df.values)
    # data_array shape: (n_steps, n_assets, 10)
    data_array = np.stack(asset_data, axis=1)
    return data_array, common_dates

# ==========================================
# 3. PortfolioEnv (Gym 환경) 정의
# ==========================================
class PortfolioEnv(gym.Env):
    """
    실제 주가 및 기술적 지표를 반영한 포트폴리오 거래 환경.
    관측: 각 자산의 10개 피처 (OHLCV, MACD, RSI, MA14, MA21, MA100)
    행동: 각 자산의 투자 비중 (0 ~ 1, 총합 1)
    보상: 하루 단위 로그 수익률 (ln(새 포트폴리오 가치 / 이전 포트폴리오 가치))
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, data: np.ndarray, initial_cash=1e6, commission_rate=0.005):
        super(PortfolioEnv, self).__init__()
        self.data = data
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate

        self.n_steps = data.shape[0]
        self.n_assets = data.shape[1]
        self.n_features = data.shape[2]

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets, self.n_features),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None, start_index=None):
        super().reset(seed=seed)
        # 에피소드 길이 제한(예: 최대 200일) 내 무작위 시작
        if start_index is None:
            if self.n_steps > 200:
                start_index = np.random.randint(0, self.n_steps - 200)
            else:
                start_index = 0
        self.current_step = start_index
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = self.cash
        return self._get_observation(), {}

    def _get_observation(self):
        return self.data[self.current_step]

    def step(self, action):
        # 행동은 0~1 사이 값을 갖도록 클리핑 및 정규화
        action = np.clip(action, 0, 1)
        if action.sum() > 0:
            action = action / action.sum()
        else:
            action = np.ones_like(action) / len(action)

        obs = self.data[self.current_step]
        current_prices = obs[:, 3]  # 종가 사용

        # 이전 포트폴리오 가치 계산
        prev_portfolio_value = self.cash + np.dot(self.holdings, current_prices)

        # 목표 배분: 현재 포트폴리오 가치에 대한 자산별 투자액
        target_allocation = action * prev_portfolio_value
        current_allocation = self.holdings * current_prices
        trade_amount = target_allocation - current_allocation
        shares_to_trade = trade_amount / current_prices

        # 거래 및 거래 수수료 반영
        for i in range(self.n_assets):
            if shares_to_trade[i] > 0:  # 매수
                cost = shares_to_trade[i] * current_prices[i]
                commission = cost * self.commission_rate
                total_cost = cost + commission
                if total_cost <= self.cash:
                    self.holdings[i] += shares_to_trade[i]
                    self.cash -= total_cost
                else:
                    affordable_shares = self.cash / (current_prices[i] * (1 + self.commission_rate))
                    cost = affordable_shares * current_prices[i]
                    commission = cost * self.commission_rate
                    total_cost = cost + commission
                    self.holdings[i] += affordable_shares
                    self.cash -= total_cost
            elif shares_to_trade[i] < 0:  # 매도
                shares_to_sell = min(abs(shares_to_trade[i]), self.holdings[i])
                revenue = shares_to_sell * current_prices[i]
                commission = revenue * self.commission_rate
                total_revenue = revenue - commission
                self.holdings[i] -= shares_to_sell
                self.cash += total_revenue

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        next_obs = self.data[self.current_step]
        new_prices = next_obs[:, 3]
        new_portfolio_value = self.cash + np.dot(self.holdings, new_prices)

        # 보상을 로그 수익률로 계산 (논문의 식 (5))
        reward = np.log(new_portfolio_value / (prev_portfolio_value + 1e-8) + 1e-8)

        self.portfolio_value = new_portfolio_value

        info = {"portfolio_value": self.portfolio_value, "cash": self.cash, "holdings": self.holdings.copy()}
        return next_obs, reward, done, False, info

    def render(self, mode="human"):
        obs = self.data[self.current_step]
        current_prices = obs[:, 3]
        print(f"Step: {self.current_step}")
        print(f"Cash: {self.cash}")
        print(f"Holdings: {self.holdings}")
        print(f"Current Prices (Close): {current_prices}")
        print(f"Portfolio Value: {self.portfolio_value}")

    def close(self):
        pass

# ==========================================
# 4. Actor-Critic 네트워크 (PPO용, 논문 로직 기반)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, n_assets, n_features, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.input_dim = n_assets * n_features
        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 액터 헤드는 Dirichlet 분포의 concentration 파라미터를 출력하도록 softplus 사용
        self.actor_head = nn.Linear(hidden_size, n_assets)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, n_assets, n_features) → flatten
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # concentration 계산 시 NaN 방지를 위해 클리핑 추가
        concentration = torch.clamp(F.softplus(self.actor_head(x)), min=1e-3, max=1e3) + 1e-3
        value = self.critic_head(x)
        return concentration, value

    def act(self, state):
        concentration, value = self.forward(state)
        # Dirichlet 분포 생성 후 샘플링 (포트폴리오 비중은 확률분포)
        dist = torch.distributions.Dirichlet(concentration)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, state, action):
        concentration, value = self.forward(state)
        dist = torch.distributions.Dirichlet(concentration)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value

# ==========================================
# 5. PPO 에이전트 구현
# ==========================================
class PPOAgent:
    def __init__(self, n_assets, n_features, hidden_size=128, lr=3e-4, gamma=0.99,
                 clip_epsilon=0.2, update_epochs=10):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs

        self.policy = ActorCritic(n_assets, n_features, hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        return returns

    def update(self, trajectories):
        # trajectories: list of (state, action, log_prob, reward, done, value)
        states = torch.stack([t[0] for t in trajectories]).to(device)
        actions = torch.stack([t[1] for t in trajectories]).to(device)
        old_log_probs = torch.stack([t[2] for t in trajectories]).to(device)
        rewards = [t[3] for t in trajectories]
        dones = [t[4] for t in trajectories]
        values = torch.stack([t[5] for t in trajectories]).to(device)

        with torch.no_grad():
            next_value = values[-1]
        returns = self.compute_returns(rewards, dones, values, next_value)
        returns = returns.detach()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            log_probs, entropy, state_values = self.policy.evaluate(states, actions)
            ratio = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# ==========================================
# 6. Integrated Gradients 함수 (설명용)
# ==========================================
def integrated_gradients(model, state, baseline, target_index=0, steps=50):
    """
    model: 평가할 모델 (정책 네트워크)
    state: 입력 상태 (torch tensor)
    baseline: 기준 입력 (보통 0으로 설정)
    target_index: 기울기를 계산할 출력 인덱스 (예, 특정 자산의 투자 비중)
    steps: 적분 구간 분할 수
    """
    scaled_inputs = [baseline + (float(i)/steps) * (state - baseline) for i in range(steps+1)]
    grads = []
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.unsqueeze(0).to(device)
        scaled_input.requires_grad = True
        concentration, _ = model.forward(scaled_input)
        # target_index에 해당하는 출력 값을 모두 합산 후 미분
        output = concentration[:, target_index].sum()
        model.zero_grad()
        output.backward(retain_graph=True)
        grad = scaled_input.grad.detach()
        grads.append(grad.squeeze(0))
    avg_grads = torch.stack(grads).mean(dim=0)
    integrated_grad = (state - baseline) * avg_grads
    return integrated_grad

# ==========================================
# 7. 하이퍼파라미터 및 데이터 분할
# ==========================================
gamma_value = 0.99
clip_epsilon = 0.2
actor_lr = 3e-4
total_episodes = 1000    # 1000 에피소드
max_episode_length = 200 # 각 에피소드 최대 200 단계
update_timestep = 200    # 일정 타임스텝마다 정책 업데이트

# 데이터 기간: 2008-01-02 ~ 2024-12-31 (S&P500 예시 종목 10개)
start_date = "2008-01-02"
end_date = "2024-12-31"
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "JPM", "JNJ", "PG", "V"]

data_array, common_dates = load_portfolio_data(tickers, start_date, end_date)
print("전체 데이터 shape:", data_array.shape)

# 분할 기준: 2021-01-01 이전은 훈련, 이후는 테스트 (백테스팅)
split_date = pd.Timestamp("2021-01-01")
train_end_idx = sum(1 for d in common_dates if pd.Timestamp(d) < split_date)
print("훈련 데이터 길이:", train_end_idx)
print("테스트 데이터 길이:", data_array.shape[0] - train_end_idx)

train_data = data_array[:train_end_idx]
test_data = data_array[train_end_idx:]

# ==========================================
# 8. 환경 및 PPO 에이전트 초기화 (훈련용)
# ==========================================
env = PortfolioEnv(train_data)
n_assets = env.n_assets
n_features = env.n_features

agent = PPOAgent(n_assets, n_features, hidden_size=128, lr=actor_lr, gamma=gamma_value,
                 clip_epsilon=clip_epsilon, update_epochs=10)

# ==========================================
# 9. 훈련 루프 (온-폴리시 방식, 에피소드별 트랜지션 수집 후 업데이트)
# ==========================================
episode_rewards = []
total_steps = 0

for episode in range(total_episodes):
    # 무작위 시작 (에피소드 길이 제한 내)
    start_index = np.random.randint(0, env.n_steps - max_episode_length)
    state, _ = env.reset(start_index=start_index)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    done = False
    episode_reward = 0
    step_count = 0
    trajectories = []
    while not done and step_count < max_episode_length:
        action, log_prob, value = agent.policy.act(state_tensor)
        action_np = action.squeeze(0).cpu().numpy()
        next_state, reward, done, _, info = env.step(action_np)
        episode_reward += reward

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        trajectories.append((state_tensor.squeeze(0), action.squeeze(0), log_prob.squeeze(0), reward, float(done), value.squeeze(0)))
        total_steps += 1
        step_count += 1
        state_tensor = next_state_tensor

        # 일정 타임스텝마다 PPO 업데이트 (온-폴리시)
        if total_steps % update_timestep == 0:
            agent.update(trajectories)
            trajectories = []

    # 에피소드 종료 후 남은 트랜지션 업데이트
    if len(trajectories) > 0:
        agent.update(trajectories)
    episode_rewards.append(episode_reward)
    if (episode + 1) % 50 == 0:
        print(f"훈련 Episode {episode+1}/{total_episodes}, Episode Reward: {episode_reward:.4f}, Portfolio Value: {info['portfolio_value']:.2f}")

print("훈련 완료!")

# ==========================================
# 10. 백테스팅 (테스트 데이터 사용 및 시각화)
# ==========================================
print("\n=== 백테스팅 시작 ===")
test_env = PortfolioEnv(test_data)
state, _ = test_env.reset(start_index=0)
portfolio_values = [test_env.portfolio_value]
done = False
while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        concentration, _ = agent.policy.forward(state_tensor)
        dist = torch.distributions.Dirichlet(concentration)
        action = dist.mean  # 결정론적 행동: 평균 사용
    action_np = action.squeeze(0).cpu().numpy()
    state, reward, done, _, info = test_env.step(action_np)
    portfolio_values.append(info["portfolio_value"])
print("백테스팅 최종 포트폴리오 가치: {:.2f}".format(info["portfolio_value"]))

final_value = portfolio_values[-1]
initial_value = portfolio_values[0]
total_return_percentage = (final_value - initial_value) / initial_value * 100
print("총 수익률: {:.2f}%".format(total_return_percentage))

# ==========================================
# 11. Integrated Gradients 계산 (설명 분석 예시)
# ==========================================
# 예를 들어, 테스트 초기 상태에 대해 IG를 계산한다.
sample_state, _ = test_env.reset(start_index=0)
sample_state_tensor = torch.tensor(sample_state, dtype=torch.float32).to(device)
baseline = torch.zeros_like(sample_state_tensor).to(device)
ig = integrated_gradients(agent.policy, sample_state_tensor.unsqueeze(0), baseline.unsqueeze(0), target_index=0, steps=50)
print("샘플 상태에 대한 Integrated Gradients 계산 완료.")

# ==========================================
# 12. 포트폴리오 가치 변화 시각화
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values, label="Portfolio Value")
plt.title("백테스팅 기간 포트폴리오 가치 변화")
plt.xlabel("시간 (스텝)")
plt.ylabel("포트폴리오 가치")
plt.legend()
plt.grid(True)
plt.show()