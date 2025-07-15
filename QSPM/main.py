import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings("ignore")

# 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 디렉토리 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


class QuantumPortfolioAgent:
    def __init__(self, n_assets, n_strategies=3, learning_rate=0.01, random_state=None):
        """
        Quantum-Inspired Superposition Portfolio Management

        Parameters:
        - n_assets: 자산 개수
        - n_strategies: 중첩할 전략 개수 (conservative, balanced, aggressive)
        - learning_rate: 학습률
        - random_state: 랜덤 시드
        """
        self.n_assets = n_assets
        self.n_strategies = n_strategies
        self.learning_rate = learning_rate
        
        # 랜덤 시드 설정
        if random_state is not None:
            np.random.seed(random_state)

        # 양자 상태 초기화: |ψ⟩ = α|conservative⟩ + β|balanced⟩ + γ|aggressive⟩
        self.amplitudes = np.ones(n_strategies) / np.sqrt(n_strategies)  # 정규화된 진폭

        # 각 전략의 포트폴리오 가중치
        self.strategy_weights = {
            0: np.ones(n_assets) / n_assets * 0.3,  # Conservative: 낮은 가중치
            1: np.ones(n_assets) / n_assets * 0.6,  # Balanced: 중간 가중치
            2: np.ones(n_assets) / n_assets * 1.0,  # Aggressive: 높은 가중치
        }

        # 양자 얽힘을 위한 상관관계 매트릭스 (랜덤 초기화)
        self.entanglement_matrix = np.random.rand(n_assets, n_assets)
        self.entanglement_matrix = (
            self.entanglement_matrix + self.entanglement_matrix.T
        ) / 2
        np.fill_diagonal(self.entanglement_matrix, 1.0)

        # 메모리
        self.state_history = []
        self.reward_history = []

    def quantum_measurement(self, market_volatility):
        """
        양자 측정: 중첩 상태에서 하나의 전략으로 붕괴
        시장 변동성에 따라 측정 확률 조정
        """
        # 시장 변동성이 높을수록 conservative 전략 선호
        volatility_bias = np.array([market_volatility, 0.5, 1.0 - market_volatility])

        # 양자 확률 계산: |amplitude|² × volatility_bias
        probabilities = (self.amplitudes**2) * volatility_bias
        probabilities = probabilities / np.sum(probabilities)  # 정규화

        # 확률적 전략 선택 (측정에 의한 붕괴)
        chosen_strategy = np.random.choice(self.n_strategies, p=probabilities)

        return chosen_strategy, probabilities

    def quantum_entanglement_adjustment(self, returns):
        """
        양자 얽힘을 모방한 자산 간 상관관계 기반 가중치 조정 (강화된 NaN 처리)
        """
        try:
            # 입력 검증
            if returns.size == 0 or np.isnan(returns).all():
                return np.ones(self.n_assets) / self.n_assets
            
            # NaN 값 처리
            returns_clean = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 상관관계 계산 (안전한 방식)
            if returns_clean.shape[0] <= 1:
                correlation = np.eye(self.n_assets)
            else:
                try:
                    correlation = np.corrcoef(returns_clean.T)
                    # 결과 검증
                    if np.isnan(correlation).any() or np.isinf(correlation).any():
                        correlation = np.eye(self.n_assets)
                except:
                    correlation = np.eye(self.n_assets)

            # 얽힘 강도 업데이트 (지수 이동 평균)
            alpha = 0.1
            correlation_abs = np.abs(correlation)
            
            # NaN 및 무한대 값 처리
            correlation_abs = np.nan_to_num(correlation_abs, nan=0.0, posinf=1.0, neginf=0.0)
            
            self.entanglement_matrix = (
                1 - alpha
            ) * self.entanglement_matrix + alpha * correlation_abs

            # 얽힘 기반 가중치 조정
            entanglement_weights = np.mean(self.entanglement_matrix, axis=1)
            
            # 가중치 정규화 (0으로 나누기 방지)
            weight_sum = np.sum(entanglement_weights)
            if weight_sum > 1e-8:
                entanglement_weights = entanglement_weights / weight_sum
            else:
                entanglement_weights = np.ones(self.n_assets) / self.n_assets

            # 최종 검증
            entanglement_weights = np.nan_to_num(entanglement_weights, nan=1.0/self.n_assets, posinf=1.0/self.n_assets, neginf=0.0)
            
            return entanglement_weights
            
        except Exception as e:
            print(f"양자 얽힘 계산 오류: {e}, 균등 가중치 반환")
            return np.ones(self.n_assets) / self.n_assets

    def get_portfolio_weights(self, market_data, lookback=20):
        """
        포트폴리오 가중치 결정 (강화된 NaN 처리)
        """
        if len(market_data) < lookback:
            return np.ones(self.n_assets) / self.n_assets

        try:
            # 최근 수익률과 변동성 계산 (안전한 방식)
            recent_data = market_data.iloc[-lookback:]
            recent_returns = recent_data.pct_change().dropna()
            
            # 수익률 데이터 검증
            if len(recent_returns) == 0 or recent_returns.isnull().all().all():
                return np.ones(self.n_assets) / self.n_assets
            
            # NaN 값 처리
            recent_returns_clean = recent_returns.fillna(0)
            
            # 변동성 계산 (안전한 방식)
            try:
                volatilities = np.std(recent_returns_clean, axis=0)
                market_volatility = np.mean(volatilities)
                
                # NaN 및 무한대 검증
                if np.isnan(market_volatility) or np.isinf(market_volatility):
                    market_volatility = 0.1  # 기본값
                    
            except:
                market_volatility = 0.1  # 기본값

            # 양자 측정으로 전략 선택
            chosen_strategy, strategy_probs = self.quantum_measurement(market_volatility)

            # 선택된 전략의 기본 가중치
            base_weights = self.strategy_weights[chosen_strategy].copy()

            # 양자 얽힘 조정
            entanglement_weights = self.quantum_entanglement_adjustment(
                recent_returns_clean.values
            )

            # 최종 가중치: 기본 전략 + 얽힘 조정
            final_weights = 0.7 * base_weights + 0.3 * entanglement_weights
            
            # 가중치 정규화 (0으로 나누기 방지)
            weight_sum = np.sum(final_weights)
            if weight_sum > 1e-8:
                final_weights = final_weights / weight_sum
            else:
                final_weights = np.ones(self.n_assets) / self.n_assets

            # 최종 검증
            final_weights = np.nan_to_num(final_weights, nan=1.0/self.n_assets, posinf=1.0/self.n_assets, neginf=0.0)

            # 메모리에 저장
            self.state_history.append(
                {
                    "strategy": chosen_strategy,
                    "strategy_probs": strategy_probs.copy(),
                    "volatility": market_volatility,
                    "weights": final_weights.copy(),
                }
            )

            return final_weights
            
        except Exception as e:
            print(f"포트폴리오 가중치 계산 오류: {e}, 균등 가중치 반환")
            return np.ones(self.n_assets) / self.n_assets

    def update_amplitudes(self, reward):
        """
        보상 기반 양자 진폭 업데이트
        좋은 성과를 낸 전략의 진폭을 강화
        """
        if len(self.state_history) == 0:
            return

        last_strategy = self.state_history[-1]["strategy"]

        # 보상 기반 진폭 조정
        amplitude_update = np.zeros(self.n_strategies)
        amplitude_update[last_strategy] = self.learning_rate * reward

        # 진폭 업데이트
        self.amplitudes += amplitude_update

        # 정규화 (양자 상태의 확률 보존)
        self.amplitudes = self.amplitudes / np.linalg.norm(self.amplitudes)

        self.reward_history.append(reward)


class PortfolioBacktester:
    def __init__(self, symbols, train_start, train_end, test_start, test_end):
        self.symbols = symbols
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

        # 데이터 파일 경로
        data_filename = f"market_data_{'_'.join(symbols)}_{train_start}_{test_end}.pkl"
        self.data_path = os.path.join(DATA_DIR, data_filename)

        # 데이터 로드 또는 다운로드
        if os.path.exists(self.data_path):
            print(f"기존 데이터 로드 중: {data_filename}")
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            print("데이터 다운로드 중...")
            raw_data = yf.download(symbols, start="2007-12-01", end="2025-01-01", progress=True)
            
            # 데이터 구조 확인 및 적절한 컬럼 선택
            if len(symbols) == 1:
                # 단일 티커인 경우
                if "Adj Close" in raw_data.columns:
                    self.data = raw_data["Adj Close"].to_frame(symbols[0])
                elif "Close" in raw_data.columns:
                    self.data = raw_data["Close"].to_frame(symbols[0])
                    print("주의: 'Adj Close' 없음, 'Close' 사용")
                else:
                    raise ValueError("가격 데이터를 찾을 수 없습니다.")
            else:
                # 다중 티커인 경우
                try:
                    # Adj Close 시도
                    self.data = raw_data["Adj Close"]
                except KeyError:
                    try:
                        # Close 시도
                        self.data = raw_data["Close"]
                        print("주의: 'Adj Close' 없음, 'Close' 사용")
                    except KeyError:
                        # MultiIndex 구조인 경우 개별 처리
                        price_data = {}
                        for symbol in symbols:
                            if ("Adj Close", symbol) in raw_data.columns:
                                price_data[symbol] = raw_data[("Adj Close", symbol)]
                            elif ("Close", symbol) in raw_data.columns:
                                price_data[symbol] = raw_data[("Close", symbol)]
                                print(f"주의: {symbol} 'Adj Close' 없음, 'Close' 사용")
                            else:
                                print(f"경고: {symbol} 가격 데이터를 찾을 수 없습니다.")
                                continue
                        
                        if not price_data:
                            raise ValueError("사용 가능한 가격 데이터가 없습니다.")
                        
                        self.data = pd.DataFrame(price_data)
            
            # 강화된 결측값 처리
            print("데이터 전처리 중...")
            
            # 1차 처리: Forward Fill → Backward Fill
            if self.data.isnull().values.any():
                print("결측값 발견, 전방향/후방향 채우기 적용")
                self.data = self.data.fillna(method="ffill").fillna(method="bfill")
            
            # 2차 처리: 여전히 NaN이 남아있으면 0으로 채움
            if self.data.isnull().values.any():
                print("잔여 결측값을 0으로 채움")
                self.data = self.data.fillna(0)
            
            # 3차 처리: 무한대 값 처리
            if np.isinf(self.data.values).any():
                print("무한대 값 발견, 유한값으로 변환")
                self.data = self.data.replace([np.inf, -np.inf], 0)
            
            # 최종 검증
            if self.data.isnull().values.any() or np.isinf(self.data.values).any():
                print("최종 데이터 정리 중...")
                self.data = pd.DataFrame(
                    np.nan_to_num(self.data.values, nan=0.0, posinf=0.0, neginf=0.0),
                    index=self.data.index,
                    columns=self.data.columns
                )
            
            # 데이터 저장
            with open(self.data_path, "wb") as f:
                pickle.dump(self.data, f)
            print(f"데이터 저장 완료: {data_filename}")

        # 훈련/테스트 데이터 분할
        self.train_data = self.data[train_start:train_end]
        self.test_data = self.data[test_start:test_end]

    def calculate_metrics(self, returns, initial_capital=1e6):
        """성과 지표 계산"""
        # 누적 수익률 계산
        cum_returns = (1 + returns).cumprod()
        final_value = initial_capital * cum_returns.iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # 연간화된 지표들
        volatility = returns.std() * np.sqrt(252)
        max_drawdown = self.calculate_max_drawdown(returns)
        
        # Sharpe Ratio (무위험 수익률 0 가정)
        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        )
        
        # Calmar Ratio
        calmar_ratio = (
            returns.mean() * 252 / abs(max_drawdown) if max_drawdown != 0 else 0
        )

        return {
            "Total Return": total_return,
            "Volatility": volatility,
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe_ratio,
            "Calmar Ratio": calmar_ratio,
            "Final Value": final_value,
            "Initial Capital": initial_capital,
        }

    def calculate_max_drawdown(self, returns):
        """최대 낙폭 계산"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def backtest_single_run(self, seed=None, return_model=False):
        """단일 백테스트 실행"""
        if seed is not None:
            np.random.seed(seed)

        agent = QuantumPortfolioAgent(n_assets=len(self.symbols), random_state=seed)

        # 훈련 단계
        train_returns = self.train_data.pct_change().dropna()
        portfolio_values = [1.0]

        for i in range(len(train_returns)):
            # 현재까지의 데이터로 포트폴리오 가중치 결정
            current_data = self.train_data.iloc[: i + 1]
            weights = agent.get_portfolio_weights(current_data)

            # 포트폴리오 수익률 계산
            portfolio_return = np.sum(weights * train_returns.iloc[i])
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

            # 보상 계산 (위험 조정 수익률)
            if len(portfolio_values) > 20:
                recent_returns = (
                    np.diff(portfolio_values[-21:]) / portfolio_values[-21:-1]
                )
                reward = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)
            else:
                reward = portfolio_return

            # 에이전트 업데이트
            agent.update_amplitudes(reward)

        # 테스트 단계
        test_returns = self.test_data.pct_change().dropna()
        test_portfolio_returns = []

        for i in range(len(test_returns)):
            # 훈련된 에이전트로 포트폴리오 가중치 결정
            current_data = self.test_data.iloc[: i + 1]
            weights = agent.get_portfolio_weights(current_data)

            # 포트폴리오 수익률 계산
            portfolio_return = np.sum(weights * test_returns.iloc[i])
            test_portfolio_returns.append(portfolio_return)

        if return_model:
            return pd.Series(test_portfolio_returns, index=test_returns.index), agent
        else:
            return pd.Series(test_portfolio_returns, index=test_returns.index)

    def save_model(self, agent, filename=None):
        """양자 에이전트 모델 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quantum_agent_{timestamp}.pkl"
        
        model_path = os.path.join(MODELS_DIR, filename)
        with open(model_path, "wb") as f:
            pickle.dump(agent, f)
        print(f"모델 저장 완료: {filename}")
        return model_path

    def load_model(self, filename):
        """양자 에이전트 모델 로드"""
        model_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                agent = pickle.load(f)
            print(f"모델 로드 완료: {filename}")
            return agent
        else:
            print(f"모델 파일을 찾을 수 없습니다: {filename}")
            return None

    def save_results(self, metrics_df, quantum_states=None, filename=None):
        """결과 저장 및 시각화"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"qspm_results_{timestamp}"
        
        # CSV 저장
        csv_path = os.path.join(RESULTS_DIR, f"{filename}.csv")
        metrics_df.to_csv(csv_path, index=False)
        
        # 시각화
        plt.figure(figsize=(16, 12))
        
        # 서브플롯 1: 성과 지표 박스플롯
        plt.subplot(3, 3, 1)
        metrics_df.boxplot(column=["Total Return"], ax=plt.gca())
        plt.title("Total Return Distribution")
        
        plt.subplot(3, 3, 2)
        metrics_df.boxplot(column=["Sharpe Ratio"], ax=plt.gca())
        plt.title("Sharpe Ratio Distribution")
        
        plt.subplot(3, 3, 3)
        metrics_df.boxplot(column=["Max Drawdown"], ax=plt.gca())
        plt.title("Max Drawdown Distribution")
        
        # 서브플롯 2: 양자 상태 시각화 (있는 경우)
        if quantum_states is not None:
            plt.subplot(3, 3, 4)
            # 전략별 확률 변화
            for i, strategy in enumerate(["Conservative", "Balanced", "Aggressive"]):
                strategy_probs = [state["strategy_probs"][i] for state in quantum_states if "strategy_probs" in state]
                if strategy_probs:
                    plt.plot(strategy_probs[:100], label=strategy)  # 처음 100개만 표시
            plt.title("Quantum Strategy Probabilities")
            plt.legend()
            
            plt.subplot(3, 3, 5)
            # 시장 변동성 변화
            volatilities = [state["volatility"] for state in quantum_states if "volatility" in state]
            if volatilities:
                plt.plot(volatilities[:100])
            plt.title("Market Volatility Over Time")
        
        # 서브플롯 3: 상관관계 히트맵
        plt.subplot(3, 3, 6)
        correlation = metrics_df.corr()
        plt.imshow(correlation, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.title("Metrics Correlation")
        plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
        plt.yticks(range(len(correlation.columns)), correlation.columns)
        
        # 서브플롯 4: 수익률 vs 비율 산점도
        plt.subplot(3, 3, 7)
        plt.scatter(metrics_df["Volatility"], metrics_df["Total Return"], alpha=0.6)
        plt.xlabel("Volatility")
        plt.ylabel("Total Return")
        plt.title("Risk-Return Scatter")
        
        # 서브플롯 5: 성과 요약
        plt.subplot(3, 3, 8)
        plt.axis("off")
        summary_text = f"""
        QSPM 백테스트 결과 요약
        
        총 수익률: {metrics_df['Total Return'].mean():.2%}
        표준편차: {metrics_df['Volatility'].mean():.3f}
        최대 낙폭: {metrics_df['Max Drawdown'].mean():.2%}
        Sharpe Ratio: {metrics_df['Sharpe Ratio'].mean():.2f}
        Calmar Ratio: {metrics_df['Calmar Ratio'].mean():.2f}
        초기 자본: {metrics_df['Initial Capital'].iloc[0]:,.0f}원
        최종 자본: {metrics_df['Final Value'].mean():,.0f}원
        """
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment="center")
        
        # 서브플롯 6: 히스토그램
        plt.subplot(3, 3, 9)
        plt.hist(metrics_df["Sharpe Ratio"], bins=10, alpha=0.7, edgecolor="black")
        plt.title("Sharpe Ratio Distribution")
        plt.xlabel("Sharpe Ratio")
        plt.ylabel("Frequency")
        
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f"{filename}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"결과 저장 완료: {csv_path}, {plot_path}")
        return csv_path, plot_path

    def run_multiple_backtests(self, n_runs=10, save_results=True):
        """다중 백테스트 실행"""
        all_metrics = []
        all_quantum_states = []
        best_agent = None
        best_sharpe = -np.inf

        print(f"QSPM 백테스트 {n_runs}회 실행 중...")

        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")

            # 각 실행마다 다른 시드 사용
            portfolio_returns, agent = self.backtest_single_run(seed=run, return_model=True)
            metrics = self.calculate_metrics(portfolio_returns)
            all_metrics.append(metrics)
            all_quantum_states.extend(agent.state_history)
            
            # 최고 성과 모델 저장
            if metrics["Sharpe Ratio"] > best_sharpe:
                best_sharpe = metrics["Sharpe Ratio"]
                best_agent = agent

        # 평균과 표준편차 계산
        metrics_df = pd.DataFrame(all_metrics)

        print("\n=== QSPM 백테스트 결과 ===")
        print(f"총 수익률: {metrics_df['Total Return'].mean():.2%}")
        print(f"표준편차: {metrics_df['Volatility'].mean():.3f}")
        print(f"최대 낙폭: {metrics_df['Max Drawdown'].mean():.2%}")
        print(f"Sharpe Ratio: {metrics_df['Sharpe Ratio'].mean():.2f}")
        print(f"Calmar Ratio: {metrics_df['Calmar Ratio'].mean():.2f}")
        print(f"초기 자본: {metrics_df['Initial Capital'].iloc[0]:,.0f}원")
        print(f"최종 자본: {metrics_df['Final Value'].mean():,.0f}원")

        # 결과 저장
        if save_results:
            self.save_results(metrics_df, all_quantum_states)
            if best_agent is not None:
                self.save_model(best_agent, "best_quantum_agent.pkl")

        return metrics_df


# 실행
if __name__ == "__main__":
    # 설정
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "JPM", "JNJ", "PG", "V"]
    train_start = "2008-01-02"
    train_end = "2020-12-31"
    test_start = "2021-01-01"
    test_end = "2024-12-31"

    # 전역 시드 초기화 (매번 다른 결과를 위해)
    import time
    global_seed = int(time.time()) % 10000
    np.random.seed(global_seed)
    print(f"Global random seed: {global_seed}")

    # 백테스터 초기화 및 실행
    backtester = PortfolioBacktester(
        symbols, train_start, train_end, test_start, test_end
    )
    results = backtester.run_multiple_backtests(n_runs=10)
