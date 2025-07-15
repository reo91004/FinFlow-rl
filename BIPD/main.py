import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
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


class ImmuneCell:
    """면역 세포 기본 클래스"""

    def __init__(self, cell_id, activation_threshold=0.5):
        self.cell_id = cell_id
        self.activation_threshold = activation_threshold
        self.activation_level = 0.0
        self.memory_strength = 0.0


class TCell(ImmuneCell):
    """T-세포: 위험 탐지 담당"""

    def __init__(self, cell_id, sensitivity=0.1, random_state=None):
        super().__init__(cell_id)
        self.sensitivity = sensitivity
        self.detector = IsolationForest(contamination=sensitivity, random_state=random_state)
        self.is_trained = False

    def detect_anomaly(self, market_features):
        """시장 이상 상황 탐지"""
        if not self.is_trained:
            # 정상 시장 상태로 훈련
            self.detector.fit(market_features)
            self.is_trained = True
            return 0.0

        # 이상 점수 계산 (-1: 이상, 1: 정상)
        anomaly_scores = self.detector.decision_function(market_features)

        # 활성화 수준 계산 (0~1)
        self.activation_level = max(0, (1 - np.mean(anomaly_scores)) / 2)

        return self.activation_level


class BCell(ImmuneCell):
    """B-세포: 위험 대응 담당"""

    def __init__(self, cell_id, risk_type, response_strategy):
        super().__init__(cell_id)
        self.risk_type = risk_type  # 'volatility', 'correlation', 'momentum'
        self.response_strategy = response_strategy
        self.antibody_strength = 0.1

    def produce_antibody(self, antigen_pattern):
        """항원 패턴에 맞는 항체(방어 전략) 생성"""
        # 항원-항체 매칭 점수 계산
        if hasattr(self, "learned_patterns"):
            similarities = [
                cosine_similarity([antigen_pattern], [pattern])[0][0]
                for pattern in self.learned_patterns
            ]
            max_similarity = max(similarities) if similarities else 0
        else:
            max_similarity = 0

        # 항체 강도 계산
        self.antibody_strength = min(1.0, max_similarity + 0.1)

        return self.antibody_strength

    def adapt_response(self, antigen_pattern, effectiveness):
        """적응적 면역 반응 학습"""
        if not hasattr(self, "learned_patterns"):
            self.learned_patterns = []

        # 효과적인 패턴을 기억에 저장
        if effectiveness > 0.6:
            self.learned_patterns.append(antigen_pattern.copy())
            # 최대 10개 패턴만 유지
            if len(self.learned_patterns) > 10:
                self.learned_patterns.pop(0)


class MemoryCell:
    """기억 세포: 과거 위기 패턴 저장"""

    def __init__(self, max_memories=20):
        self.max_memories = max_memories
        self.crisis_memories = []  # (패턴, 대응전략, 효과)

    def store_memory(self, crisis_pattern, response_strategy, effectiveness):
        """위기 기억 저장"""
        memory = {
            "pattern": crisis_pattern.copy(),
            "strategy": response_strategy.copy(),
            "effectiveness": effectiveness,
            "strength": 1.0,
        }

        self.crisis_memories.append(memory)

        # 메모리 용량 관리
        if len(self.crisis_memories) > self.max_memories:
            # 효과가 낮은 기억부터 제거
            self.crisis_memories.sort(key=lambda x: x["effectiveness"])
            self.crisis_memories.pop(0)

    def recall_memory(self, current_pattern):
        """현재 패턴과 유사한 과거 기억 회상"""
        if not self.crisis_memories:
            return None, 0.0

        similarities = []
        for memory in self.crisis_memories:
            similarity = cosine_similarity([current_pattern], [memory["pattern"]])[0][0]
            similarities.append(similarity * memory["effectiveness"])

        best_memory_idx = np.argmax(similarities)
        best_similarity = similarities[best_memory_idx]

        if best_similarity > 0.7:  # 임계값 이상일 때만 기억 활용
            return self.crisis_memories[best_memory_idx], best_similarity

        return None, 0.0


class ImmunePortfolioSystem:
    """생체모방 면역 포트폴리오 시스템"""

    def __init__(self, n_assets, n_tcells=3, n_bcells=5, random_state=None):
        self.n_assets = n_assets

        # T-세포들 (다양한 민감도로 위험 탐지)
        # 각 T-세포마다 다른 random_state 사용
        self.tcells = [
            TCell(f"T{i}", sensitivity=0.05 + i * 0.02, 
                  random_state=None if random_state is None else random_state + i) 
            for i in range(n_tcells)
        ]

        # B-세포들 (위험 유형별 대응)
        self.bcells = [
            BCell("B1", "volatility", self._volatility_response),
            BCell("B2", "correlation", self._correlation_response),
            BCell("B3", "momentum", self._momentum_response),
            BCell("B4", "liquidity", self._liquidity_response),
            BCell("B5", "macro", self._macro_response),
        ]

        # 기억 세포
        self.memory_cell = MemoryCell()

        # 기본 포트폴리오 가중치
        self.base_weights = np.ones(n_assets) / n_assets
        self.current_weights = self.base_weights.copy()

        # 면역 시스템 상태
        self.immune_activation = 0.0
        self.crisis_level = 0.0

    def extract_market_features(self, market_data, lookback=20):
        """시장 특성 추출 (강화된 NaN 처리)"""
        if len(market_data) < lookback:
            return np.zeros(8)  # 기본 특성 벡터

        returns = market_data.pct_change().dropna()
        if len(returns) == 0:
            return np.zeros(8)

        recent_returns = returns.iloc[-lookback:]
        if len(recent_returns) == 0:
            return np.zeros(8)

        # NaN 안전 계산을 위한 헬퍼 함수
        def safe_mean(x):
            if len(x) == 0 or x.isnull().all():
                return 0.0
            return x.mean() if not np.isnan(x.mean()) else 0.0

        def safe_std(x):
            if len(x) == 0 or x.isnull().all():
                return 0.0
            return x.std() if not np.isnan(x.std()) else 0.0

        def safe_corr(x):
            try:
                if len(x) <= 1 or x.isnull().all().all():
                    return 0.0
                corr_matrix = np.corrcoef(x.T)
                if np.isnan(corr_matrix).any():
                    return 0.0
                return np.mean(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)])
            except:
                return 0.0

        def safe_skew(x):
            try:
                skew_vals = x.skew()
                if skew_vals.isnull().all():
                    return 0.0
                return skew_vals.mean() if not np.isnan(skew_vals.mean()) else 0.0
            except:
                return 0.0

        def safe_kurtosis(x):
            try:
                kurt_vals = x.kurtosis()
                if kurt_vals.isnull().all():
                    return 0.0
                return kurt_vals.mean() if not np.isnan(kurt_vals.mean()) else 0.0
            except:
                return 0.0

        features = [
            safe_std(recent_returns.std()),  # 평균 변동성
            safe_corr(recent_returns),  # 평균 상관관계
            safe_mean(recent_returns.mean()),  # 평균 수익률
            safe_skew(recent_returns),  # 평균 왜도
            safe_kurtosis(recent_returns),  # 평균 첨도
            safe_std(recent_returns.std()),  # 변동성의 변동성
            len(recent_returns[recent_returns.sum(axis=1) < -0.02])
            / max(len(recent_returns), 1),  # 하락일 비율
            (
                max(recent_returns.max().max() - recent_returns.min().min(), 0)
                if not recent_returns.empty
                else 0
            ),  # 수익률 범위
        ]

        # NaN 및 무한대 값 최종 처리
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def _volatility_response(self, activation_level):
        """변동성 위험 대응 전략"""
        # 변동성이 높을 때 안전 자산 비중 증가
        risk_reduction = activation_level * 0.3
        weights = self.base_weights * (1 - risk_reduction)
        # 안전 자산(JPM, JNJ, PG)에 추가 배분
        safe_indices = [6, 7, 8]  # JPM, JNJ, PG
        for idx in safe_indices:
            if idx < len(weights):
                weights[idx] += risk_reduction / len(safe_indices)
        return weights / np.sum(weights)

    def _correlation_response(self, activation_level):
        """상관관계 위험 대응 전략"""
        # 상관관계가 높을 때 분산 강화
        diversification_boost = activation_level * 0.2
        weights = self.base_weights.copy()
        # 가장 상관관계가 낮은 자산에 추가 배분
        weights = weights * (1 - diversification_boost) + diversification_boost / len(
            weights
        )
        return weights / np.sum(weights)

    def _momentum_response(self, activation_level):
        """모멘텀 위험 대응 전략"""
        # 모멘텀 급변 시 중립 포지션으로 복귀
        neutral_adjustment = activation_level * 0.25
        weights = self.base_weights * (1 - neutral_adjustment) + (
            self.base_weights * neutral_adjustment
        )
        return weights / np.sum(weights)

    def _liquidity_response(self, activation_level):
        """유동성 위험 대응 전략"""
        # 유동성 위험 시 대형주 비중 증가
        large_cap_boost = activation_level * 0.2
        weights = self.base_weights.copy()
        # 대형주 (AAPL, MSFT, AMZN, GOOGL) 비중 증가
        large_cap_indices = [0, 1, 2, 3]
        for idx in large_cap_indices:
            if idx < len(weights):
                weights[idx] += large_cap_boost / len(large_cap_indices)
        return weights / np.sum(weights)

    def _macro_response(self, activation_level):
        """거시경제 위험 대응 전략"""
        # 거시경제 불안 시 방어주 비중 증가
        defensive_boost = activation_level * 0.3
        weights = self.base_weights * (1 - defensive_boost)
        # 방어주 (JNJ, PG, V) 비중 증가
        defensive_indices = [7, 8, 9]
        for idx in defensive_indices:
            if idx < len(weights):
                weights[idx] += defensive_boost / len(defensive_indices)
        return weights / np.sum(weights)

    def immune_response(self, market_features):
        """면역 반응 실행"""
        # 1. T-세포 활성화 (위험 탐지)
        tcell_activations = []
        for tcell in self.tcells:
            activation = tcell.detect_anomaly(market_features.reshape(1, -1))
            tcell_activations.append(activation)

        # 전체 위험 수준 계산
        self.crisis_level = np.mean(tcell_activations)

        # 2. 기억 세포 확인
        recalled_memory, memory_strength = self.memory_cell.recall_memory(
            market_features
        )

        if recalled_memory and memory_strength > 0.8:
            # 강한 기억이 있으면 즉시 대응
            return recalled_memory["strategy"], "memory_response"

        # 3. B-세포 활성화 (위험 대응)
        if self.crisis_level > 0.3:  # 위험 임계값
            response_weights = []
            antibody_strengths = []

            for bcell in self.bcells:
                antibody_strength = bcell.produce_antibody(market_features)
                response_weight = bcell.response_strategy(
                    self.crisis_level * antibody_strength
                )
                response_weights.append(response_weight)
                antibody_strengths.append(antibody_strength)

            # 가장 강한 항체 반응 선택
            best_response_idx = np.argmax(antibody_strengths)
            self.immune_activation = antibody_strengths[best_response_idx]

            return (
                response_weights[best_response_idx],
                f"bcell_{self.bcells[best_response_idx].risk_type}",
            )

        # 위험 수준이 낮으면 기본 가중치 유지
        return self.base_weights, "normal"

    def update_memory(self, crisis_pattern, response_strategy, effectiveness):
        """면역 기억 업데이트"""
        self.memory_cell.store_memory(crisis_pattern, response_strategy, effectiveness)

        # B-세포 적응
        for bcell in self.bcells:
            bcell.adapt_response(crisis_pattern, effectiveness)


class ImmunePortfolioBacktester:
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
            raw_data = yf.download(
                symbols, start="2007-12-01", end="2025-01-01", progress=True
            )

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
                    columns=self.data.columns,
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

        immune_system = ImmunePortfolioSystem(n_assets=len(self.symbols), random_state=seed)

        # 훈련 단계 (면역 시스템 훈련)
        train_returns = self.train_data.pct_change().dropna()
        portfolio_values = [1.0]

        for i in range(len(train_returns)):
            current_data = self.train_data.iloc[: i + 1]

            # 시장 특성 추출
            market_features = immune_system.extract_market_features(current_data)

            # 면역 반응 실행
            weights, response_type = immune_system.immune_response(market_features)

            # 포트폴리오 수익률 계산
            portfolio_return = np.sum(weights * train_returns.iloc[i])
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

            # 면역 시스템 효과성 평가 및 학습
            if len(portfolio_values) > 20:
                recent_returns = (
                    np.diff(portfolio_values[-21:]) / portfolio_values[-21:-1]
                )
                effectiveness = np.mean(recent_returns) / (
                    np.std(recent_returns) + 1e-6
                )
                effectiveness = max(0, min(1, (effectiveness + 1) / 2))  # 0~1 정규화

                # 위기 상황에서의 대응 효과를 기억에 저장
                if immune_system.crisis_level > 0.3:
                    immune_system.update_memory(market_features, weights, effectiveness)

        # 테스트 단계
        test_returns = self.test_data.pct_change().dropna()
        test_portfolio_returns = []

        for i in range(len(test_returns)):
            current_data = self.test_data.iloc[: i + 1]

            # 시장 특성 추출
            market_features = immune_system.extract_market_features(current_data)

            # 면역 반응 실행
            weights, response_type = immune_system.immune_response(market_features)

            # 포트폴리오 수익률 계산
            portfolio_return = np.sum(weights * test_returns.iloc[i])
            test_portfolio_returns.append(portfolio_return)

        if return_model:
            return (
                pd.Series(test_portfolio_returns, index=test_returns.index),
                immune_system,
            )
        else:
            return pd.Series(test_portfolio_returns, index=test_returns.index)

    def save_model(self, immune_system, filename=None):
        """면역 시스템 모델 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"immune_system_{timestamp}.pkl"

        model_path = os.path.join(MODELS_DIR, filename)
        with open(model_path, "wb") as f:
            pickle.dump(immune_system, f)
        print(f"모델 저장 완료: {filename}")
        return model_path

    def load_model(self, filename):
        """면역 시스템 모델 로드"""
        model_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                immune_system = pickle.load(f)
            print(f"모델 로드 완료: {filename}")
            return immune_system
        else:
            print(f"모델 파일을 찾을 수 없습니다: {filename}")
            return None

    def save_results(self, metrics_df, filename=None):
        """결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bipd_results_{timestamp}"

        # CSV 저장
        csv_path = os.path.join(RESULTS_DIR, f"{filename}.csv")
        metrics_df.to_csv(csv_path, index=False)

        # 시각화 및 저장
        plt.figure(figsize=(15, 10))

        # 서브플롯 1: 성과 지표 박스플롯
        plt.subplot(2, 3, 1)
        metrics_df.boxplot(column=["Total Return"], ax=plt.gca())
        plt.title("Total Return Distribution")

        plt.subplot(2, 3, 2)
        metrics_df.boxplot(column=["Sharpe Ratio"], ax=plt.gca())
        plt.title("Sharpe Ratio Distribution")

        plt.subplot(2, 3, 3)
        metrics_df.boxplot(column=["Max Drawdown"], ax=plt.gca())
        plt.title("Max Drawdown Distribution")

        # 서브플롯 2: 상관관계 히트맵
        plt.subplot(2, 2, 3)
        correlation = metrics_df.corr()
        plt.imshow(correlation, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.title("Metrics Correlation")
        plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
        plt.yticks(range(len(correlation.columns)), correlation.columns)

        # 서브플롯 3: 성과 요약
        plt.subplot(2, 2, 4)
        plt.axis("off")
        summary_text = f"""
        BIPD 백테스트 결과 요약
        
        총 수익률: {metrics_df['Total Return'].mean():.2%}
        표준편차: {metrics_df['Volatility'].mean():.3f}
        최대 낙폭: {metrics_df['Max Drawdown'].mean():.2%}
        Sharpe Ratio: {metrics_df['Sharpe Ratio'].mean():.2f}
        Calmar Ratio: {metrics_df['Calmar Ratio'].mean():.2f}
        초기 자본: {metrics_df['Initial Capital'].iloc[0]:,.0f}원
        최종 자본: {metrics_df['Final Value'].mean():,.0f}원
        """
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment="center")

        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f"{filename}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"결과 저장 완료: {csv_path}, {plot_path}")
        return csv_path, plot_path

    def run_multiple_backtests(self, n_runs=10, save_results=True):
        """다중 백테스트 실행"""
        all_metrics = []
        best_immune_system = None
        best_sharpe = -np.inf

        print(f"BIPD 백테스트 {n_runs}회 실행 중...")

        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")

            # 각 실행마다 다른 시드 사용
            portfolio_returns, immune_system = self.backtest_single_run(
                seed=run, return_model=True
            )
            metrics = self.calculate_metrics(portfolio_returns)
            all_metrics.append(metrics)

            # 최고 성과 모델 저장
            if metrics["Sharpe Ratio"] > best_sharpe:
                best_sharpe = metrics["Sharpe Ratio"]
                best_immune_system = immune_system

        # 평균과 표준편차 계산
        metrics_df = pd.DataFrame(all_metrics)

        print("\n=== BIPD 백테스트 결과 ===")
        print(f"총 수익률: {metrics_df['Total Return'].mean():.2%}")
        print(f"표준편차: {metrics_df['Volatility'].mean():.3f}")
        print(f"최대 낙폭: {metrics_df['Max Drawdown'].mean():.2%}")
        print(f"Sharpe Ratio: {metrics_df['Sharpe Ratio'].mean():.2f}")
        print(f"Calmar Ratio: {metrics_df['Calmar Ratio'].mean():.2f}")
        print(f"초기 자본: {metrics_df['Initial Capital'].iloc[0]:,.0f}원")
        print(f"최종 자본: {metrics_df['Final Value'].mean():,.0f}원")

        # 결과 저장
        if save_results:
            self.save_results(metrics_df)
            if best_immune_system is not None:
                self.save_model(best_immune_system, "best_immune_system.pkl")

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
    backtester = ImmunePortfolioBacktester(
        symbols, train_start, train_end, test_start, test_end
    )
    results = backtester.run_multiple_backtests(n_runs=10)
