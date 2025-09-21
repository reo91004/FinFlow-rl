# src/agents/t_cell.py

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, List, Tuple
from collections import deque
from src.utils.logger import FinFlowLogger
from scipy import stats
from scipy.special import gammaln

import shap

class TCell:
    """
    T-Cell: Crisis Detection Agent
    
    시장 이상 상황을 감지하고 위기 수준을 4차원으로 분석
    - Overall crisis (전체)
    - Volatility crisis (변동성)
    - Correlation crisis (상관관계)
    - Volume crisis (거래량)
    """
    
    def __init__(self, 
                 feature_dim: Optional[int] = None,
                 contamination: float = 0.1,
                 n_estimators: int = 100,
                 ema_beta: float = 0.9,
                 window_size: int = 100,
                 feature_config: Optional[Dict] = None):
        """
        Args:
            feature_dim: 특성 차원 수 (None이면 config에서 계산)
            contamination: 이상치 비율
            n_estimators: Isolation Forest 트리 개수
            ema_beta: EMA 평활 계수
            window_size: 적응형 임계값 윈도우
            feature_config: feature 설정 딕셔너리
        """
        # feature_dim 동적 계산
        if feature_dim is None:
            if feature_config and 'dimensions' in feature_config:
                self.feature_dim = sum(feature_config['dimensions'].values())
            else:
                # 기본값
                # 기본값은 config에서 정의된 총 차원
                self.feature_dim = feature_config.get('total_dim', sum(feature_config.get('dimensions', {}).values())) if feature_config else sum({'returns': 3, 'technical': 4, 'structure': 3, 'momentum': 2}.values())
        else:
            self.feature_dim = feature_dim
        self.contamination = contamination
        self.ema_beta = ema_beta
        self.window_size = window_size
        self.logger = FinFlowLogger("TCell")
        
        # Isolation Forest 모델
        self.detector = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42
        )
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # 학습 상태
        self.is_fitted = False
        
        # Crisis history
        self.crisis_history = deque(maxlen=window_size)
        
        # Feature importance (SHAP용)
        self.feature_importance = None
        self.explainer = None
        
        # Crisis thresholds (adaptive)
        self.thresholds = {
            'overall': 0.7,
            'volatility': 0.7,
            'correlation': 0.7,
            'volume': 0.7
        }
        
        # EMA tracking
        self.crisis_ema = {
            'overall': 0.0,
            'volatility': 0.0,
            'correlation': 0.0,
            'volume': 0.0
        }

        # Auto-fit 용 데이터 버퍼
        self.feature_buffer = []

        self.logger.info(f"T-Cell 초기화 완료 - contamination={contamination}, n_estimators={n_estimators}")
    
    def fit(self, features: np.ndarray):
        """
        정상 시장 패턴 학습
        
        Args:
            features: Training features [n_samples, feature_dim]
        """
        self.logger.info(f"T-Cell 학습 시작 - {len(features)} 샘플")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit Isolation Forest
        self.detector.fit(features_scaled)
        self.is_fitted = True
        
        # Initialize SHAP explainer for Isolation Forest
        self.explainer = shap.Explainer(
            lambda x: -self.detector.score_samples(x),
            features_scaled[:100]  # Background samples
        )
        self.logger.info("SHAP explainer 초기화 완료")
        
        self.logger.info("T-Cell 학습 완료")
    
    def detect_crisis(self, market_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        다차원 위기 감지
        
        Args:
            market_data: 시장 데이터 딕셔너리 (env.get_market_data()의 반환값)
            
        Returns:
            crisis_info: Dictionary with crisis levels
        """
        # Extract features from market_data
        if isinstance(market_data, dict):
            features = market_data.get('features', np.zeros(self.feature_dim))
        else:
            features = market_data

        # Auto-fit 로직: 충분한 데이터가 버퍼에 쌓이면 자동 학습
        if not self.is_fitted:
            # 버퍼에 데이터 추가
            self.feature_buffer.append(features.copy())

            # 100개 이상 데이터가 모이면 자동 학습
            if len(self.feature_buffer) >= 100:
                self.logger.info("T-Cell 자동 학습 실행 (100개 샘플 누적)")
                buffer_array = np.array(self.feature_buffer)
                self.fit(buffer_array)
                # 버퍼 비우기
                self.feature_buffer = []
            else:
                # 아직 학습되지 않음 - 기본값 반환
                return {
                    'overall_crisis': 0.0,
                    'volatility_crisis': 0.0,
                    'correlation_crisis': 0.0,
                    'volume_crisis': 0.0,
                    'is_anomaly': False
                }
        
        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Anomaly score from Isolation Forest
        # score_samples returns negative scores, lower = more anomalous
        anomaly_score = -self.detector.score_samples(features_scaled)[0]
        
        # Normalize to [0, 1]
        overall_crisis = self._normalize_score(anomaly_score)
        
        # Decompose into specific crisis types (heuristic based on feature indices)
        # Assuming features: [returns(3), technical(4), structure(3), momentum(2)]
        volatility_crisis = self._compute_volatility_crisis(features[0])
        correlation_crisis = self._compute_correlation_crisis(features[0])
        volume_crisis = self._compute_volume_crisis(features[0])
        
        # Update EMA
        self.crisis_ema['overall'] = self.ema_beta * self.crisis_ema['overall'] + (1 - self.ema_beta) * overall_crisis
        self.crisis_ema['volatility'] = self.ema_beta * self.crisis_ema['volatility'] + (1 - self.ema_beta) * volatility_crisis
        self.crisis_ema['correlation'] = self.ema_beta * self.crisis_ema['correlation'] + (1 - self.ema_beta) * correlation_crisis
        self.crisis_ema['volume'] = self.ema_beta * self.crisis_ema['volume'] + (1 - self.ema_beta) * volume_crisis
        
        # Determine if anomaly
        is_anomaly = self.detector.predict(features_scaled)[0] == -1
        
        # Store in history
        self.crisis_history.append(overall_crisis)
        
        crisis_info = {
            'overall_crisis': overall_crisis,
            'volatility_crisis': volatility_crisis,
            'correlation_crisis': correlation_crisis,
            'volume_crisis': volume_crisis,
            'overall_ema': self.crisis_ema['overall'],
            'volatility_ema': self.crisis_ema['volatility'],
            'correlation_ema': self.crisis_ema['correlation'],
            'volume_ema': self.crisis_ema['volume'],
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(anomaly_score)
        }
        
        return crisis_info
    
    def get_feature_importance(self, features: np.ndarray) -> Dict[str, List[Tuple[int, float]]]:
        """
        Get feature importance using SHAP
        
        Args:
            features: Market features
            
        Returns:
            importance: Top important features for crisis detection
        """
        if self.explainer is None:
            # Explainer not initialized - raise error
            raise RuntimeError("SHAP explainer not initialized. Call train() first to initialize the explainer.")
        
        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get SHAP values
        shap_values = self.explainer(features_scaled)
        
        # Get absolute importance
        importance = np.abs(shap_values.values[0])
        
        # Get top 5 features
        top_indices = np.argsort(importance)[-5:][::-1]
        top_features = [(int(idx), float(importance[idx])) for idx in top_indices]
        
        # Feature names (if available)
        feature_names = [
            "recent_return", "avg_return", "volatility",
            "rsi", "macd", "bollinger", "volume",
            "correlation", "beta", "max_drawdown",
            "short_momentum", "long_momentum"
        ]
        
        return {
            'top_features': top_features,
            'feature_names': [feature_names[idx] if idx < len(feature_names) else f"feature_{idx}" 
                              for idx, _ in top_features]
        }
    
    def _normalize_score(self, score: float) -> float:
        """Normalize anomaly score to [0, 1]"""
        # Isolation Forest scores typically range from -0.5 to 0.5
        # More negative = more anomalous
        # Map to [0, 1] where 1 = high crisis
        return np.clip((score + 0.5) * 2, 0, 1)
    
    def _compute_volatility_crisis(self, features: np.ndarray) -> float:
        """
        Compute volatility crisis level
        Based on volatility feature (index 2)
        """
        if len(features) > 2:
            volatility = features[2]
            # High volatility = high crisis
            return np.clip(volatility * 10, 0, 1)  # Scale assuming vol ~ 0.1
        return 0.0
    
    def _compute_correlation_crisis(self, features: np.ndarray) -> float:
        """
        Compute correlation crisis level
        Based on correlation feature (index 7)
        """
        if len(features) > 7:
            correlation = features[7]
            # High correlation = potential crisis (contagion)
            return np.clip(abs(correlation), 0, 1)
        return 0.0
    
    def _compute_volume_crisis(self, features: np.ndarray) -> float:
        """
        Compute volume crisis level
        Based on volume proxy feature (index 6)
        """
        if len(features) > 6:
            volume = features[6]
            # Abnormal volume = crisis
            return np.clip(volume, 0, 1)
        return 0.0
    
    def get_crisis_regime(self, crisis_level: float) -> str:
        """
        Classify crisis regime
        
        Args:
            crisis_level: Overall crisis level [0, 1]
            
        Returns:
            regime: 'normal', 'medium', 'high'
        """
        if crisis_level < 0.4:
            return 'normal'
        elif crisis_level < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def update_thresholds(self):
        """Update adaptive thresholds based on crisis history"""
        if len(self.crisis_history) < 10:
            return
        
        # Compute quantiles
        crisis_array = np.array(self.crisis_history)
        
        # Update thresholds to maintain target crisis rates
        self.thresholds['overall'] = np.quantile(crisis_array, 0.85)
        
        # Log updates
        if len(self.crisis_history) % 50 == 0:
            self.logger.debug(f"임계값 업데이트: {self.thresholds}")
    
    def reset(self):
        """Reset crisis tracking"""
        self.crisis_history.clear()
        self.crisis_ema = {k: 0.0 for k in self.crisis_ema}
        self.logger.debug("T-Cell 상태 초기화")
    
    def get_state(self) -> Dict:
        """T-Cell 상태 반환 (체크포인트용)"""
        return {
            'is_fitted': self.is_fitted,
            'crisis_history': list(self.crisis_history),
            'crisis_ema': self.crisis_ema.copy(),
            'thresholds': self.thresholds.copy(),
            'scaler_mean': self.scaler.mean_ if self.is_fitted else None,
            'scaler_scale': self.scaler.scale_ if self.is_fitted else None
        }
    
    def load_state(self, state: Dict, training_data: Optional[np.ndarray] = None):
        """T-Cell 상태 로드

        Args:
            state: 저장된 T-Cell 상태
            training_data: IsolationForest 재학습용 데이터 (선택적)
        """
        self.is_fitted = state.get('is_fitted', False)
        self.crisis_history = deque(state.get('crisis_history', []), maxlen=self.window_size)
        self.crisis_ema = state.get('crisis_ema', {k: 0.0 for k in self.crisis_ema})
        self.thresholds = state.get('thresholds', self.thresholds)

        # Restore scaler state if fitted
        if self.is_fitted and state.get('scaler_mean') is not None:
            self.scaler.mean_ = np.array(state['scaler_mean'])
            self.scaler.scale_ = np.array(state['scaler_scale'])
            self.scaler.n_features_in_ = len(state['scaler_mean'])

            # IsolationForest는 state 저장이 불가능하므로 재학습 필요
            # 실제 데이터가 제공되면 사용, 아니면 일단 초기화만 (나중에 detect_crisis에서 학습)
            if training_data is not None and len(training_data) > 0:
                # 제공된 실제 데이터로 재학습
                self.detector.fit(training_data)
                self.logger.info(f"T-Cell detector 재학습 완료: {training_data.shape[0]} 샘플")
            else:
                # 데이터가 없으면 초기화된 상태로 둠
                # 첫 detect_crisis 호출 시 자동으로 학습됨
                self.logger.warning("T-Cell detector 재학습용 데이터 없음 - 첫 사용 시 학습 예정")
    
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        crisis_array = np.array(self.crisis_history) if self.crisis_history else np.array([0])
        
        return {
            'is_fitted': self.is_fitted,
            'avg_crisis': float(np.mean(crisis_array)),
            'max_crisis': float(np.max(crisis_array)),
            'current_crisis': float(self.crisis_ema.get('overall', 0)),
            'crisis_counts': {
                'normal': int(np.sum(crisis_array < 0.4)),
                'medium': int(np.sum((crisis_array >= 0.4) & (crisis_array < 0.7))),
                'high': int(np.sum(crisis_array >= 0.7))
            },
            'thresholds': self.thresholds,
            'window_size': len(self.crisis_history)
        }


class BOCPD:
    """
    Bayesian Online Changepoint Detection
    
    Detects regime changes in time series data using Bayesian inference
    """
    
    def __init__(self, hazard_rate: float = 0.01, mu_prior: float = 0.0, 
                 kappa_prior: float = 1.0, alpha_prior: float = 1.0, 
                 beta_prior: float = 1.0):
        """
        Args:
            hazard_rate: Prior probability of changepoint at each timestep
            mu_prior: Prior mean for Gaussian
            kappa_prior: Prior precision (confidence) for mean
            alpha_prior: Prior shape for Gamma (precision prior)
            beta_prior: Prior rate for Gamma (precision prior)
        """
        self.hazard_rate = hazard_rate
        self.mu_prior = mu_prior
        self.kappa_prior = kappa_prior
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Run length probabilities
        self.run_length_probs = np.array([1.0])
        
        # Sufficient statistics for each run length
        self.mus = np.array([mu_prior])
        self.kappas = np.array([kappa_prior])
        self.alphas = np.array([alpha_prior])
        self.betas = np.array([beta_prior])
        
        self.logger = FinFlowLogger("BOCPD")
        self.time_step = 0
        
    def update(self, x: float) -> Tuple[np.ndarray, float]:
        """
        Update with new observation
        
        Args:
            x: New observation
            
        Returns:
            run_length_probs: Posterior probabilities of run lengths
            changepoint_prob: Probability of changepoint at current time
        """
        self.time_step += 1
        
        # Predictive probabilities for each run length
        pred_probs = self._predictive_probability(x)
        
        # Growth probabilities (no changepoint)
        growth_probs = self.run_length_probs * pred_probs * (1 - self.hazard_rate)
        
        # Changepoint probability
        cp_prob = np.sum(self.run_length_probs * pred_probs * self.hazard_rate)
        
        # New run length distribution
        new_run_length_probs = np.zeros(len(growth_probs) + 1)
        new_run_length_probs[0] = cp_prob
        new_run_length_probs[1:] = growth_probs
        
        # Normalize
        new_run_length_probs = new_run_length_probs / np.sum(new_run_length_probs)
        
        # Update sufficient statistics
        self._update_statistics(x)
        
        # Truncate if too long (keep computational cost manageable)
        max_len = 100
        if len(new_run_length_probs) > max_len:
            new_run_length_probs = new_run_length_probs[:max_len]
            new_run_length_probs = new_run_length_probs / np.sum(new_run_length_probs)
            self.mus = self.mus[:max_len]
            self.kappas = self.kappas[:max_len]
            self.alphas = self.alphas[:max_len]
            self.betas = self.betas[:max_len]
        
        self.run_length_probs = new_run_length_probs
        
        # Changepoint probability is mass at run length 0
        changepoint_prob = float(new_run_length_probs[0])
        
        return new_run_length_probs, changepoint_prob
    
    def _predictive_probability(self, x: float) -> np.ndarray:
        """
        Compute predictive probability for each run length
        Using Student-t distribution (conjugate prior for unknown mean and variance)
        """
        # Degrees of freedom
        dfs = 2 * self.alphas
        
        # Means
        means = self.mus
        
        # Variances
        variances = self.betas * (self.kappas + 1) / (self.alphas * self.kappas)
        
        # Student-t log probabilities
        log_probs = []
        for df, mean, var in zip(dfs, means, variances):
            # Avoid numerical issues
            df = max(df, 1e-6)
            var = max(var, 1e-6)
            
            # Student-t log pdf
            log_prob = -0.5 * (df + 1) * np.log(1 + (x - mean)**2 / (df * var))
            log_prob += gammaln((df + 1) / 2) - gammaln(df / 2)
            log_prob -= 0.5 * np.log(df * np.pi * var)
            
            log_probs.append(log_prob)
        
        log_probs = np.array(log_probs)
        
        # Convert to probabilities (avoid overflow)
        max_log_prob = np.max(log_probs)
        probs = np.exp(log_probs - max_log_prob)
        
        return probs
    
    def _update_statistics(self, x: float):
        """
        Update sufficient statistics with new observation
        """
        # Update existing statistics
        new_mus = (self.kappas * self.mus + x) / (self.kappas + 1)
        new_kappas = self.kappas + 1
        new_alphas = self.alphas + 0.5
        new_betas = self.betas + self.kappas * (x - self.mus)**2 / (2 * (self.kappas + 1))
        
        # Add prior for new run length
        self.mus = np.concatenate([[self.mu_prior], new_mus])
        self.kappas = np.concatenate([[self.kappa_prior], new_kappas])
        self.alphas = np.concatenate([[self.alpha_prior], new_alphas])
        self.betas = np.concatenate([[self.beta_prior], new_betas])
    
    def detect_changepoint(self, threshold: float = 0.5) -> bool:
        """
        Detect if changepoint occurred
        
        Args:
            threshold: Probability threshold for detection
            
        Returns:
            is_changepoint: True if changepoint detected
        """
        if len(self.run_length_probs) > 0:
            return self.run_length_probs[0] > threshold
        return False
    
    def get_most_likely_run_length(self) -> int:
        """Get most likely run length (time since last changepoint)"""
        if len(self.run_length_probs) > 0:
            return int(np.argmax(self.run_length_probs))
        return 0
    
    def reset(self):
        """Reset to initial state"""
        self.run_length_probs = np.array([1.0])
        self.mus = np.array([self.mu_prior])
        self.kappas = np.array([self.kappa_prior])
        self.alphas = np.array([self.alpha_prior])
        self.betas = np.array([self.beta_prior])
        self.time_step = 0


class HMMRegimeDetector:
    """
    Hidden Markov Model for Regime Detection
    
    Identifies market regimes (bull/bear/sideways) using HMM
    """
    
    def __init__(self, n_states: int = 3, n_iter: int = 100):
        """
        Args:
            n_states: Number of hidden states (regimes)
            n_iter: Number of EM iterations for training
        """
        self.n_states = n_states
        self.n_iter = n_iter
        
        # Model parameters (will be learned)
        self.transition_matrix = None
        self.emission_means = None
        self.emission_stds = None
        self.initial_probs = None
        
        # Current state probabilities
        self.state_probs = None
        
        self.logger = FinFlowLogger("HMMRegime")
        self.is_fitted = False
        
    def fit(self, observations: np.ndarray):
        """
        Fit HMM using EM algorithm (Baum-Welch)
        
        Args:
            observations: Training data [n_samples, n_features]
        """
        assert observations.ndim == 2, "Observations must be 2D array"
        n_samples, n_features = observations.shape
        
        # Initialize parameters randomly
        self._initialize_parameters(observations)
        
        for iteration in range(self.n_iter):
            # E-step: Forward-backward algorithm
            alphas, betas, gammas, xis = self._forward_backward(observations)
            
            # M-step: Update parameters
            self._update_parameters(observations, gammas, xis)
            
            # Check convergence (simplified)
            if iteration > 0 and iteration % 10 == 0:
                log_likelihood = self._compute_log_likelihood(observations, alphas)
                self.logger.debug(f"Iteration {iteration}, Log-likelihood: {log_likelihood:.4f}")
        
        self.is_fitted = True
        self.logger.info(f"HMM fitted with {self.n_states} states")
    
    def _initialize_parameters(self, observations: np.ndarray):
        """Initialize HMM parameters"""
        n_samples, n_features = observations.shape
        
        # Random transition matrix (row-stochastic)
        self.transition_matrix = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)
        
        # K-means initialization for emission parameters
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_states, n_init=10, random_state=42)
        labels = kmeans.fit_predict(observations)
        
        self.emission_means = np.zeros((self.n_states, n_features))
        self.emission_stds = np.ones((self.n_states, n_features))
        
        for state in range(self.n_states):
            state_obs = observations[labels == state]
            if len(state_obs) > 0:
                self.emission_means[state] = np.mean(state_obs, axis=0)
                self.emission_stds[state] = np.std(state_obs, axis=0) + 1e-6
        
        # Uniform initial probabilities
        self.initial_probs = np.ones(self.n_states) / self.n_states
    
    def _forward_backward(self, observations: np.ndarray):
        """Forward-backward algorithm"""
        n_samples = len(observations)
        
        # Forward pass
        alphas = np.zeros((n_samples, self.n_states))
        
        # Initial step
        for state in range(self.n_states):
            alphas[0, state] = self.initial_probs[state] * self._emission_prob(observations[0], state)
        
        # Normalize
        alphas[0] = alphas[0] / (np.sum(alphas[0]) + 1e-10)
        
        # Forward recursion
        for t in range(1, n_samples):
            for state in range(self.n_states):
                alphas[t, state] = self._emission_prob(observations[t], state) * \
                                   np.sum(alphas[t-1] * self.transition_matrix[:, state])
            alphas[t] = alphas[t] / (np.sum(alphas[t]) + 1e-10)
        
        # Backward pass
        betas = np.zeros((n_samples, self.n_states))
        betas[-1] = 1.0
        
        for t in range(n_samples - 2, -1, -1):
            for state in range(self.n_states):
                for next_state in range(self.n_states):
                    betas[t, state] += self.transition_matrix[state, next_state] * \
                                      self._emission_prob(observations[t+1], next_state) * \
                                      betas[t+1, next_state]
            betas[t] = betas[t] / (np.sum(betas[t]) + 1e-10)
        
        # Compute gammas (state probabilities)
        gammas = alphas * betas
        gammas = gammas / (np.sum(gammas, axis=1, keepdims=True) + 1e-10)
        
        # Compute xis (transition probabilities)
        xis = np.zeros((n_samples - 1, self.n_states, self.n_states))
        
        for t in range(n_samples - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xis[t, i, j] = alphas[t, i] * self.transition_matrix[i, j] * \
                                  self._emission_prob(observations[t+1], j) * betas[t+1, j]
            xis[t] = xis[t] / (np.sum(xis[t]) + 1e-10)
        
        return alphas, betas, gammas, xis
    
    def _emission_prob(self, observation: np.ndarray, state: int) -> float:
        """Compute emission probability (Gaussian)"""
        mean = self.emission_means[state]
        std = self.emission_stds[state]
        
        # Multivariate Gaussian (assuming independence)
        log_prob = -0.5 * np.sum(((observation - mean) / std) ** 2)
        log_prob -= 0.5 * len(observation) * np.log(2 * np.pi)
        log_prob -= np.sum(np.log(std))
        
        return np.exp(np.clip(log_prob, -100, 100))
    
    def _update_parameters(self, observations: np.ndarray, gammas: np.ndarray, xis: np.ndarray):
        """M-step: Update HMM parameters"""
        n_samples = len(observations)
        
        # Update initial probabilities
        self.initial_probs = gammas[0]
        
        # Update transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = np.sum(xis[:, i, j])
                denominator = np.sum(gammas[:-1, i])
                self.transition_matrix[i, j] = numerator / (denominator + 1e-10)
        
        # Normalize rows
        self.transition_matrix = self.transition_matrix / \
                                (np.sum(self.transition_matrix, axis=1, keepdims=True) + 1e-10)
        
        # Update emission parameters
        for state in range(self.n_states):
            weight_sum = np.sum(gammas[:, state])
            
            # Weighted mean
            self.emission_means[state] = np.sum(observations * gammas[:, state:state+1], axis=0) / (weight_sum + 1e-10)
            
            # Weighted std
            diff = observations - self.emission_means[state]
            self.emission_stds[state] = np.sqrt(
                np.sum(diff**2 * gammas[:, state:state+1], axis=0) / (weight_sum + 1e-10)
            ) + 1e-6
    
    def _compute_log_likelihood(self, observations: np.ndarray, alphas: np.ndarray) -> float:
        """Compute log-likelihood of observations"""
        # Sum of log scaling factors (simplified)
        return np.sum(np.log(np.sum(alphas, axis=1) + 1e-10))
    
    def predict(self, observation: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict regime for new observation
        
        Args:
            observation: New observation [n_features]
            
        Returns:
            regime: Most likely regime
            probs: Probability distribution over regimes
        """
        assert self.is_fitted, "Model must be fitted first"
        
        # Update state probabilities
        if self.state_probs is None:
            self.state_probs = self.initial_probs.copy()
        
        # Compute new state probabilities
        new_probs = np.zeros(self.n_states)
        
        for state in range(self.n_states):
            # Emission probability
            emission = self._emission_prob(observation, state)
            
            # Transition probability
            transition = np.sum(self.state_probs * self.transition_matrix[:, state])
            
            new_probs[state] = emission * transition
        
        # Normalize
        new_probs = new_probs / (np.sum(new_probs) + 1e-10)
        self.state_probs = new_probs
        
        # Most likely regime
        regime = int(np.argmax(new_probs))
        
        return regime, new_probs
    
    def get_regime_characteristics(self) -> Dict[int, Dict]:
        """Get characteristics of each regime"""
        assert self.is_fitted, "Model must be fitted first"
        
        characteristics = {}
        
        for state in range(self.n_states):
            # Classify based on mean returns (assuming first feature is returns)
            mean_return = self.emission_means[state, 0] if len(self.emission_means[state]) > 0 else 0
            volatility = self.emission_stds[state, 0] if len(self.emission_stds[state]) > 0 else 1
            
            if mean_return > 0.001:
                regime_type = "bull"
            elif mean_return < -0.001:
                regime_type = "bear"
            else:
                regime_type = "sideways"
            
            characteristics[state] = {
                'type': regime_type,
                'mean_return': float(mean_return),
                'volatility': float(volatility),
                'persistence': float(self.transition_matrix[state, state])
            }
        
        return characteristics
    
    def reset(self):
        """Reset state probabilities"""
        self.state_probs = None