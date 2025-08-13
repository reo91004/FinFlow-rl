# core/curriculum.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
from constant import *
from utils.logger import BIPDLogger

warnings.filterwarnings("ignore")


class CurriculumScheduler:
    """커리큘럼 학습 스케줄러"""

    def __init__(
        self,
        total_episodes: int = TOTAL_EPISODES,
        difficulty_levels: int = 3,
        transition_threshold: float = 0.7,
        performance_window: int = 50,
    ):
        self.total_episodes = total_episodes
        self.difficulty_levels = difficulty_levels
        self.transition_threshold = transition_threshold
        self.performance_window = performance_window

        # 현재 커리큘럼 상태
        self.current_level = 0
        self.current_episode = 0
        self.level_performance_history = []
        self.level_start_episode = 0

        # 레벨별 설정
        self.level_configs = self._initialize_level_configs()

        # 성과 추적
        self.level_transitions = []
        self.episode_rewards = []

    def _initialize_level_configs(self) -> List[Dict]:
        """레벨별 설정 초기화"""
        return [
            {
                "name": "stable_market",
                "description": "Stable bull market conditions",
                "volatility_range": (0.05, 0.15),
                "crisis_probability": 0.1,
                "market_conditions": ["bull", "stable"],
                "min_episodes": CURRICULUM_MIN_EPISODES[0],
                "difficulty": 1.0,
            },
            {
                "name": "moderate_volatility",
                "description": "Moderate volatility with occasional corrections",
                "volatility_range": (0.15, 0.30),
                "crisis_probability": 0.3,
                "market_conditions": ["bull", "bear", "sideways"],
                "min_episodes": CURRICULUM_MIN_EPISODES[1],
                "difficulty": 2.0,
            },
            {
                "name": "crisis_conditions",
                "description": "High volatility crisis conditions",
                "volatility_range": (0.30, 0.60),
                "crisis_probability": 0.7,
                "market_conditions": ["crisis", "bear", "volatile"],
                "min_episodes": CURRICULUM_MIN_EPISODES[2],
                "difficulty": 3.0,
            },
        ]

    def get_current_curriculum(self) -> Dict:
        """현재 커리큘럼 레벨 정보 반환"""
        if self.current_level < len(self.level_configs):
            config = self.level_configs[self.current_level].copy()
            config.update(
                {
                    "level": self.current_level,
                    "episode": self.current_episode,
                    "progress": self.get_level_progress(),
                }
            )
            return config
        else:
            return self.level_configs[-1].copy()

    def get_level_progress(self) -> float:
        """현재 레벨에서의 진행률"""
        episodes_in_level = self.current_episode - self.level_start_episode
        min_episodes = self.level_configs[self.current_level]["min_episodes"]
        return min(episodes_in_level / min_episodes, 1.0)

    def should_advance_level(self) -> bool:
        """다음 레벨로 진행할지 결정"""
        if self.current_level >= len(self.level_configs) - 1:
            return False

        # 최소 에피소드 수 확인
        episodes_in_level = self.current_episode - self.level_start_episode
        min_episodes = self.level_configs[self.current_level]["min_episodes"]

        if episodes_in_level < min_episodes:
            return False

        # 성과 기준 확인
        if len(self.level_performance_history) >= self.performance_window:
            recent_performance = np.mean(
                self.level_performance_history[-self.performance_window :]
            )
            return recent_performance >= self.transition_threshold

        return False

    def advance_level(self):
        """다음 레벨로 진행"""
        if self.should_advance_level():
            transition_info = {
                "from_level": self.current_level,
                "to_level": self.current_level + 1,
                "episode": self.current_episode,
                "performance": np.mean(
                    self.level_performance_history[-self.performance_window :]
                ),
                "timestamp": datetime.now(),
            }

            self.level_transitions.append(transition_info)
            self.current_level += 1
            self.level_start_episode = self.current_episode
            self.level_performance_history = []

            print(
                f"커리큘럼 레벨을 {transition_info['from_level']}에서 {transition_info['to_level']}로 진행합니다."
            )
            print(f"성과: {transition_info['performance']:.3f}")

            return True
        return False

    def add_episode_result(self, reward: float, additional_metrics: Dict = None):
        """에피소드 결과 추가"""
        self.current_episode += 1
        self.episode_rewards.append(reward)

        # 정규화된 성과 점수 계산 (0-1 범위)
        normalized_score = self._calculate_normalized_score(reward, additional_metrics)
        self.level_performance_history.append(normalized_score)

        # 자동 레벨 진행
        self.advance_level()

    def _calculate_normalized_score(
        self, reward: float, additional_metrics: Dict = None
    ) -> float:
        """정규화된 성과 점수 계산"""
        # 개선된 보상 정규화 (실제 보상 범위 -1 ~ +1을 0 ~ 1로 변환)
        # 대부분의 보상이 -1 ~ +1 범위에 있으므로 이를 기준으로 정규화
        base_score = np.clip((reward + 1) / 2, 0, 1)

        # 추가 메트릭이 있는 경우 종합 점수 계산
        if additional_metrics:
            sharpe_score = np.clip(
                (additional_metrics.get("sharpe_ratio", 0) + 1) / 3, 0, 1
            )
            drawdown_score = np.clip(
                1 - abs(additional_metrics.get("max_drawdown", 0)), 0, 1
            )

            # 가중 평균
            final_score = base_score * 0.5 + sharpe_score * 0.3 + drawdown_score * 0.2
            return final_score

        return base_score

    def get_curriculum_summary(self) -> Dict:
        """커리큘럼 학습 요약"""
        return {
            "current_level": self.current_level,
            "total_episodes": self.current_episode,
            "level_transitions": len(self.level_transitions),
            "current_config": self.get_current_curriculum(),
            "transition_history": self.level_transitions,
            "average_performance_by_level": self._get_performance_by_level(),
        }

    def _get_performance_by_level(self) -> Dict:
        """레벨별 평균 성과"""
        performance_by_level = {}

        current_start = 0
        for i, transition in enumerate(self.level_transitions):
            level_rewards = self.episode_rewards[current_start : transition["episode"]]
            if level_rewards:
                performance_by_level[f"level_{i}"] = {
                    "avg_reward": np.mean(level_rewards),
                    "episodes": len(level_rewards),
                }
            current_start = transition["episode"]

        # 현재 레벨 성과
        current_level_rewards = self.episode_rewards[current_start:]
        if current_level_rewards:
            performance_by_level[f"level_{self.current_level}"] = {
                "avg_reward": np.mean(current_level_rewards),
                "episodes": len(current_level_rewards),
            }

        return performance_by_level


class MarketDataCurator:
    """시장 데이터 큐레이터 - 커리큘럼에 맞는 데이터 선별"""

    def __init__(self, market_data: pd.DataFrame):
        self.market_data = market_data
        self.returns = market_data.pct_change().dropna()
        
        # 로거 초기화
        self.logger = BIPDLogger().get_episode_logger()

        # 시장 조건 분석
        self.market_periods = self._analyze_market_periods()

    def _analyze_market_periods(self) -> Dict:
        """시장 조건별 기간 분석"""
        returns = self.returns.mean(axis=1)
        volatility = returns.rolling(20).std()

        periods = {
            "stable": [],
            "bull": [],
            "bear": [],
            "crisis": [],
            "sideways": [],
            "volatile": [],
        }

        for i in range(len(returns)):
            if i < 20:  # 초기 20일은 건너뛰기
                continue

            current_vol = volatility.iloc[i]
            recent_returns = returns.iloc[i - 19 : i + 1].mean()

            # 조건 분류
            if current_vol < 0.15:
                if recent_returns > 0.001:
                    periods["stable"].append(i)
                elif recent_returns > -0.001:
                    periods["sideways"].append(i)
                else:
                    periods["bear"].append(i)
            elif current_vol < 0.30:
                if recent_returns > 0.002:
                    periods["bull"].append(i)
                elif recent_returns < -0.002:
                    periods["bear"].append(i)
                else:
                    periods["volatile"].append(i)
            else:
                periods["crisis"].append(i)

        return periods

    def get_curriculum_data(
        self, curriculum_config: Dict, episode_length: int = EPISODE_LENGTH  # 기본값을 252로 변경
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """커리큘럼에 맞는 데이터 선별"""
        target_conditions = curriculum_config["market_conditions"]
        volatility_range = curriculum_config["volatility_range"]
        crisis_prob = curriculum_config["crisis_probability"]

        # 에피소드 길이 검증 및 조정 - 최소 100일 보장
        min_episode_length = 100  # 20에서 100으로 증가
        max_possible_length = len(self.market_data) - 50  # 안전 마진
        
        if episode_length > max_possible_length:
            episode_length = max(min_episode_length, max_possible_length)
            self.logger.warning(f"에피소드 길이를 {episode_length}로 조정")
        elif episode_length < min_episode_length:
            episode_length = min_episode_length
            self.logger.warning(f"에피소드 길이를 최소 {min_episode_length}로 조정")

        # 조건에 맞는 인덱스 수집
        candidate_indices = []
        for condition in target_conditions:
            if condition in self.market_periods:
                valid_indices = [
                    idx
                    for idx in self.market_periods[condition]
                    if idx + episode_length < len(self.market_data)
                ]  # 경계 검사
                candidate_indices.extend(valid_indices)

        if not candidate_indices:
            # 전체 데이터에서 변동성 기준으로 안전하게 선별
            volatility = self.returns.mean(axis=1).rolling(20).std()
            volatility = volatility.dropna()

            if len(volatility) > episode_length:
                mask = (volatility >= volatility_range[0]) & (
                    volatility <= volatility_range[1]
                )
                valid_indices = volatility[mask].index.tolist()

                # 인덱스를 정수로 변환하고 경계 검사
                candidate_indices = []
                for idx in valid_indices:
                    if isinstance(idx, pd.Timestamp):
                        idx_pos = self.market_data.index.get_loc(idx)
                    else:
                        idx_pos = idx

                    if idx_pos + episode_length < len(self.market_data):
                        candidate_indices.append(idx_pos)

        if not candidate_indices:
            # 마지막 폴백: 데이터 중간 부분에서 안전하게 선택
            safe_start = 50
            safe_end = len(self.market_data) - episode_length - 50
            if safe_end > safe_start:
                candidate_indices = list(range(safe_start, safe_end, episode_length))
            else:
                # 정말 데이터가 부족한 경우
                episode_length = min(20, len(self.market_data) // 2)
                candidate_indices = [len(self.market_data) // 4]
                print(
                    f"[경고] 데이터 부족으로 에피소드 길이를 {episode_length}로 축소했습니다."
                )

        # 위기 상황 강제 삽입 (확률 기반)
        if np.random.random() < crisis_prob and self.market_periods.get("crisis"):
            crisis_candidates = [
                idx
                for idx in self.market_periods["crisis"]
                if idx + episode_length < len(self.market_data)
            ]
            if crisis_candidates:
                crisis_idx = np.random.choice(crisis_candidates)
                candidate_indices.append(crisis_idx)

        # 안전한 시작점 선택
        try:
            start_idx = np.random.choice(candidate_indices)
            end_idx = start_idx + episode_length

            # 최종 경계 검사
            if end_idx >= len(self.market_data):
                end_idx = len(self.market_data) - 1
                start_idx = max(0, end_idx - episode_length)

            selected_data = self.market_data.iloc[start_idx:end_idx]

            # 데이터 유효성 검사
            if len(selected_data) < 10:  # 최소 10일 데이터 보장
                print(
                    f"[경고] 선택된 데이터가 너무 짧습니다 ({len(selected_data)}일). 기본 구간 사용."
                )
                # 안전한 기본 구간 선택
                default_start = len(self.market_data) // 3
                default_end = default_start + max(20, episode_length)
                selected_data = self.market_data.iloc[default_start:default_end]

            market_features = self._extract_episode_features(selected_data)
            
            # 최종 데이터 크기 검증 (로그 파일에만)
            self.logger.debug(f"에피소드 데이터 준비 완료: shape={selected_data.shape}, "
                            f"features_shape={market_features.shape if market_features is not None else 'None'}")

            return selected_data, market_features

        except Exception as e:
            print(f"[오류] 커리큘럼 데이터 선택 실패: {e}")
            # 오류 복구: 안전한 기본 구간
            safe_start = len(self.market_data) // 3
            safe_end = safe_start + min(episode_length, len(self.market_data) // 3)
            recovery_data = self.market_data.iloc[safe_start:safe_end]
            recovery_features = self._extract_episode_features(recovery_data)
            return recovery_data, recovery_features

    def _extract_episode_features(self, episode_data: pd.DataFrame) -> np.ndarray:
        """에피소드 데이터에서 특성 추출"""
        try:
            if len(episode_data) == 0:
                return np.zeros(8, dtype=np.float32)

            returns = episode_data.pct_change().dropna()

            if len(returns) == 0:
                return np.zeros(8, dtype=np.float32)

            # 통계 계산
            def safe_stat(func, data, default=0.0):
                try:
                    if len(data) == 0:
                        return default
                    result = func(data)
                    if np.isnan(result) or np.isinf(result):
                        return default
                    return float(result)
                except Exception:
                    return default

            features = [
                safe_stat(lambda x: x.std().mean(), returns),  # 평균 변동성
                safe_stat(
                    lambda x: self._safe_correlation(x), returns
                ),  # 평균 상관계수
                safe_stat(lambda x: x.mean().mean(), returns),  # 평균 수익률
                safe_stat(lambda x: x.skew().mean(), returns),  # 평균 왜도
                safe_stat(lambda x: x.kurtosis().mean(), returns),  # 평균 첨도
                safe_stat(
                    lambda x: (x < -0.02).sum().sum() / max(1, len(x)), returns
                ),  # 하락일 비율
                safe_stat(
                    lambda x: x.max().max() - x.min().min(), returns
                ),  # 가격 범위
                safe_stat(
                    lambda x: len(x[x.sum(axis=1) < -0.05]) / max(1, len(x)), returns
                ),  # 큰 하락일 비율
            ]

            features = np.array(features, dtype=np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            return features

        except Exception as e:
            print(f"[경고] 에피소드 특성 추출 실패: {e}")
            return np.zeros(8, dtype=np.float32)

    def _safe_correlation(self, returns):
        """상관계수 계산"""
        try:
            if len(returns) <= 1 or returns.isnull().all().all():
                return 0.0

            # 최소 2개 이상의 유효한 열이 있는지 확인
            valid_cols = returns.dropna(axis=1, how="all").columns
            if len(valid_cols) < 2:
                return 0.0

            corr_matrix = returns[valid_cols].corr()
            if corr_matrix.isnull().all().all():
                return 0.0

            # 대각선 제외한 평균 상관계수
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            corr_values = corr_matrix.values[mask]
            valid_corr = corr_values[~np.isnan(corr_values)]

            if len(valid_corr) > 0:
                return np.mean(valid_corr)
            else:
                return 0.0
        except Exception:
            return 0.0


class CurriculumLearningManager:
    """커리큘럼 학습 관리자"""

    def __init__(
        self,
        market_data: pd.DataFrame,
        total_episodes: int = TOTAL_EPISODES,
        episode_length: int = EPISODE_LENGTH,
    ):
        self.scheduler = CurriculumScheduler(total_episodes=total_episodes)
        self.data_curator = MarketDataCurator(market_data)
        self.episode_length = episode_length

        # 학습 이력
        self.training_history = []

    def get_next_training_episode(self) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """다음 훈련 에피소드 데이터 반환"""
        curriculum_config = self.scheduler.get_current_curriculum()
        episode_data, features = self.data_curator.get_curriculum_data(
            curriculum_config, self.episode_length
        )

        return episode_data, features, curriculum_config

    def record_episode_result(
        self,
        reward: float,
        portfolio_return: float = None,
        sharpe_ratio: float = None,
        max_drawdown: float = None,
    ):
        """에피소드 결과 기록"""
        additional_metrics = {}
        if portfolio_return is not None:
            additional_metrics["portfolio_return"] = portfolio_return
        if sharpe_ratio is not None:
            additional_metrics["sharpe_ratio"] = sharpe_ratio
        if max_drawdown is not None:
            additional_metrics["max_drawdown"] = max_drawdown

        self.scheduler.add_episode_result(reward, additional_metrics)

        # 이력 기록
        episode_record = {
            "episode": self.scheduler.current_episode,
            "level": self.scheduler.current_level,
            "reward": reward,
            "curriculum_config": self.scheduler.get_current_curriculum()["name"],
            **additional_metrics,
        }
        self.training_history.append(episode_record)

    def is_curriculum_complete(self) -> bool:
        """커리큘럼 학습 완료 여부"""
        # 1000 에피소드에 도달하면 레벨과 관계없이 종료
        if self.scheduler.current_episode >= self.scheduler.total_episodes:
            return True
            
        # 또는 최고 레벨에 도달하고 해당 레벨의 최소 에피소드를 완료한 경우
        if self.scheduler.current_level >= len(self.scheduler.level_configs) - 1:
            episodes_in_level = (
                self.scheduler.current_episode - self.scheduler.level_start_episode
            )
            min_episodes = self.scheduler.level_configs[self.scheduler.current_level][
                "min_episodes"
            ]
            return episodes_in_level >= min_episodes
            
        return False

    def get_curriculum_progress(self) -> Dict:
        """커리큘럼 진행 상황"""
        return {
            "current_episode": self.scheduler.current_episode,
            "total_episodes": self.scheduler.total_episodes,
            "current_level": self.scheduler.current_level,
            "max_level": len(self.scheduler.level_configs) - 1,
            "level_progress": self.scheduler.get_level_progress(),
            "overall_progress": self.scheduler.current_episode
            / self.scheduler.total_episodes,
            "recent_performance": (
                np.mean(self.scheduler.level_performance_history[-10:])
                if len(self.scheduler.level_performance_history) >= 10
                else 0.0
            ),
        }

    def get_training_summary(self) -> Dict:
        """훈련 요약 정보"""
        summary = self.scheduler.get_curriculum_summary()
        summary["training_history"] = self.training_history[
            -100:
        ]  # 최근 100개 에피소드

        if self.training_history:
            df = pd.DataFrame(self.training_history)
            summary["performance_trend"] = {
                "avg_reward_by_level": df.groupby("level")["reward"].mean().to_dict(),
                "episode_count_by_level": df.groupby("level").size().to_dict(),
                "recent_improvement": (
                    df["reward"].tail(50).mean() - df["reward"].head(50).mean()
                    if len(df) >= 100
                    else 0.0
                ),
            }

        return summary

    def save_curriculum_state(self, filepath: str):
        """커리큘럼 상태 저장"""
        import pickle

        state = {
            "scheduler": self.scheduler,
            "training_history": self.training_history,
            "episode_length": self.episode_length,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def load_curriculum_state(self, filepath: str):
        """커리큘럼 상태 로드"""
        import pickle

        try:
            with open(filepath, "rb") as f:
                state = pickle.load(f)

            self.scheduler = state["scheduler"]
            self.training_history = state["training_history"]
            self.episode_length = state["episode_length"]

            return True
        except Exception as e:
            print(f"커리큘럼 상태 로드 중 오류 발생: {e}")
            return False
