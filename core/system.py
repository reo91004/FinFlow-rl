# core/system.py

import numpy as np
import os
from typing import Dict, Tuple, List, Optional
from agents import TCell, BCell, MemoryCell
from utils.logger import BIPDLogger
from config import *


class ImmunePortfolioSystem:
    """
    면역 시스템 기반 포트폴리오 관리자

    T-Cell, B-Cell, Memory Cell을 통합하여
    시장 상황에 적응적으로 반응하는 포트폴리오 전략 실행
    """

    def __init__(
        self, n_assets: int, state_dim: int, symbols: Optional[List[str]] = None
    ):
        self.n_assets = n_assets
        self.state_dim = state_dim
        self.symbols = symbols or SYMBOLS  # config.py의 SYMBOLS 사용

        # T-Cell (위기 감지)
        self.tcell = TCell(
            contamination=TCELL_CONTAMINATION,
            sensitivity=TCELL_SENSITIVITY,
            random_state=GLOBAL_SEED,
        )

        # B-Cell 전문가 그룹 (위험 유형별 특화)
        self.bcells = {
            "volatility": BCell(
                "volatility", state_dim, n_assets, ACTOR_LR, CRITIC_LR, ALPHA_LR
            ),  # 고변동성 시장
            "correlation": BCell(
                "correlation", state_dim, n_assets, ACTOR_LR, CRITIC_LR, ALPHA_LR
            ),  # 상관관계 변화
            "momentum": BCell(
                "momentum", state_dim, n_assets, ACTOR_LR, CRITIC_LR, ALPHA_LR
            ),  # 모멘텀 추세
            "defensive": BCell(
                "defensive", state_dim, n_assets, ACTOR_LR, CRITIC_LR, ALPHA_LR
            ),  # 방어적 전략
            "growth": BCell(
                "growth", state_dim, n_assets, ACTOR_LR, CRITIC_LR, ALPHA_LR
            ),  # 성장 중심
        }

        # Memory Cell (경험 저장 및 회상)
        self.memory = MemoryCell(
            capacity=MEMORY_CAPACITY,
            embedding_dim=EMBEDDING_DIM,
            similarity_threshold=0.7,
        )

        # 시스템 상태
        self.is_trained = False
        self.training_steps = 0
        self.decision_count = 0

        # 의사결정 히스토리
        self.decision_history = []
        self.performance_history = []

        # 메모리 가이던스 통계
        self.memory_stats = {
            "guidance_applied": 0,
            "total_guidance_attempts": 0,
            "confidence_sum": 0.0,
            "last_report_step": 0,
        }

        # B-Cell 성과 추적 시스템
        self.bcell_performance = {
            name: {
                "recent_rewards": [],  # 최근 10회 성과
                "consecutive_selections": 0,
                "total_selections": 0,
            }
            for name in self.bcells.keys()
        }

        # 로깅 통계 (장황한 로그 줄이기)
        self.logging_stats = {
            "random_selections": 0,
            "penalty_applications": 0,
            "last_log_report": 0,
            "log_interval": 100,  # 100번마다 통계 요약 출력
        }

        self.logger = BIPDLogger("ImmuneSystem")

        self.logger.info(
            f"면역 포트폴리오 시스템이 초기화되었습니다. "
            f"자산수={n_assets}, 상태차원={state_dim}, "
            f"B-Cell={len(self.bcells)}개, "
            f"Device={get_device_info()}"
        )

    def fit_tcell(self, historical_features: np.ndarray) -> bool:
        """T-Cell 학습 (정상 시장 패턴)"""
        success = self.tcell.fit(historical_features)
        if success:
            self.logger.info("T-Cell 학습이 완료되었습니다.")
        return success

    def decide(
        self, state: np.ndarray, training: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        포트폴리오 의사결정

        Args:
            state: [market_features(12), crisis_level(1), prev_weights(n_assets)]
            training: 훈련 모드 여부

        Returns:
            weights: 포트폴리오 가중치
            info: 의사결정 상세 정보
        """
        # 상태 분해
        market_features = state[:FEATURE_DIM]
        crisis_level = state[FEATURE_DIM]
        prev_weights = state[FEATURE_DIM + 1 :]

        # T-Cell 다차원 위기 감지
        crisis_detection = self.tcell.detect_crisis(market_features)

        # 기존 호환성을 위한 단일 위기 수준 계산
        if isinstance(crisis_detection, dict):
            detected_crisis = crisis_detection["overall_crisis"]
            final_crisis_level = max(crisis_level, detected_crisis)
            crisis_info = crisis_detection  # 다차원 정보 보존
        else:
            # 하위 호환성 (단일 float 반환 시)
            final_crisis_level = max(crisis_level, crisis_detection)
            crisis_info = final_crisis_level

        # B-Cell 선택 (다차원 위기 정보 기반)
        selected_bcell_name = self._select_bcell(crisis_info)
        selected_bcell = self.bcells[selected_bcell_name]

        # Phase 3: 적응형 엔트로피 업데이트 (선택된 B-Cell에 T-Cell 신호 전달)
        if hasattr(selected_bcell, "update_adaptive_target_entropy"):
            # 시장 안정도 추정 (과거 위기 레벨의 역수)
            market_stability = 1.0 - final_crisis_level
            regime_info = selected_bcell.update_adaptive_target_entropy(
                crisis_level=final_crisis_level, market_stability=market_stability
            )

        # Memory 회상
        memory_guidance = self.memory.get_memory_guidance(
            market_features, final_crisis_level, k=MEMORY_K
        )

        # 포트폴리오 가중치 생성
        weights = selected_bcell.get_action(state, deterministic=not training)

        # Memory 가이던스 적용 (있는 경우)
        self.memory_stats["total_guidance_attempts"] += 1
        if (
            memory_guidance["has_guidance"]
            and memory_guidance["confidence"] > 0.8
            and memory_guidance["recommended_action"] is not None
        ):

            memory_weight = min(0.3, memory_guidance["confidence"] - 0.5)
            weights = (1 - memory_weight) * weights + memory_weight * memory_guidance[
                "recommended_action"
            ]
            weights = weights / weights.sum()  # 정규화

            # 통계 업데이트
            self.memory_stats["guidance_applied"] += 1
            self.memory_stats["confidence_sum"] += memory_guidance["confidence"]

            # 간헐적 로깅 (매 100회마다)
            if self.memory_stats["guidance_applied"] % 100 == 0:
                avg_confidence = (
                    self.memory_stats["confidence_sum"]
                    / self.memory_stats["guidance_applied"]
                )
                usage_rate = (
                    self.memory_stats["guidance_applied"]
                    / self.memory_stats["total_guidance_attempts"]
                )
                self.logger.debug(
                    f"Memory 가이던스 통계 (최근 100회): 사용률={usage_rate:.1%}, "
                    f"평균 신뢰도={avg_confidence:.3f}"
                )

        # 의사결정 정보 수집 (XAI 강화)
        decision_info = {
            "crisis_level": float(final_crisis_level),
            "selected_bcell": selected_bcell_name,
            "memory_guidance": memory_guidance["has_guidance"],
            "memory_confidence": memory_guidance.get("confidence", 0.0),
            "specialization_scores": {
                name: bcell.get_specialization_score(crisis_info)
                for name, bcell in self.bcells.items()
            },
            "weights_concentration": float(np.sum(weights**2)),
            "decision_count": self.decision_count,
            # XAI 확장 데이터
            "xai_data": {
                "tcell_data": (
                    crisis_info
                    if isinstance(crisis_info, dict)
                    else {
                        "overall_crisis": crisis_info,
                        "volatility_crisis": 0.0,
                        "correlation_crisis": 0.0,
                        "volume_crisis": 0.0,
                    }
                ),
                "bcell_data": {
                    "strategy_scores": {
                        name: bcell.get_specialization_score(crisis_info)
                        for name, bcell in self.bcells.items()
                    },
                    "selected_strategy": selected_bcell_name,
                    "selection_reason": self._get_selection_reason(
                        selected_bcell_name, crisis_info
                    ),
                },
                "memory_data": {
                    "has_guidance": memory_guidance["has_guidance"],
                    "confidence": memory_guidance.get("confidence", 0.0),
                    "similar_episodes": memory_guidance.get("similar_episodes", []),
                    "similarity_scores": memory_guidance.get("similarity_scores", []),
                },
                "portfolio_data": {
                    "weights": (
                        {self.symbols[i]: float(w) for i, w in enumerate(weights)}
                        if hasattr(self, "symbols")
                        else {}
                    ),
                    "top_holdings": self._get_top_holdings(weights, 5),
                    "concentration": float(np.sum(weights**2)),
                    "diversification": float(
                        1.0 / np.sum(weights**2)
                    ),  # 역집중도 = 다양성
                },
                "step": self.decision_count,
            },
        }

        # 의사결정 히스토리 저장
        self.decision_history.append(
            {
                "step": self.decision_count,
                "crisis_level": final_crisis_level,
                "selected_bcell": selected_bcell_name,
                "weights": weights.copy(),
                "memory_used": memory_guidance["has_guidance"],
            }
        )

        self.decision_count += 1

        # 로깅 (주기적으로만)
        if self.decision_count % 50 == 0:
            self.logger.debug(
                f"의사결정 #{self.decision_count}: "
                f"위기수준={final_crisis_level:.3f}, "
                f"선택전략={selected_bcell_name}, "
                f"메모리={memory_guidance['has_guidance']}"
            )

        return weights, decision_info

    def _select_bcell(self, crisis_info) -> str:
        """
        다차원 위기 정보에 따른 B-Cell 선택 (다양성 확보)

        성능 기반 동적 페널티 + 확률적 선택을 통한 편향성 해결
        """
        # 각 B-Cell의 기본 전문성 점수 계산 (다차원 위기 정보 사용)
        scores = {}
        for name, bcell in self.bcells.items():
            base_score = bcell.get_specialization_score(crisis_info)

            # 최근 성과 기반 페널티 계산
            performance_data = self.bcell_performance[name]
            recent_rewards = performance_data["recent_rewards"]

            if len(recent_rewards) > 5:
                avg_reward = np.mean(recent_rewards)
                # 성과가 나쁠수록 페널티 적용
                performance_penalty = max(0, -avg_reward * 0.5)
                base_score -= performance_penalty

            # 연속 선택 페널티 (강제 순환)
            consecutive_count = performance_data["consecutive_selections"]
            if consecutive_count >= 5:  # 5회 연속 선택시
                base_score *= 0.3  # 점수 대폭 감소
                self.logging_stats["penalty_applications"] += 1

            scores[name] = base_score

        # 확률적 선택 (완전 greedy 방지)
        if np.random.random() < 0.2:  # 20% 확률로 랜덤 선택
            selected = np.random.choice(list(self.bcells.keys()))
            self.logging_stats["random_selections"] += 1
        else:
            # 최고 점수 전략 선택
            selected = max(scores, key=scores.get)

        # 선택 통계 업데이트
        self._update_selection_stats(selected)

        # 주기적 로깅 통계 보고
        decisions_since_last_report = (
            self.decision_count - self.logging_stats["last_log_report"]
        )
        if decisions_since_last_report >= self.logging_stats["log_interval"]:
            self._log_selection_statistics()
            self.logging_stats["last_log_report"] = self.decision_count

        return selected

    def _update_selection_stats(self, selected_bcell: str) -> None:
        """B-Cell 선택 통계 업데이트"""
        # 선택된 B-Cell 통계 업데이트
        self.bcell_performance[selected_bcell]["total_selections"] += 1

        # 연속 선택 카운트 업데이트
        if (
            self.decision_history
            and self.decision_history[-1]["selected_bcell"] == selected_bcell
        ):
            self.bcell_performance[selected_bcell]["consecutive_selections"] += 1
        else:
            self.bcell_performance[selected_bcell]["consecutive_selections"] = 1

        # 다른 B-Cell들의 연속 선택 카운트 리셋
        for name, data in self.bcell_performance.items():
            if name != selected_bcell:
                data["consecutive_selections"] = 0

    def _get_recent_rewards(self, bcell_name: str, window: int = 10) -> list:
        """특정 B-Cell의 최근 성과 반환"""
        return self.bcell_performance[bcell_name]["recent_rewards"][-window:]

    def _get_consecutive_count(self, bcell_name: str) -> int:
        """특정 B-Cell의 연속 선택 횟수 반환"""
        return self.bcell_performance[bcell_name]["consecutive_selections"]

    def _log_selection_statistics(self) -> None:
        """B-Cell 선택 통계 요약 로깅 (장황한 로그 대신)"""
        interval = self.logging_stats["log_interval"]
        random_count = self.logging_stats["random_selections"]
        penalty_count = self.logging_stats["penalty_applications"]

        # 현재 연속 선택 상황 체크
        consecutive_issues = []
        for name, data in self.bcell_performance.items():
            consecutive = data["consecutive_selections"]
            if consecutive >= 3:  # 3회 이상 연속시 보고
                consecutive_issues.append(f"{name}:{consecutive}회")

        # 통계 요약 로깅
        self.logger.debug(
            f"B-Cell 선택 통계 (최근 {interval}회): "
            f"탐험적 선택 {random_count}회 ({random_count/interval:.1%}), "
            f"페널티 적용 {penalty_count}회, "
            f"연속선택: [{', '.join(consecutive_issues) if consecutive_issues else '정상'}]"
        )

        # 통계 리셋
        self.logging_stats["random_selections"] = 0
        self.logging_stats["penalty_applications"] = 0

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done,
    ) -> None:
        """
        시스템 업데이트 (학습)

        Args:
            state: 현재 상태
            action: 선택된 행동 (포트폴리오 가중치)
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        # NumPy 타입을 Python native 타입으로 안전하게 변환
        done = bool(done)  # numpy.bool을 Python bool로 변환
        reward = float(reward)  # numpy.float을 Python float로 변환

        # 상태 분해
        market_features = state[:FEATURE_DIM]
        crisis_level = float(state[FEATURE_DIM])  # numpy scalar을 Python float로 변환

        # Memory에 경험 저장
        self.memory.store(
            state=market_features,
            action=action,
            reward=reward,
            crisis_level=crisis_level,
            additional_info={
                "step": int(self.training_steps),  # numpy int를 Python int로 변환
                "done": done,
            },
        )

        # B-Cell 업데이트 (모든 전문가 학습)
        for bcell in self.bcells.values():
            bcell.store_experience(state, action, reward, next_state, done)

            # 주기적 학습
            if self.training_steps % UPDATE_FREQUENCY == 0:
                bcell.update()

        # 성과 히스토리 업데이트
        self.performance_history.append(
            {
                "step": self.training_steps,
                "reward": reward,
                "crisis_level": crisis_level,
            }
        )

        # 마지막 선택된 B-Cell의 성과 추적 업데이트
        if self.decision_history:
            last_decision = self.decision_history[-1]
            selected_bcell = last_decision["selected_bcell"]

            # 최근 성과 리스트에 추가 (최대 10개 유지)
            recent_rewards = self.bcell_performance[selected_bcell]["recent_rewards"]
            recent_rewards.append(reward)
            if len(recent_rewards) > 10:
                recent_rewards.pop(0)  # 가장 오래된 것 제거

        self.training_steps += 1

        # 주기적 로깅
        if self.training_steps % (LOG_INTERVAL * 10) == 0:
            self.logger.info(
                f"시스템 업데이트: 스텝={self.training_steps}, "
                f"보상={reward:.4f}, 메모리={len(self.memory.memories)}개"
            )

    def get_system_explanation(self, state: np.ndarray) -> Dict:
        """
        현재 의사결정에 대한 종합 설명 (XAI)
        """
        market_features = state[:FEATURE_DIM]
        crisis_level = state[FEATURE_DIM]

        # T-Cell 다차원 위기 감지
        crisis_detection = self.tcell.detect_crisis(market_features)

        # 기존 호환성을 위한 처리
        if isinstance(crisis_detection, dict):
            crisis_info = crisis_detection
            overall_crisis = crisis_detection["overall_crisis"]
        else:
            crisis_info = crisis_level
            overall_crisis = crisis_level

        # T-Cell 설명
        tcell_explanation = self.tcell.get_anomaly_explanation(market_features)

        # 선택된 B-Cell 설명 (다차원 위기 정보 기반)
        selected_bcell_name = self._select_bcell(crisis_info)
        bcell_explanation = self.bcells[selected_bcell_name].get_explanation(state)

        # Memory 통계
        memory_stats = self.memory.get_statistics()

        # 전체 설명
        explanation = {
            "system_overview": {
                "training_steps": self.training_steps,
                "decision_count": self.decision_count,
                "tcell_fitted": self.tcell.is_fitted,
            },
            "crisis_detection": tcell_explanation,
            "multidimensional_crisis": (
                crisis_detection if isinstance(crisis_detection, dict) else None
            ),
            "strategy_selection": {
                "selected_strategy": selected_bcell_name,
                "selection_reason": f"다차원 위기 분석 결과에 최적화됨 (전체: {overall_crisis:.3f})",
                "all_specialization_scores": {
                    name: bcell.get_specialization_score(crisis_info)
                    for name, bcell in self.bcells.items()
                },
            },
            "portfolio_generation": bcell_explanation,
            "memory_system": memory_stats,
            "recent_decisions": (
                self.decision_history[-5:] if self.decision_history else []
            ),
        }

        return explanation

    def get_performance_summary(self) -> Dict:
        """성과 요약 통계"""
        if not self.performance_history:
            return {"message": "성과 데이터가 없습니다."}

        rewards = [p["reward"] for p in self.performance_history]
        crisis_levels = [p["crisis_level"] for p in self.performance_history]

        # B-Cell 사용 통계
        bcell_usage = {}
        for decision in self.decision_history:
            bcell = decision["selected_bcell"]
            bcell_usage[bcell] = bcell_usage.get(bcell, 0) + 1

        # B-Cell 다양성 통계
        bcell_diversity_stats = {}
        total_selections = sum(
            self.bcell_performance[name]["total_selections"]
            for name in self.bcells.keys()
        )

        for name, data in self.bcell_performance.items():
            recent_rewards = data["recent_rewards"]
            selection_rate = data["total_selections"] / max(total_selections, 1)

            bcell_diversity_stats[name] = {
                "total_selections": data["total_selections"],
                "selection_rate": selection_rate,
                "consecutive_selections": data["consecutive_selections"],
                "avg_recent_reward": np.mean(recent_rewards) if recent_rewards else 0.0,
                "recent_performance_count": len(recent_rewards),
            }

        # 다양성 지수 계산 (Shannon Entropy)
        diversity_index = 0.0
        if total_selections > 0:
            for name in self.bcells.keys():
                selection_rate = (
                    self.bcell_performance[name]["total_selections"] / total_selections
                )
                if selection_rate > 0:
                    diversity_index -= selection_rate * np.log(selection_rate)

        summary = {
            "training_steps": self.training_steps,
            "avg_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            "total_decisions": self.decision_count,
            "avg_crisis_level": np.mean(crisis_levels),
            "crisis_std": np.std(crisis_levels),
            "bcell_usage": bcell_usage,
            "bcell_diversity_stats": bcell_diversity_stats,
            "bcell_diversity_index": diversity_index,
            "memory_size": len(self.memory.memories),
            "recent_performance": (
                np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
            ),
        }

        return summary

    def save_system(self, base_path: str) -> bool:
        """전체 시스템을 단일 파일로 저장"""
        try:
            # 저장 디렉토리 생성 보장
            base_dir = os.path.dirname(base_path)
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)

            # 전체 시스템을 하나의 딕셔너리로 통합
            system_data = {
                "tcell": {
                    "detector": self.tcell.detector if self.tcell.is_fitted else None,
                    "scaler": self.tcell.scaler if self.tcell.is_fitted else None,
                    "contamination": self.tcell.contamination,
                    "sensitivity": self.tcell.sensitivity,
                    "is_fitted": self.tcell.is_fitted,
                },
                "bcells": {},
                "memory": {
                    "memories": list(self.memory.memories),
                    "embeddings": list(self.memory.embeddings),
                    "capacity": self.memory.capacity,
                    "embedding_dim": self.memory.embedding_dim,
                    "similarity_threshold": self.memory.similarity_threshold,
                },
                "system_config": {
                    "n_assets": self.n_assets,
                    "state_dim": self.state_dim,
                    "training_steps": self.training_steps,
                    "decision_count": self.decision_count,
                },
            }

            # B-Cell 데이터 추가 (SAC 버전)
            for name, bcell in self.bcells.items():
                system_data["bcells"][name] = {
                    "actor_state_dict": bcell.actor.state_dict(),
                    "critic1_state_dict": bcell.critic1.state_dict(),
                    "critic2_state_dict": bcell.critic2.state_dict(),
                    "target_critic1_state_dict": bcell.target_critic1.state_dict(),
                    "target_critic2_state_dict": bcell.target_critic2.state_dict(),
                    "log_alpha": bcell.log_alpha,
                    "target_entropy": bcell.target_entropy,
                    "risk_type": bcell.risk_type,
                    "update_count": bcell.update_count,
                }

            # 단일 파일로 저장
            import pickle

            with open(f"{base_path}.pkl", "wb") as f:
                pickle.dump(system_data, f)

            self.logger.info(f"면역 시스템이 저장되었습니다: {base_path}.pkl")
            return True

        except Exception as e:
            self.logger.error(f"시스템 저장 실패: {e}")
            return False

    def load_system(self, base_path: str) -> bool:
        """전체 시스템 로드"""
        try:
            # T-Cell 로드
            tcell_path = f"{base_path}_tcell.pkl"
            self.tcell.load_model(tcell_path)

            # B-Cell 로드
            for name, bcell in self.bcells.items():
                bcell_path = f"{base_path}_bcell_{name}.pth"
                bcell.load_model(bcell_path)

            # Memory 로드
            memory_path = f"{base_path}_memory.pkl"
            self.memory.load_memory(memory_path)

            self.is_trained = True
            self.logger.info(f"면역 시스템이 로드되었습니다: {base_path}")
            return True

        except Exception as e:
            self.logger.error(f"시스템 로드 실패: {e}")
            return False

    def _get_selection_reason(self, selected_bcell: str, crisis_info) -> str:
        """B-Cell 선택 이유 설명 생성"""

        if isinstance(crisis_info, dict):
            crisis_level = crisis_info.get("overall_crisis", 0)
            crisis_types = []

            if crisis_info.get("volatility_crisis", 0) > 0.5:
                crisis_types.append("high volatility")
            if crisis_info.get("correlation_crisis", 0) > 0.5:
                crisis_types.append("correlation shift")
            if crisis_info.get("volume_crisis", 0) > 0.5:
                crisis_types.append("volume anomaly")
        else:
            crisis_level = crisis_info
            crisis_types = []

        if crisis_level > 0.7:
            crisis_desc = "High crisis detected"
        elif crisis_level > 0.4:
            crisis_desc = "Moderate crisis detected"
        else:
            crisis_desc = "Normal market conditions"

        type_desc = f" ({', '.join(crisis_types)})" if crisis_types else ""

        strategy_reasons = {
            "volatility": f"High volatility strategy selected for unstable market{type_desc}",
            "correlation": f"Correlation strategy selected for regime change{type_desc}",
            "momentum": f"Momentum strategy selected for trending market{type_desc}",
            "defensive": f"Defensive strategy selected for risk management{type_desc}",
            "growth": f"Growth strategy selected for opportunity capture{type_desc}",
        }

        return strategy_reasons.get(selected_bcell, f"{crisis_desc}{type_desc}")

    def _get_top_holdings(self, weights: np.ndarray, top_n: int = 5) -> List[str]:
        """상위 보유 종목 리스트 반환"""

        if hasattr(self, "symbols") and len(self.symbols) == len(weights):
            # 가중치가 높은 순으로 정렬
            sorted_indices = np.argsort(weights)[::-1]
            return [self.symbols[i] for i in sorted_indices[:top_n]]
        else:
            # 심볼이 없으면 인덱스로 표시
            sorted_indices = np.argsort(weights)[::-1]
            return [f"Asset_{i}" for i in sorted_indices[:top_n]]
