# core/system.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import traceback
from typing import Dict, Tuple, List, Optional, Any, Union
from agents import TCell, BCell, MemoryCell
from utils.logger import BIPDLogger
from config import *


class SerializationUtils:
    """직렬화 및 로깅 강화 유틸리티"""
    
    @staticmethod
    def safe_tensor_to_python(obj: Any) -> Any:
        """텐서와 numpy 배열을 Python 네이티브 타입으로 안전하게 변환"""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: SerializationUtils.safe_tensor_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [SerializationUtils.safe_tensor_to_python(item) for item in obj]
        else:
            return obj
    
    @staticmethod
    def safe_json_serialize(obj: Any, max_length: int = 1000) -> str:
        """복잡한 객체를 안전하게 JSON으로 직렬화"""
        try:
            safe_obj = SerializationUtils.safe_tensor_to_python(obj)
            json_str = json.dumps(safe_obj, ensure_ascii=False, separators=(',', ':'))
            if len(json_str) > max_length:
                return json_str[:max_length-3] + "..."
            return json_str
        except Exception as e:
            return f"<직렬화 실패: {type(obj).__name__}({str(e)[:50]})>"
    
    @staticmethod
    def safe_log_rewards(rewards: Any, logger: BIPDLogger, context: str = "") -> str:
        """보상 구조를 안전하게 로깅"""
        prefix = f"[{context}] " if context else ""
        
        if isinstance(rewards, torch.Tensor):
            shape_info = f"Tensor{list(rewards.shape)}"
            if rewards.numel() <= 10:
                values = rewards.detach().cpu().numpy().tolist()
                return f"{prefix}보상: {shape_info} = {values}"
            else:
                stats = {
                    'mean': float(rewards.mean().item()),
                    'std': float(rewards.std().item()),
                    'min': float(rewards.min().item()),
                    'max': float(rewards.max().item())
                }
                return f"{prefix}보상: {shape_info} stats={stats}"
                
        elif isinstance(rewards, dict):
            try:
                summary = {}
                for k, v in rewards.items():
                    if isinstance(v, (int, float)):
                        summary[k] = round(v, 4)
                    elif isinstance(v, dict) and 'reward' in v:
                        summary[k] = round(v['reward'], 4)
                    else:
                        summary[k] = type(v).__name__
                return f"{prefix}보상: {SerializationUtils.safe_json_serialize(summary, 200)}"
            except:
                return f"{prefix}보상: Dict[{len(rewards)} keys]"
                
        elif isinstance(rewards, (list, tuple)):
            if len(rewards) <= 10:
                safe_list = SerializationUtils.safe_tensor_to_python(rewards)
                return f"{prefix}보상: {safe_list}"
            else:
                return f"{prefix}보상: List[{len(rewards)} items]"
        else:
            return f"{prefix}보상: {type(rewards).__name__}"
    
    @staticmethod
    def safe_exception_log(e: Exception, logger: BIPDLogger, context: str = "") -> None:
        """예외를 안전하게 로깅"""
        prefix = f"[{context}] " if context else ""
        error_msg = f"{prefix}예외 발생: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        
        # 스택 트레이스 간략화 (중요한 부분만)
        tb_lines = traceback.format_exc().split('\n')
        relevant_lines = [line for line in tb_lines if any(keyword in line for keyword in 
                         ['core/', 'agents/', 'trainer.py', 'system.py', 'bcell.py'])][:5]
        if relevant_lines:
            logger.debug(f"{prefix}관련 스택 트레이스: {' | '.join(relevant_lines)}")


class LoggingEnhancer:
    """로깅 강화 유틸리티"""
    
    def __init__(self, logger: BIPDLogger, context: str = "System"):
        self.logger = logger
        self.context = context
        self.call_count = 0
    
    def debug_with_context(self, message: str, data: Any = None) -> None:
        """컨텍스트와 데이터를 포함한 디버그 로깅"""
        self.call_count += 1
        prefix = f"[{self.context}#{self.call_count:04d}]"
        
        if data is not None:
            data_str = SerializationUtils.safe_json_serialize(data, 150)
            self.logger.debug(f"{prefix} {message} | 데이터: {data_str}")
        else:
            self.logger.debug(f"{prefix} {message}")
    
    def log_method_call(self, method_name: str, args: Dict[str, Any] = None, result: Any = None) -> None:
        """메서드 호출과 결과를 안전하게 로깅"""
        prefix = f"[{self.context}#{self.call_count:04d}]"
        
        if args:
            args_str = SerializationUtils.safe_json_serialize(args, 100)
            self.logger.debug(f"{prefix} {method_name}() 호출 | 인자: {args_str}")
        
        if result is not None:
            result_str = SerializationUtils.safe_json_serialize(result, 100)
            self.logger.debug(f"{prefix} {method_name}() 결과: {result_str}")
    
    def log_tensor_stats(self, tensor: torch.Tensor, name: str) -> None:
        """텐서 통계를 안전하게 로깅"""
        if tensor is None:
            self.logger.debug(f"[{self.context}] {name}: None")
            return
            
        try:
            stats = {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'device': str(tensor.device),
                'requires_grad': tensor.requires_grad
            }
            
            if tensor.numel() > 0:
                stats.update({
                    'mean': float(tensor.mean().item()),
                    'std': float(tensor.std().item()),
                    'min': float(tensor.min().item()),
                    'max': float(tensor.max().item())
                })
            
            self.logger.debug(f"[{self.context}] {name}: {SerializationUtils.safe_json_serialize(stats, 150)}")
            
        except Exception as e:
            self.logger.debug(f"[{self.context}] {name}: <텐서 통계 실패: {e}>")


class GateNetwork(nn.Module):
    """
    MoE 게이팅 네트워크 - 전문가 선택을 위한 신경망
    
    시장 특성을 입력받아 전문가 선택 로짓을 출력 (미분 가능한 구조)
    """
    
    def __init__(self, in_dim: int, num_experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, num_experts)   # logits만 출력
        )
    
    def forward(self, x):
        return self.net(x)                # [B, K] logits


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
                "volatility", state_dim, n_assets, LR_ACTOR, LR_CRITIC, ALPHA_LR
            ),  # 고변동성 시장
            "correlation": BCell(
                "correlation", state_dim, n_assets, LR_ACTOR, LR_CRITIC, ALPHA_LR
            ),  # 상관관계 변화
            "momentum": BCell(
                "momentum", state_dim, n_assets, LR_ACTOR, LR_CRITIC, ALPHA_LR
            ),  # 모멘텀 추세
            "defensive": BCell(
                "defensive", state_dim, n_assets, LR_ACTOR, LR_CRITIC, ALPHA_LR
            ),  # 방어적 전략
            "growth": BCell(
                "growth", state_dim, n_assets, LR_ACTOR, LR_CRITIC, ALPHA_LR
            ),  # 성장 중심
        }

        # Memory Cell (경험 저장 및 회상)
        self.memory = MemoryCell(
            capacity=MEMORY_CAPACITY,
            embedding_dim=EMBEDDING_DIM,
            similarity_threshold=0.7,
        )

        # MoE 게이팅 네트워크 초기화
        self.bcell_names = list(self.bcells.keys())
        self.n_experts = len(self.bcells)
        
        # 게이팅 네트워크 입력 차원: 시장 특성(12D) 만 사용 (단순화)
        gate_input_dim = 12  # market features only
        self.gate_network = GateNetwork(
            in_dim=gate_input_dim,
            num_experts=self.n_experts
        ).to(DEVICE)
        
        # 게이팅 네트워크 상태 초기화
        self.gate_temp = float(GATE_TEMP_INIT)
        self.gate_baseline = torch.zeros(self.n_experts, device=DEVICE)  # EMA baseline per expert
        self.gate_optimizer = torch.optim.Adam(self.gate_network.parameters(), lr=GATE_LR)
        self.gate_network.train()
        
        # 상태 저장을 위한 변수 (게이팅 네트워크 입력용)
        self.last_state_tensor = None
        
        # MoE 상태 추적
        self.current_expert_idx = 0
        self.expert_dwell_count = 0
        self.min_dwell_steps = SWITCH_DWELL_STEPS if 'SWITCH_DWELL_STEPS' in globals() else 5

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
        
        # 로깅 강화 유틸리티 초기화
        self.logging_enhancer = LoggingEnhancer(self.logger, "ImmuneSystem")
        self.serialization_utils = SerializationUtils()

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
        
        # 게이팅 네트워크용 상태 저장 (시장 특성만)
        self.last_state_tensor = torch.tensor(market_features[:12], dtype=torch.float32)

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

        # B-Cell 선택 (MoE 게이팅 네트워크 기반)
        selected_bcell_name = self._select_bcell(market_features, crisis_info)
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

    def _select_bcell(self, state_features: np.ndarray, crisis_info) -> str:
        """
        MoE 게이팅 네트워크를 통한 B-Cell 선택
        
        Args:
            state_features: 시장 상태 특성 (12차원)
            crisis_info: T-Cell 위기 정보
            
        Returns:
            str: 선택된 B-Cell 이름
        """
        # 시장 특성 텐서 변환 (12차원만 사용)
        market_features = torch.FloatTensor(state_features[:12]).unsqueeze(0).to(DEVICE)
        
        # 게이팅 네트워크로 전문가 확률 계산 (추론 시에는 no_grad 사용)
        with torch.no_grad():
            gate_logits = self.gate_network(market_features)  # 로짓만 반환
            gate_probs = torch.softmax(gate_logits, dim=-1)   # 확률 계산
        
        gate_probs_np = gate_probs.cpu().numpy().flatten()
        
        # Top-1 게이팅: 최고 확률 전문가 선택 (minimum dwell time 고려)
        if self.expert_dwell_count >= self.min_dwell_steps:
            # 최소 거주 시간 만족시 새로운 전문가 선택 가능
            best_expert_idx = np.argmax(gate_probs_np)
            
            if best_expert_idx != self.current_expert_idx:
                # 전문가 변경
                self.current_expert_idx = best_expert_idx
                self.expert_dwell_count = 1
                self.logging_stats["expert_switches"] = self.logging_stats.get("expert_switches", 0) + 1
            else:
                # 같은 전문가 유지
                self.expert_dwell_count += 1
        else:
            # 최소 거주 시간 미만시 현재 전문가 유지
            self.expert_dwell_count += 1
        
        selected_name = self.bcell_names[self.current_expert_idx]
        
        # 선택 통계 업데이트
        self._update_selection_stats(selected_name)
        
        # 게이팅 네트워크 성능 기록 (나중에 학습용)
        self._record_gating_decision(gate_logits.cpu().numpy(), selected_name)
        
        # 주기적 로깅
        decisions_since_last_report = (
            self.decision_count - self.logging_stats["last_log_report"]
        )
        if decisions_since_last_report >= self.logging_stats["log_interval"]:
            self._log_moe_statistics(gate_probs_np)
            self.logging_stats["last_log_report"] = self.decision_count
        
        return selected_name

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

        # MoE B-Cell 업데이트 (선택된 전문가만 학습)
        selected_expert_name = self.bcell_names[self.current_expert_idx]
        selected_bcell = self.bcells[selected_expert_name]
        
        # 선택된 전문가만 경험 저장 및 업데이트
        selected_bcell.store_experience(state, action, reward, next_state, done)
        
        # 주기적 학습 (선택된 전문가만)
        if self.training_steps % UPDATE_FREQUENCY == 0:
            selected_bcell.update()
            
            # 선택되지 않은 전문가들은 주기적 EMA만 수행 (선택사항)
            if self.training_steps % (UPDATE_FREQUENCY * 5) == 0:
                for name, bcell in self.bcells.items():
                    if name != selected_expert_name:
                        # 타겟 네트워크만 부드럽게 동기화 (과도한 divergence 방지)
                        bcell._soft_update_targets(tau=0.001)

        # 성과 히스토리 업데이트
        self.performance_history.append(
            {
                "step": self.training_steps,
                "reward": reward,
                "crisis_level": crisis_level,
            }
        )

        # 현재 선택된 B-Cell의 성과 추적 업데이트 (MoE 정합성)
        selected_expert_name = self.bcell_names[self.current_expert_idx]
        
        # 최근 성과 리스트에 추가 (최대 10개 유지)
        recent_rewards = self.bcell_performance[selected_expert_name]["recent_rewards"]
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

    # ===== MoE 게이팅 시스템 헬퍼 메소드 =====
    
    def _record_gating_decision(self, gate_logits: np.ndarray, selected_expert: str) -> None:
        """게이팅 네트워크 결정 기록 (훈련용)"""
        if not hasattr(self, 'gating_history'):
            self.gating_history = []
        
        self.gating_history.append({
            'logits': gate_logits.flatten(),
            'selected_expert': selected_expert,
            'expert_idx': self.bcell_names.index(selected_expert),
            'step': self.decision_count
        })
        
        # 히스토리 크기 제한
        if len(self.gating_history) > 1000:
            self.gating_history.pop(0)

    def _log_moe_statistics(self, gate_probs: np.ndarray) -> None:
        """MoE 게이팅 통계 로깅"""
        interval = self.logging_stats["log_interval"]
        expert_switches = self.logging_stats.get("expert_switches", 0)
        
        # 현재 전문가 확률 분포
        prob_str = ", ".join([f"{name}:{prob:.2f}" for name, prob in zip(self.bcell_names, gate_probs)])
        
        # 현재 전문가와 거주 시간
        current_expert = self.bcell_names[self.current_expert_idx]
        
        self.logger.debug(
            f"MoE 통계 (최근 {interval}회): "
            f"현재 전문가={current_expert} (거주:{self.expert_dwell_count}회), "
            f"전환횟수={expert_switches}회, "
            f"확률분포=[{prob_str}]"
        )
        
        # 통계 리셋
        self.logging_stats["expert_switches"] = 0

    def _normalize_expert_rewards(self, expert_rewards):
        """
        다양한 보상 스키마를 안전하게 정규화
        
        지원하는 입력 형태:
        - List[float]
        - List[Dict]: {'reward': float, 'parts': Dict}
        - Dict[str, float]
        - Dict[str, Dict]: {'reward': float, 'parts': Dict}
        
        Returns:
            rewards_tensor: torch.FloatTensor [K] on device (no grad)
            parts_list: Optional[List[Dict]] aligned to bcell_names (or None)
        """
        K = len(self.bcell_names)
        parts_list = None

        # Case 1: List[...] (floats or dicts)
        if isinstance(expert_rewards, (list, tuple)):
            if len(expert_rewards) != K:
                # 길이 불일치 → 최선의 정렬 시도 또는 건너뛰기
                self.logger.warning(f"[Gate] Reward length {len(expert_rewards)} != num_experts {K}. Best-effort alignment.")
            
            # dict 경로 시도
            if len(expert_rewards) > 0 and isinstance(expert_rewards[0], dict):
                rewards = []
                parts_list = []
                for i, item in enumerate(expert_rewards[:K]):
                    r = float(item.get("reward", 0.0))
                    rewards.append(r)
                    parts_list.append(item.get("parts", {}))
                # 필요시 패딩
                while len(rewards) < K:
                    rewards.append(0.0); parts_list.append({})
            else:
                # float로 가정
                rewards = [float(x) for x in expert_rewards[:K]]
                while len(rewards) < K:
                    rewards.append(0.0)
            rewards_tensor = torch.tensor(rewards, device=DEVICE, dtype=torch.float32)
            return rewards_tensor, parts_list

        # Case 2: Dict[...] 전문가 이름으로 키 구성
        if isinstance(expert_rewards, dict):
            mapping = expert_rewards
            rewards = []
            parts_list = []
            for name in self.bcell_names:
                val = mapping.get(name, 0.0)
                if isinstance(val, dict):
                    r = float(val.get("reward", 0.0))
                    rewards.append(r)
                    parts_list.append(val.get("parts", {}))
                else:
                    rewards.append(float(val))
                    parts_list.append({})
            rewards_tensor = torch.tensor(rewards, device=DEVICE, dtype=torch.float32)
            return rewards_tensor, parts_list

        # Fallback: 잘못된 스키마 → 제로 텐서
        self.logger.error(f"[Gate] Invalid expert_rewards type: {type(expert_rewards)}; skipping gate update.")
        return torch.zeros(K, device=DEVICE, dtype=torch.float32), None

    def update_gating_network(self, expert_rewards) -> torch.Tensor:
        """
        게이팅 네트워크 업데이트 (미분 가능한 Softmax 기반, 강건한 스키마 정규화)
        
        Args:
            expert_rewards: 다양한 형태의 전문가 보상 (List, Dict 등)
            
        Returns:
            gate loss (tensor with grad).
        """
        try:
            # 입력 보상 안전 로깅
            rewards_log = self.serialization_utils.safe_log_rewards(expert_rewards, self.logger, "GatingUpdate")
            self.logging_enhancer.debug_with_context("게이팅 네트워크 업데이트 시작", {"input_rewards": rewards_log})
            
            self.gate_network.train()

            # 1) 보상 정규화
            rewards, parts_list = self._normalize_expert_rewards(expert_rewards)  # rewards: [K] float tensor (no grad)
            
            # 정규화된 보상 로깅
            self.logging_enhancer.log_tensor_stats(rewards, "정규화된_보상")

            # 조기 종료 조건 (빈 보상, 모든 제로 등)
            if rewards.numel() == 0:
                self.logging_enhancer.debug_with_context("빈 보상으로 인한 조기 종료")
                return torch.tensor(0.0, device=DEVICE)  # detached zero

            # 2) 모델 입력 구성 (last_state_tensor 사용)
            if self.last_state_tensor is None:
                self.logger.warning("[Gate] last_state_tensor is None; skipping gate update.")
                return torch.tensor(0.0, device=DEVICE)
                
            x = self.last_state_tensor.to(DEVICE)            # [feat_dim]; 텐서 확인
            x = x.unsqueeze(0)                               # [1, feat_dim]

            # 3) 순전파로 로짓과 미분 가능한 확률 계산
            logits = self.gate_network(x)                    # [1, K]
            tau = torch.clamp(torch.tensor(self.gate_temp, device=DEVICE), GATE_TEMP_MIN, GATE_TEMP_MAX)
            p = torch.nn.functional.softmax(logits / tau, dim=-1)  # [1, K]
            p = p.squeeze(0)                                 # [K]

            # EMA baseline per expert (detach from graph) - NaN/Inf 안전 처리
            with torch.no_grad():
                # 유한한 보상만 사용하여 베이스라인 업데이트
                finite_mask = torch.isfinite(rewards)
                if finite_mask.any():
                    # 유한한 값만 사용하여 베이스라인 업데이트
                    finite_rewards = torch.where(finite_mask, rewards, self.gate_baseline)
                    self.gate_baseline.mul_(GATE_BASELINE_MOMENTUM).add_(
                        (1.0 - GATE_BASELINE_MOMENTUM) * finite_rewards
                    )
                    
                # 베이스라인에서도 NaN/Inf 제거
                baseline_finite_mask = torch.isfinite(self.gate_baseline)
                if not baseline_finite_mask.all():
                    self.gate_baseline = torch.where(baseline_finite_mask, self.gate_baseline, torch.zeros_like(self.gate_baseline))
                    
            # 어드밴티지 계산 - NaN/Inf 안전 처리
            adv = (rewards - self.gate_baseline).detach()    # [K], no grad on reward path
            
            # 어드밴티지에서 NaN/Inf 제거
            finite_adv_mask = torch.isfinite(adv)
            if not finite_adv_mask.all():
                adv = torch.where(finite_adv_mask, adv, torch.zeros_like(adv))

            # 4) 손실: negative expected advantage - 안전 검증 포함
            expected_adv = (p * adv).sum()                   # scalar, has grad via p->logits->gate params
            
            # 최종 손실 검증
            if not torch.isfinite(expected_adv):
                self.logging_enhancer.debug_with_context("비유한 expected_adv 감지, 제로 손실로 대체")
                expected_adv = torch.tensor(0.0, device=DEVICE)
                
            loss = -expected_adv

            # 선택적: 엔트로피 정규화 on p to avoid collapse - NaN/Inf 안전 처리
            if GATE_ENTROPY_BETA > 0.0:
                ent = -(p * (p.clamp_min(1e-8)).log()).sum()
                if torch.isfinite(ent):
                    loss = loss - GATE_ENTROPY_BETA * ent
                else:
                    self.logging_enhancer.debug_with_context("비유한 엔트로피 감지, 엔트로피 정규화 건너뜀")

            # 5) 역전파 - 최종 안전성 검증
            self.gate_optimizer.zero_grad(set_to_none=True)
            
            # 최종 손실 검증
            if not torch.isfinite(loss):
                self.logging_enhancer.debug_with_context("최종 손실이 비유한, 제로 손실로 대체")
                loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            
            # 안전장치: must require grad
            if loss.requires_grad:
                loss.backward()
            else:
                self.logging_enhancer.debug_with_context("손실이 gradient를 요구하지 않음, 역전파 건너뜀")
            torch.nn.utils.clip_grad_norm_(self.gate_network.parameters(), max_norm=5.0)
            self.gate_optimizer.step()

            # 6) (선택적) 온도 어닐링
            self.gate_temp = float(torch.clamp(tau * 0.9995, GATE_TEMP_MIN, GATE_TEMP_MAX))  # gentle decay
            
            # 성공적인 업데이트 로깅
            loss_value = float(loss.detach().item())
            self.logging_enhancer.debug_with_context("게이팅 네트워크 업데이트 완료", {
                "loss": loss_value, 
                "temperature": self.gate_temp,
                "rewards_stats": self.serialization_utils.safe_tensor_to_python(rewards)
            })

            return loss.detach()
            
        except Exception as e:
            # 예외 안전 로깅
            self.serialization_utils.safe_exception_log(e, self.logger, "GatingUpdate")
            # 안전한 기본값 반환
            return torch.tensor(0.0, device=DEVICE)

    def update_bcell_performance(self, bcell_name: str, reward: float) -> None:
        """B-Cell 성과 업데이트 (기존 호환성 유지)"""
        performance_data = self.bcell_performance[bcell_name]
        performance_data["recent_rewards"].append(reward)

        # 윈도우 크기 제한 (최근 10회 성과만 보관)
        if len(performance_data["recent_rewards"]) > 10:
            performance_data["recent_rewards"].pop(0)
