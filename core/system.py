# bipd/core/system.py

import numpy as np
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
    
    def __init__(self, n_assets: int, state_dim: int):
        self.n_assets = n_assets
        self.state_dim = state_dim
        
        # T-Cell (위기 감지)
        self.tcell = TCell(
            contamination=TCELL_CONTAMINATION,
            sensitivity=TCELL_SENSITIVITY,
            random_state=GLOBAL_SEED
        )
        
        # B-Cell 전문가 그룹 (위험 유형별 특화)
        self.bcells = {
            'volatility': BCell('volatility', state_dim, n_assets, ACTOR_LR, CRITIC_LR),     # 고변동성 시장
            'correlation': BCell('correlation', state_dim, n_assets, ACTOR_LR, CRITIC_LR),   # 상관관계 변화
            'momentum': BCell('momentum', state_dim, n_assets, ACTOR_LR, CRITIC_LR),         # 모멘텀 추세
            'defensive': BCell('defensive', state_dim, n_assets, ACTOR_LR, CRITIC_LR),       # 방어적 전략
            'growth': BCell('growth', state_dim, n_assets, ACTOR_LR, CRITIC_LR)              # 성장 중심
        }
        
        # Memory Cell (경험 저장 및 회상)
        self.memory = MemoryCell(
            capacity=MEMORY_CAPACITY,
            embedding_dim=EMBEDDING_DIM,
            similarity_threshold=0.7
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
            'guidance_applied': 0,
            'total_guidance_attempts': 0,
            'confidence_sum': 0.0,
            'last_report_step': 0
        }
        
        # B-Cell 성과 추적 시스템
        self.bcell_performance = {
            name: {
                'recent_rewards': [],  # 최근 10회 성과
                'consecutive_selections': 0,
                'total_selections': 0
            } for name in self.bcells.keys()
        }
        
        self.logger = BIPDLogger("ImmuneSystem")
        
        self.logger.info(
            f"면역 포트폴리오 시스템이 초기화되었습니다. "
            f"자산수={n_assets}, 상태차원={state_dim}, "
            f"B-Cell={len(self.bcells)}개"
        )
    
    def fit_tcell(self, historical_features: np.ndarray) -> bool:
        """T-Cell 학습 (정상 시장 패턴)"""
        success = self.tcell.fit(historical_features)
        if success:
            self.logger.info("T-Cell 학습이 완료되었습니다.")
        return success
    
    def decide(self, state: np.ndarray, training: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        포트폴리오 의사결정
        
        Args:
            state: [market_features(12), crisis_level(1), prev_weights(n_assets)]
            training: 훈련 모드 여부
            
        Returns:
            weights: 포트폴리오 가중치
            info: 의사결정 상세 정보
        """
        try:
            # 상태 분해
            market_features = state[:FEATURE_DIM]
            crisis_level = state[FEATURE_DIM]
            prev_weights = state[FEATURE_DIM + 1:]
            
            # T-Cell 위기 감지 (재검증)
            detected_crisis = self.tcell.detect_crisis(market_features)
            
            # 더 정확한 위기 수준 사용
            final_crisis_level = max(crisis_level, detected_crisis)
            
            # B-Cell 선택 (위기 수준 기반)
            selected_bcell_name = self._select_bcell(final_crisis_level)
            selected_bcell = self.bcells[selected_bcell_name]
            
            # Memory 회상
            memory_guidance = self.memory.get_memory_guidance(
                market_features, final_crisis_level, k=MEMORY_K
            )
            
            # 포트폴리오 가중치 생성
            weights = selected_bcell.get_action(state, deterministic=not training)
            
            # Memory 가이던스 적용 (있는 경우)
            self.memory_stats['total_guidance_attempts'] += 1
            if (memory_guidance['has_guidance'] and 
                memory_guidance['confidence'] > 0.8 and
                memory_guidance['recommended_action'] is not None):
                
                memory_weight = min(0.3, memory_guidance['confidence'] - 0.5)
                weights = (1 - memory_weight) * weights + memory_weight * memory_guidance['recommended_action']
                weights = weights / weights.sum()  # 정규화
                
                # 통계 업데이트
                self.memory_stats['guidance_applied'] += 1
                self.memory_stats['confidence_sum'] += memory_guidance['confidence']
                
                # 간헐적 로깅 (매 100회마다)
                if self.memory_stats['guidance_applied'] % 100 == 0:
                    avg_confidence = self.memory_stats['confidence_sum'] / self.memory_stats['guidance_applied']
                    usage_rate = self.memory_stats['guidance_applied'] / self.memory_stats['total_guidance_attempts']
                    self.logger.debug(
                        f"Memory 가이던스 통계 (최근 100회): 사용률={usage_rate:.1%}, "
                        f"평균 신뢰도={avg_confidence:.3f}"
                    )
            
            # 의사결정 정보 수집
            decision_info = {
                'crisis_level': float(final_crisis_level),
                'selected_bcell': selected_bcell_name,
                'memory_guidance': memory_guidance['has_guidance'],
                'memory_confidence': memory_guidance.get('confidence', 0.0),
                'specialization_scores': {
                    name: bcell.get_specialization_score(final_crisis_level)
                    for name, bcell in self.bcells.items()
                },
                'weights_concentration': float(np.sum(weights ** 2)),
                'decision_count': self.decision_count
            }
            
            # 의사결정 히스토리 저장
            self.decision_history.append({
                'step': self.decision_count,
                'crisis_level': final_crisis_level,
                'selected_bcell': selected_bcell_name,
                'weights': weights.copy(),
                'memory_used': memory_guidance['has_guidance']
            })
            
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
            
        except Exception as e:
            self.logger.error(f"의사결정 실패: {e}")
            # 폴백: 균등 가중치
            uniform_weights = np.ones(self.n_assets) / self.n_assets
            fallback_info = {
                'crisis_level': 0.0,
                'selected_bcell': 'fallback',
                'memory_guidance': False,
                'error': str(e)
            }
            return uniform_weights, fallback_info
    
    def _select_bcell(self, crisis_level: float) -> str:
        """
        위기 수준에 따른 B-Cell 선택 (다양성 확보)
        
        성능 기반 동적 페널티 + 확률적 선택을 통한 편향성 해결
        """
        # 각 B-Cell의 기본 전문성 점수 계산
        scores = {}
        for name, bcell in self.bcells.items():
            base_score = bcell.get_specialization_score(crisis_level)
            
            # 최근 성과 기반 페널티 계산
            performance_data = self.bcell_performance[name]
            recent_rewards = performance_data['recent_rewards']
            
            if len(recent_rewards) > 5:
                avg_reward = np.mean(recent_rewards)
                # 성과가 나쁠수록 페널티 적용
                performance_penalty = max(0, -avg_reward * 0.5)
                base_score -= performance_penalty
            
            # 연속 선택 페널티 (강제 순환)
            consecutive_count = performance_data['consecutive_selections']
            if consecutive_count >= 5:  # 5회 연속 선택시
                base_score *= 0.3  # 점수 대폭 감소
                self.logger.debug(f"{name} 전략이 {consecutive_count}회 연속 선택되어 페널티 적용")
            
            scores[name] = base_score
        
        # 확률적 선택 (완전 greedy 방지)
        if np.random.random() < 0.2:  # 20% 확률로 랜덤 선택
            selected = np.random.choice(list(self.bcells.keys()))
            self.logger.debug(f"탐험적 랜덤 선택: {selected}")
        else:
            # 최고 점수 전략 선택
            selected = max(scores, key=scores.get)
        
        # 선택 통계 업데이트
        self._update_selection_stats(selected)
        
        return selected
    
    def _update_selection_stats(self, selected_bcell: str) -> None:
        """B-Cell 선택 통계 업데이트"""
        # 선택된 B-Cell 통계 업데이트
        self.bcell_performance[selected_bcell]['total_selections'] += 1
        
        # 연속 선택 카운트 업데이트
        if (self.decision_history and 
            self.decision_history[-1]['selected_bcell'] == selected_bcell):
            self.bcell_performance[selected_bcell]['consecutive_selections'] += 1
        else:
            self.bcell_performance[selected_bcell]['consecutive_selections'] = 1
        
        # 다른 B-Cell들의 연속 선택 카운트 리셋
        for name, data in self.bcell_performance.items():
            if name != selected_bcell:
                data['consecutive_selections'] = 0
    
    def _get_recent_rewards(self, bcell_name: str, window: int = 10) -> list:
        """특정 B-Cell의 최근 성과 반환"""
        return self.bcell_performance[bcell_name]['recent_rewards'][-window:]
    
    def _get_consecutive_count(self, bcell_name: str) -> int:
        """특정 B-Cell의 연속 선택 횟수 반환"""
        return self.bcell_performance[bcell_name]['consecutive_selections']
    
    def update(self, state: np.ndarray, action: np.ndarray, reward: float, 
              next_state: np.ndarray, done: bool) -> None:
        """
        시스템 업데이트 (학습)
        
        Args:
            state: 현재 상태
            action: 선택된 행동 (포트폴리오 가중치)
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        try:
            # 상태 분해
            market_features = state[:FEATURE_DIM]
            crisis_level = state[FEATURE_DIM]
            
            # Memory에 경험 저장
            self.memory.store(
                state=market_features,
                action=action,
                reward=reward,
                crisis_level=crisis_level,
                additional_info={
                    'step': self.training_steps,
                    'done': done
                }
            )
            
            # B-Cell 업데이트 (모든 전문가 학습)
            for bcell in self.bcells.values():
                bcell.store_experience(state, action, reward, next_state, done)
                
                # 주기적 학습
                if self.training_steps % UPDATE_FREQUENCY == 0:
                    bcell.update()
            
            # 성과 히스토리 업데이트
            self.performance_history.append({
                'step': self.training_steps,
                'reward': reward,
                'crisis_level': crisis_level
            })
            
            # 마지막 선택된 B-Cell의 성과 추적 업데이트
            if self.decision_history:
                last_decision = self.decision_history[-1]
                selected_bcell = last_decision['selected_bcell']
                
                # 최근 성과 리스트에 추가 (최대 10개 유지)
                recent_rewards = self.bcell_performance[selected_bcell]['recent_rewards']
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
                
        except Exception as e:
            self.logger.error(f"시스템 업데이트 실패: {e}")
    
    def get_system_explanation(self, state: np.ndarray) -> Dict:
        """
        현재 의사결정에 대한 종합 설명 (XAI)
        """
        try:
            market_features = state[:FEATURE_DIM]
            crisis_level = state[FEATURE_DIM]
            
            # T-Cell 설명
            tcell_explanation = self.tcell.get_anomaly_explanation(market_features)
            
            # 선택된 B-Cell 설명
            selected_bcell_name = self._select_bcell(crisis_level)
            bcell_explanation = self.bcells[selected_bcell_name].get_explanation(state)
            
            # Memory 통계
            memory_stats = self.memory.get_statistics()
            
            # 전체 설명
            explanation = {
                'system_overview': {
                    'training_steps': self.training_steps,
                    'decision_count': self.decision_count,
                    'tcell_fitted': self.tcell.is_fitted
                },
                'crisis_detection': tcell_explanation,
                'strategy_selection': {
                    'selected_strategy': selected_bcell_name,
                    'selection_reason': f'위기 수준 {crisis_level:.3f}에 최적화됨',
                    'all_specialization_scores': {
                        name: bcell.get_specialization_score(crisis_level)
                        for name, bcell in self.bcells.items()
                    }
                },
                'portfolio_generation': bcell_explanation,
                'memory_system': memory_stats,
                'recent_decisions': self.decision_history[-5:] if self.decision_history else []
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"시스템 설명 생성 실패: {e}")
            return {'error': str(e)}
    
    def get_performance_summary(self) -> Dict:
        """성과 요약 통계"""
        if not self.performance_history:
            return {'message': '성과 데이터가 없습니다.'}
        
        rewards = [p['reward'] for p in self.performance_history]
        crisis_levels = [p['crisis_level'] for p in self.performance_history]
        
        # B-Cell 사용 통계
        bcell_usage = {}
        for decision in self.decision_history:
            bcell = decision['selected_bcell']
            bcell_usage[bcell] = bcell_usage.get(bcell, 0) + 1
        
        # B-Cell 다양성 통계
        bcell_diversity_stats = {}
        total_selections = sum(self.bcell_performance[name]['total_selections'] 
                             for name in self.bcells.keys())
        
        for name, data in self.bcell_performance.items():
            recent_rewards = data['recent_rewards']
            selection_rate = data['total_selections'] / max(total_selections, 1)
            
            bcell_diversity_stats[name] = {
                'total_selections': data['total_selections'],
                'selection_rate': selection_rate,
                'consecutive_selections': data['consecutive_selections'],
                'avg_recent_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
                'recent_performance_count': len(recent_rewards)
            }
        
        # 다양성 지수 계산 (Shannon Entropy)
        diversity_index = 0.0
        if total_selections > 0:
            for name in self.bcells.keys():
                selection_rate = self.bcell_performance[name]['total_selections'] / total_selections
                if selection_rate > 0:
                    diversity_index -= selection_rate * np.log(selection_rate)
        
        summary = {
            'training_steps': self.training_steps,
            'avg_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'total_decisions': self.decision_count,
            'avg_crisis_level': np.mean(crisis_levels),
            'crisis_std': np.std(crisis_levels),
            'bcell_usage': bcell_usage,
            'bcell_diversity_stats': bcell_diversity_stats,
            'bcell_diversity_index': diversity_index,
            'memory_size': len(self.memory.memories),
            'recent_performance': np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        }
        
        return summary
    
    def save_system(self, base_path: str) -> bool:
        """전체 시스템 저장"""
        try:
            # T-Cell 저장
            tcell_path = f"{base_path}_tcell.pkl"
            self.tcell.save_model(tcell_path)
            
            # B-Cell 저장
            for name, bcell in self.bcells.items():
                bcell_path = f"{base_path}_bcell_{name}.pth"
                bcell.save_model(bcell_path)
            
            # Memory 저장
            memory_path = f"{base_path}_memory.pkl"
            self.memory.save_memory(memory_path)
            
            self.logger.info(f"면역 시스템이 저장되었습니다: {base_path}")
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