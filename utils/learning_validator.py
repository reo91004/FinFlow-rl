# utils/learning_validator.py

import torch
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time

from utils.logger import BIPDLogger

class LearningValidator:
    """학습 진행 상태 검증 시스템"""
    
    def __init__(self):
        self.logger = BIPDLogger().get_validation_logger()
        
        # 검증 기록
        self.episode_times = []
        self.learning_events = 0
        self.weight_updates = 0
        self.gradient_flow_checks = 0
        
        # 성능 추적
        self.episode_start_time = None
        self.total_episodes = 0
        
        # 주기적 요약을 위한 통계
        self.summary_interval = 25  # 25 에피소드마다 요약
        self.last_summary_episode = 0
        
    def start_episode_timing(self):
        """에피소드 타이밍 시작"""
        self.episode_start_time = time.time()
        
    def end_episode_timing(self):
        """에피소드 타이밍 종료 및 검증"""
        if self.episode_start_time is None:
            return
            
        episode_duration = time.time() - self.episode_start_time
        self.episode_times.append(episode_duration)
        self.total_episodes += 1
        
        # 실행 시간 검증 (축약)
        if episode_duration < 10.0:  # 10초 미만
            self.logger.warning(f"에피소드 {self.total_episodes} 실행 시간 빠름: {episode_duration:.2f}초")
            
        # 10 에피소드마다만 요약
        if self.total_episodes % 10 == 0:
            avg_time = np.mean(self.episode_times[-10:]) if len(self.episode_times) >= 10 else np.mean(self.episode_times)
            self.logger.info(f"에피소드 {self.total_episodes} 완료 (최근 평균: {avg_time:.2f}초)")
        
        # 주기적 검증 요약 (25 에피소드마다)
        if self.total_episodes % self.summary_interval == 0 and self.total_episodes > 0:
            self._log_periodic_summary()
        
    def validate_gradient_flow(self, model: torch.nn.Module, 
                              loss: torch.Tensor,
                              model_name: str = "Unknown") -> bool:
        """그라디언트 흐름 검증"""
        self.gradient_flow_checks += 1
        
        # 역전파 전 가중치 저장
        old_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                old_params[name] = param.data.clone()
        
        # 역전파 수행
        loss.backward(retain_graph=True)
        
        # 그라디언트 존재 여부 확인
        total_grad_norm = 0.0
        grad_count = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm
                grad_count += 1
                
        if grad_count == 0:
            self.logger.error(f"{model_name}: 그라디언트가 전혀 계산되지 않음!")
            return False
            
        avg_grad_norm = total_grad_norm / grad_count
        
        if avg_grad_norm < 1e-8:
            self.logger.warning(f"{model_name}: 그라디언트가 너무 작음 (평균: {avg_grad_norm:.2e})")
            return False
            
        self.logger.debug(f"{model_name}: 그라디언트 흐름 정상 (평균 노름: {avg_grad_norm:.6f})")
        return True
        
    def validate_weight_updates(self, model: torch.nn.Module,
                               old_state_dict: Dict,
                               model_name: str = "Unknown") -> bool:
        """가중치 업데이트 검증"""
        new_state_dict = model.state_dict()
        
        total_change = 0.0
        param_count = 0
        
        for key in old_state_dict.keys():
            if key in new_state_dict:
                old_param = old_state_dict[key]
                new_param = new_state_dict[key]
                
                change = torch.sum(torch.abs(new_param - old_param)).item()
                total_change += change
                param_count += 1
                
        if param_count == 0:
            self.logger.error(f"{model_name}: 파라미터가 없음!")
            return False
            
        avg_change = total_change / param_count
        
        if avg_change < 1e-8:
            self.logger.error(f"{model_name}: 가중치 업데이트 없음! (평균 변화: {avg_change:.2e})")
            return False
        elif avg_change < 1e-6:
            self.logger.warning(f"{model_name}: 가중치 변화가 매우 작음 (평균: {avg_change:.2e})")
            
        self.weight_updates += 1
        # 100번마다만 요약 로그
        if self.weight_updates % 100 == 0:
            self.logger.debug(f"가중치 업데이트 요약: 총 {self.weight_updates}회")
        return True
        
    def validate_episode_data(self, episode_data, episode_features=None) -> Dict:
        """에피소드 데이터 검증"""
        validation_result = {
            "data_valid": False,
            "features_valid": False,
            "issues": []
        }
        
        # 데이터 크기 검증
        if episode_data is None or len(episode_data) == 0:
            validation_result["issues"].append("에피소드 데이터 없음")
            return validation_result
            
        if len(episode_data) < 100:
            validation_result["issues"].append(f"에피소드 데이터가 너무 짧음: {len(episode_data)}일")
            
        # 데이터 품질 검증
        if episode_data.isnull().any().any():
            null_count = episode_data.isnull().sum().sum()
            validation_result["issues"].append(f"결측값 발견: {null_count}개")
            
        # 가격 데이터 검증 (음수 가격 체크)
        if (episode_data <= 0).any().any():
            validation_result["issues"].append("음수 또는 0 가격 데이터 발견")
            
        # 변동성 검증 (너무 낮거나 높은 변동성)
        returns = episode_data.pct_change().dropna()
        if len(returns) > 0:
            volatility = returns.std().mean()
            if volatility < 0.001:
                validation_result["issues"].append(f"변동성이 너무 낮음: {volatility:.6f}")
            elif volatility > 0.5:
                validation_result["issues"].append(f"변동성이 너무 높음: {volatility:.6f}")
                
        validation_result["data_valid"] = len(validation_result["issues"]) == 0
        
        # 특성 데이터 검증
        if episode_features is not None:
            if len(episode_features) == 0:
                validation_result["issues"].append("특성 데이터 없음")
            elif np.isnan(episode_features).any():
                validation_result["issues"].append("특성 데이터에 NaN 값 발견")
            elif len(episode_features) != 12:
                validation_result["issues"].append(f"특성 데이터 크기 불일치: {len(episode_features)} != 12")
            else:
                validation_result["features_valid"] = True
        else:
            validation_result["features_valid"] = True  # 특성이 없어도 일단 유효
            
        if validation_result["issues"]:
            self.logger.warning(f"에피소드 검증 문제: {validation_result['issues']}")
        else:
            self.logger.debug("에피소드 데이터 검증 통과")
            
        return validation_result
        
    def validate_realistic_timing(self) -> bool:
        """현실적인 실행 시간 검증"""
        if len(self.episode_times) < 5:
            return True  # 충분한 데이터 없음
            
        avg_time = np.mean(self.episode_times)
        
        if avg_time < 10.0:
            self.logger.error(f"에피소드 평균 실행 시간이 비정상적으로 빠름: {avg_time:.2f}초")
            self.logger.error("이는 학습이 실제로 이루어지지 않고 있음을 시사")
            return False
            
        self.logger.info(f"에피소드 평균 실행 시간: {avg_time:.2f}초 (정상)")
        return True
        
    def log_learning_event(self, event_type: str, details: str = ""):
        """학습 이벤트 기록 (100번마다 요약)"""
        self.learning_events += 1
        # 100번마다만 요약 로그
        if self.learning_events % 100 == 0:
            self.logger.debug(f"학습 이벤트 요약: 총 {self.learning_events}회 (최근: {event_type})")
        
    def get_validation_summary(self) -> Dict:
        """검증 결과 요약"""
        return {
            "total_episodes": self.total_episodes,
            "learning_events": self.learning_events,
            "weight_updates": self.weight_updates,
            "gradient_checks": self.gradient_flow_checks,
            "avg_episode_time": np.mean(self.episode_times) if self.episode_times else 0.0,
            "min_episode_time": np.min(self.episode_times) if self.episode_times else 0.0,
            "max_episode_time": np.max(self.episode_times) if self.episode_times else 0.0,
            "realistic_timing": self.validate_realistic_timing()
        }
        
    def log_summary(self):
        """검증 결과 요약 로깅"""
        summary = self.get_validation_summary()
        
        self.logger.info("=== 학습 검증 요약 ===")
        self.logger.info(f"총 에피소드: {summary['total_episodes']}")
        self.logger.info(f"학습 이벤트: {summary['learning_events']}")
        self.logger.info(f"가중치 업데이트: {summary['weight_updates']}")
        self.logger.info(f"그라디언트 검사: {summary['gradient_checks']}")
        self.logger.info(f"평균 에피소드 시간: {summary['avg_episode_time']:.2f}초")
        self.logger.info(f"현실적 타이밍: {'통과' if summary['realistic_timing'] else '실패'}")
        
        if not summary['realistic_timing']:
            self.logger.error("학습이 제대로 이루어지지 않는 것으로 판단됨!")
    
    def _log_periodic_summary(self):
        """주기적 검증 요약 로깅"""
        recent_episodes = self.summary_interval
        recent_times = self.episode_times[-recent_episodes:] if len(self.episode_times) >= recent_episodes else self.episode_times
        
        if recent_times:
            avg_time = np.mean(recent_times)
            min_time = np.min(recent_times)
            max_time = np.max(recent_times)
            
            # 최근 간격의 학습 이벤트와 가중치 업데이트 계산
            recent_learning_events = max(0, self.learning_events - getattr(self, '_last_learning_events', 0))
            recent_weight_updates = max(0, self.weight_updates - getattr(self, '_last_weight_updates', 0))
            
            self.logger.debug("=" * 50)
            self.logger.debug(f"검증 요약 (에피소드 {self.total_episodes-recent_episodes+1}-{self.total_episodes})")
            self.logger.debug("=" * 50)
            self.logger.debug(f"평균 실행시간: {avg_time:.2f}초 (범위: {min_time:.2f}-{max_time:.2f})")
            self.logger.debug(f"학습 이벤트: {recent_learning_events}회")
            self.logger.debug(f"가중치 업데이트: {recent_weight_updates}회")
            self.logger.debug(f"그라디언트 검사: {self.gradient_flow_checks}회 (누적)")
            
            # 이전 값 저장
            self._last_learning_events = self.learning_events
            self._last_weight_updates = self.weight_updates
            
# 전역 검증자 인스턴스
learning_validator = LearningValidator()