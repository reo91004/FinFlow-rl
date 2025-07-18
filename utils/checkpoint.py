# utils/checkpoint.py

import os
import pickle
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings


class CheckpointManager:
    """모델 체크포인트 관리"""

    def __init__(self, checkpoint_dir, save_interval=100, max_checkpoints=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.episode_count = 0

    def should_save_checkpoint(self):
        """체크포인트 저장 여부 결정"""
        self.episode_count += 1
        return self.episode_count % self.save_interval == 0

    def save_checkpoint(
        self, immune_system, curriculum_manager=None, episode_info=None
    ):
        """체크포인트 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = (
                self.checkpoint_dir
                / f"checkpoint_ep{self.episode_count}_{timestamp}.pkl"
            )

            checkpoint_data = {
                "episode": self.episode_count,
                "timestamp": timestamp,
                "episode_info": episode_info or {},
                "immune_system_state": self._extract_immune_system_state(immune_system),
                "curriculum_state": self._extract_curriculum_state(curriculum_manager),
            }

            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)

            print(f"체크포인트 저장: {checkpoint_path.name}")

            # 오래된 체크포인트 정리
            self._cleanup_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            warnings.warn(f"체크포인트 저장 실패: {e}")
            return None

    def load_checkpoint(self, checkpoint_path):
        """체크포인트 로드"""
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)

            print(f"체크포인트 로드: {checkpoint_path}")
            return checkpoint_data

        except Exception as e:
            warnings.warn(f"체크포인트 로드 실패: {e}")
            return None

    def _extract_immune_system_state(self, immune_system):
        """면역 시스템 상태 추출"""
        state = {
            "n_assets": immune_system.n_assets,
            "use_learning_bcells": immune_system.use_learning_bcells,
            "crisis_level": getattr(immune_system, "crisis_level", 0.0),
            "immune_activation": getattr(immune_system, "immune_activation", 0.0),
            "current_weights": getattr(immune_system, "current_weights", None),
            "previous_weights": getattr(immune_system, "previous_weights", None),
        }

        # B-Cell 네트워크 상태 저장
        if immune_system.use_learning_bcells:
            bcell_states = []
            for bcell in immune_system.bcells:
                if hasattr(bcell, "actor_network"):
                    bcell_state = {
                        "cell_id": bcell.cell_id,
                        "risk_type": bcell.risk_type,
                        "actor_state_dict": bcell.actor_network.state_dict(),
                        "critic_state_dict": bcell.critic_network.state_dict(),
                        "specialization_strength": getattr(
                            bcell, "specialization_strength", 0.1
                        ),
                        "epsilon": getattr(bcell, "epsilon", 0.3),
                    }
                    bcell_states.append(bcell_state)
            state["bcell_states"] = bcell_states

        # 기억 세포 상태 저장
        if hasattr(immune_system, "memory_cell"):
            memory_stats = immune_system.memory_cell.get_memory_statistics()
            state["memory_stats"] = memory_stats

        return state

    def _extract_curriculum_state(self, curriculum_manager):
        """커리큘럼 상태 추출"""
        if curriculum_manager is None:
            return None

        try:
            return {
                "current_level": curriculum_manager.scheduler.current_level,
                "current_episode": curriculum_manager.scheduler.current_episode,
                "level_transitions": curriculum_manager.scheduler.level_transitions,
                "episode_rewards": list(curriculum_manager.scheduler.episode_rewards)[
                    -100:
                ],  # 최근 100개만
                "training_history": curriculum_manager.training_history[
                    -50:
                ],  # 최근 50개만
            }
        except Exception as e:
            warnings.warn(f"커리큘럼 상태 추출 실패: {e}")
            return None

    def _cleanup_old_checkpoints(self):
        """오래된 체크포인트 정리"""
        try:
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
            if len(checkpoints) > self.max_checkpoints:
                # 수정 시간 기준으로 정렬
                checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                # 오래된 파일들 삭제
                for old_checkpoint in checkpoints[self.max_checkpoints :]:
                    old_checkpoint.unlink()
                    print(f"오래된 체크포인트 삭제: {old_checkpoint.name}")

        except Exception as e:
            warnings.warn(f"체크포인트 정리 실패: {e}")

    def get_latest_checkpoint(self):
        """가장 최근 체크포인트 경로 반환"""
        try:
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
            if checkpoints:
                latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
                return latest
            return None
        except Exception:
            return None
