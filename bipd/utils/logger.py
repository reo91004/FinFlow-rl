# bipd/utils/logger.py

import logging
import os
import datetime
from config import LOGS_DIR

# 전역 타임스탬프 및 디렉토리 (모든 로거가 같은 폴더 사용)
_GLOBAL_TIMESTAMP = None
_GLOBAL_SESSION_DIR = None


def get_global_timestamp():
    global _GLOBAL_TIMESTAMP
    if _GLOBAL_TIMESTAMP is None:
        _GLOBAL_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _GLOBAL_TIMESTAMP


def get_session_directory():
    """Logs/타임스탬프/ 폴더 생성 및 반환"""
    global _GLOBAL_SESSION_DIR
    if _GLOBAL_SESSION_DIR is None:
        timestamp = get_global_timestamp()
        _GLOBAL_SESSION_DIR = os.path.join(LOGS_DIR, timestamp)
        os.makedirs(_GLOBAL_SESSION_DIR, exist_ok=True)
        
        # 시각화 하위 디렉토리 도 생성
        os.makedirs(os.path.join(_GLOBAL_SESSION_DIR, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(_GLOBAL_SESSION_DIR, "models"), exist_ok=True)
        
    return _GLOBAL_SESSION_DIR


class BIPDLogger:
    """BIPD 시스템 전용 로거"""

    def __init__(self, name="BIPD", level=logging.DEBUG, console_level=logging.WARNING):
        self.name = name
        self.level = level
        self.console_level = console_level
        self.logger = None
        self._setup_logger()

    def _setup_logger(self):
        """로거 설정"""
        # 로그 디렉토리 생성
        os.makedirs(LOGS_DIR, exist_ok=True)

        # 로거 생성
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)

        # 중복 핸들러 방지 - 기존 핸들러 모두 제거
        if self.logger.handlers:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)

        # 부모 로거로의 전파 방지 (중복 출력 방지)
        self.logger.propagate = False

        # 포맷터 설정
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-17s | %(message)s",
            datefmt="%H:%M:%S",
        )

        # 콘솔 핸들러 (WARNING 이상만 출력)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 파일 핸들러 (모든 로그 저장) - 세션 디렉토리 사용
        session_dir = get_session_directory()
        log_file = os.path.join(session_dir, "bipd_training.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 초기화 메시지 (첫 번째 로거만)
        if self.name == "Main":
            timestamp = get_global_timestamp()
            self.logger.info(f"=== BIPD 시스템 로깅 시작 ===")
            self.logger.info(f"로그 파일: {log_file}")
            self.logger.info(f"타임스탬프: {timestamp}")
            self.logger.info("=" * 50)

    def info(self, message):
        """정보 로그"""
        self.logger.info(message)

    def debug(self, message):
        """디버그 로그"""
        self.logger.debug(message)

    def warning(self, message):
        """경고 로그"""
        self.logger.warning(message)

    def error(self, message):
        """오류 로그"""
        self.logger.error(message)

    def log_episode(
        self, episode, reward, portfolio_return, crisis_level, selected_bcell
    ):
        """에피소드 정보 로그"""
        self.info(
            f"에피소드 {episode}: "
            f"보상={reward:.4f}, "
            f"수익률={portfolio_return:.4f}, "
            f"위기수준={crisis_level:.3f}, "
            f"선택된전략={selected_bcell}"
        )

    def log_training_summary(self, episode, avg_reward, avg_return, epsilon):
        """학습 요약 로그"""
        self.info(
            f"[학습 요약] 에피소드 {episode}: "
            f"평균보상={avg_reward:.4f}, "
            f"평균수익률={avg_return:.4f}, "
            f"탐험률={epsilon:.3f}"
        )

    def log_system_decision(
        self, crisis_level, selected_strategy, memory_count, weights
    ):
        """시스템 의사결정 로그"""
        max_weight = weights.max()
        self.debug(
            f"[의사결정] 위기수준={crisis_level:.3f}, "
            f"선택전략={selected_strategy}, "
            f"기억활용={memory_count}개, "
            f"최대가중치={max_weight:.3f}"
        )
