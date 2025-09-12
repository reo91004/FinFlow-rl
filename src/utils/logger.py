# src/utils/logger.py

import logging
import os
import datetime
import json
from typing import Optional, Dict, Any

from tqdm import tqdm

class TqdmLoggingHandler(logging.StreamHandler):
    """tqdm 진행바를 방해하지 않는 로깅 핸들러"""
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg, file=self.stream)
        self.flush()

# 전역 타임스탬프 및 디렉토리
_GLOBAL_TIMESTAMP = None
_GLOBAL_SESSION_DIR = None

def get_global_timestamp():
    global _GLOBAL_TIMESTAMP
    if _GLOBAL_TIMESTAMP is None:
        _GLOBAL_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _GLOBAL_TIMESTAMP

def get_session_directory(base_dir: str = "logs"):
    """세션별 로그 디렉토리 생성 및 반환"""
    global _GLOBAL_SESSION_DIR
    if _GLOBAL_SESSION_DIR is None:
        timestamp = get_global_timestamp()
        _GLOBAL_SESSION_DIR = os.path.join(base_dir, timestamp)
        os.makedirs(_GLOBAL_SESSION_DIR, exist_ok=True)
        
        # 하위 디렉토리 생성
        os.makedirs(os.path.join(_GLOBAL_SESSION_DIR, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(_GLOBAL_SESSION_DIR, "models"), exist_ok=True)
        os.makedirs(os.path.join(_GLOBAL_SESSION_DIR, "reports"), exist_ok=True)
        
    return _GLOBAL_SESSION_DIR

class FinFlowLogger:
    """FinFlow 시스템 전용 로거 (기존 BIPDLogger 스타일 유지)"""

    def __init__(self, name="FinFlow", level=logging.DEBUG, console_level=logging.INFO):
        self.name = name
        self.level = level
        self.console_level = console_level
        self.logger = None
        self._setup_logger()

    def _setup_logger(self):
        """로거 설정"""
        # 로거 생성
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)

        # 중복 핸들러 방지
        if self.logger.handlers:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)

        # 부모 로거로의 전파 방지
        self.logger.propagate = False

        # 포맷터 설정
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%H:%M:%S",
        )

        # tqdm 호환 콘솔 핸들러
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 파일 핸들러 (세션 디렉토리 사용)
        session_dir = get_session_directory()
        log_file = os.path.join(session_dir, "finflow_training.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # JSON 로거 추가 (메트릭 추적용)
        self.json_file_path = os.path.join(session_dir, "metrics.jsonl")

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """메트릭을 JSON 형식으로 저장"""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "step": step,
            **metrics
        }
        with open(self.json_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    

# Backward compatibility alias
BIPDLogger = FinFlowLogger