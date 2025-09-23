# src/utils/logger.py

"""
로거: FinFlow 통합 로깅 시스템

목적: 일관된 로깅 및 메트릭 추적
의존: tqdm (진행바 호환)
사용처: 모든 모듈 (전역 로거)
역할: 세션별 로그 관리 및 메트릭 저장

구현 내용:
- 세션별 로그 디렉토리 자동 생성 (YYYYMMDD_HHMMSS)
- tqdm 진행바와 충돌 없는 로깅
- 콘솔 (INFO) + 파일 (DEBUG) 이중 로깅
- metrics.jsonl로 학습 메트릭 추적
- BIPDLogger 별칭으로 하위 호환성 유지
"""

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

def set_session_directory(directory: str):
    """세션 디렉토리를 명시적으로 설정 (evaluate 모드용)"""
    global _GLOBAL_SESSION_DIR
    _GLOBAL_SESSION_DIR = directory
    # 하위 디렉토리는 생성하지 않음 (evaluate 모드에서는 이미 존재)

def get_session_directory(base_dir: str = "logs", create_new: bool = True):
    """세션별 로그 디렉토리 생성 및 반환

    Args:
        base_dir: 기본 로그 디렉토리
        create_new: True일 때만 새 디렉토리 생성 (train 모드)
    """
    global _GLOBAL_SESSION_DIR
    if _GLOBAL_SESSION_DIR is None:
        if not create_new:
            # evaluate 모드에서 새 디렉토리 생성 방지
            raise ValueError("Session directory not set. Use set_session_directory() for evaluate mode.")

        timestamp = get_global_timestamp()
        _GLOBAL_SESSION_DIR = os.path.join(base_dir, timestamp)
        os.makedirs(_GLOBAL_SESSION_DIR, exist_ok=True)

        # 하위 디렉토리 생성 (중복 제거)
        os.makedirs(os.path.join(_GLOBAL_SESSION_DIR, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(_GLOBAL_SESSION_DIR, "checkpoints"), exist_ok=True)  # 모델 체크포인트
        os.makedirs(os.path.join(_GLOBAL_SESSION_DIR, "results"), exist_ok=True)      # 최종 결과 및 보고서
        os.makedirs(os.path.join(_GLOBAL_SESSION_DIR, "tensorboard"), exist_ok=True)

    return _GLOBAL_SESSION_DIR

class FinFlowLogger:
    """FinFlow 시스템 전용 로거 (기존 BIPDLogger 스타일 유지)"""

    def __init__(self, name="FinFlow", level=logging.DEBUG, console_level=logging.INFO, use_file=None):
        self.name = name
        self.level = level
        self.console_level = console_level
        # 환경 변수로 파일 로깅 비활성화 가능 (평가 스크립트용)
        if use_file is None:
            self.use_file = os.environ.get('FINFLOW_NO_FILE_LOG') != '1'
        else:
            self.use_file = use_file
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

        # 파일 핸들러 (세션 디렉토리 사용) - use_file=True일 때만
        if self.use_file:
            session_dir = get_session_directory()
            log_file = os.path.join(session_dir, "finflow_training.log")
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # JSON 로거 추가 (메트릭 추적용)
            self.json_file_path = os.path.join(session_dir, "metrics.jsonl")
        else:
            self.json_file_path = None

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
        if self.json_file_path:  # 파일 경로가 있을 때만 저장
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "step": step,
                **metrics
            }
            with open(self.json_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
    

# Backward compatibility alias
BIPDLogger = FinFlowLogger