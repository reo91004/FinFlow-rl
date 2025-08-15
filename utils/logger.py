# utils/logger.py

import logging
import os
from datetime import datetime
from typing import Optional
import sys

class BIPDLogger:
    """BIPD 시스템 전용 파일 기반 로거"""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            # 로그 디렉토리와 파일은 나중에 설정
            self.log_dir = None
            self.log_file = None
            
            # 예쁜 로그 파일 포맷터
            self.formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
                datefmt='%H:%M:%S'
            )
            
            # 터미널용 최소 포맷터 (핵심 정보만)
            self.console_formatter = logging.Formatter(
                '%(message)s'
            )
    
    def get_logger(self, name: str, level: int = logging.INFO, 
                   console_output: bool = False) -> logging.Logger:
        """특정 모듈용 로거 획득"""
        
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # 로거 자체는 DEBUG 레벨로
        logger.handlers = []  # 기존 핸들러 제거
        logger.propagate = False  # 부모 로거로 전파 방지 (기본 핸들러 방지)
        
        # 로그 파일이 설정되지 않은 경우 비활성화
        if self.log_file is None:
            # 로그 파일이 없으면 로그를 비활성화 (기본 logs 폴더 생성 방지)
            logger.addHandler(logging.NullHandler())
            self._loggers[name] = logger
            return logger
        
        # 파일 핸들러 (모든 로그)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # DEBUG도 파일에 기록
        file_handler.setFormatter(self.formatter)
        logger.addHandler(file_handler)
        
        # 콘솔 핸들러 (중요 정보는 콘솔에도 출력)
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)  # INFO 이상 콘솔 출력
            console_handler.setFormatter(self.console_formatter)
            logger.addHandler(console_handler)
        
        self._loggers[name] = logger
        return logger
    
    @classmethod
    def get_episode_logger(cls) -> logging.Logger:
        """에피소드 실행 전용 로거 (파일만)"""
        instance = cls()
        return instance.get_logger('EPISODE', logging.INFO, console_output=False)
    
    @classmethod
    def get_learning_logger(cls) -> logging.Logger:
        """학습 과정 전용 로거 (파일만)"""
        instance = cls()
        return instance.get_logger('LEARNING', logging.INFO, console_output=False)
    
    @classmethod
    def get_validation_logger(cls) -> logging.Logger:
        """검증 전용 로거 (파일만)"""
        instance = cls()
        return instance.get_logger('VALIDATION', logging.INFO, console_output=False)
    
    @classmethod
    def get_reward_logger(cls) -> logging.Logger:
        """보상 계산 전용 로거 (파일만)"""
        instance = cls()
        return instance.get_logger('REWARD', logging.INFO, console_output=False)
    
    @classmethod
    def get_system_logger(cls) -> logging.Logger:
        """시스템 전반 로거 (콘솔 + 파일)"""
        instance = cls()
        return instance.get_logger('SYSTEM', logging.INFO, console_output=True)
    
    @classmethod
    def get_file_only_logger(cls) -> logging.Logger:
        """파일 전용 로거 (터미널 출력 없음)"""
        instance = cls()
        return instance.get_logger('FILE_ONLY', logging.INFO, console_output=False)
    
    def write_section_header(self, title: str, logger_name: str = 'BIPD'):
        """로그 파일에 섹션 헤더 작성"""
        logger = self.get_logger(logger_name)
        separator = "=" * 80
        logger.debug(separator)
        logger.debug(f" {title}")
        logger.debug(separator)
    
    def write_subsection_header(self, title: str, logger_name: str = 'BIPD'):
        """로그 파일에 서브섹션 헤더 작성"""
        logger = self.get_logger(logger_name)
        separator = "-" * 50
        logger.debug(separator)
        logger.debug(f" {title}")
        logger.debug(separator)
    
    def set_log_directory(self, log_dir: str):
        """로그 디렉토리를 동적으로 설정 (각 실행별 분리를 위해)"""
        # 새로운 로그 디렉토리 생성
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 새로운 로그 파일 경로
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"bipd_{timestamp}.log")
        
        # 기존 로거들의 핸들러를 새로운 파일로 업데이트
        for logger in self._loggers.values():
            # 기존 파일 핸들러만 제거 (콘솔 핸들러는 유지)
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)
                    handler.close()
            
            # 새로운 파일 핸들러 추가
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(self.formatter)
            logger.addHandler(file_handler)

# 전역 로거 인스턴스 (필요 시에만 생성하도록 지연 초기화)
logger_instance = None

def get_global_system_logger():
    """전역 시스템 로거 획득 (지연 초기화)"""
    global logger_instance
    if logger_instance is None:
        logger_instance = BIPDLogger()
    return logger_instance.get_system_logger()