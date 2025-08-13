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
            self.log_dir = "logs"
            os.makedirs(self.log_dir, exist_ok=True)
            
            # 단일 로그 파일로 통합
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(self.log_dir, f"bipd_{timestamp}.log")
            
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
        logger.setLevel(level)
        logger.handlers = []  # 기존 핸들러 제거
        
        # 파일 핸들러 (모든 로그)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)  # DEBUG 줄이고 INFO 이상만
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
        """에피소드 실행 전용 로거"""
        instance = cls()
        return instance.get_logger('BIPD', logging.INFO)
    
    @classmethod
    def get_learning_logger(cls) -> logging.Logger:
        """학습 과정 전용 로거"""
        instance = cls()
        return instance.get_logger('BIPD', logging.INFO)
    
    @classmethod
    def get_validation_logger(cls) -> logging.Logger:
        """검증 전용 로거"""
        instance = cls()
        return instance.get_logger('BIPD', logging.INFO)
    
    @classmethod
    def get_reward_logger(cls) -> logging.Logger:
        """보상 계산 전용 로거"""
        instance = cls()
        return instance.get_logger('BIPD', logging.INFO)
    
    @classmethod
    def get_system_logger(cls) -> logging.Logger:
        """시스템 전반 로거"""
        instance = cls()
        return instance.get_logger('BIPD', logging.INFO, console_output=True)
    
    def write_section_header(self, title: str, logger_name: str = 'BIPD'):
        """로그 파일에 섹션 헤더 작성"""
        logger = self.get_logger(logger_name)
        separator = "=" * 80
        logger.info(separator)
        logger.info(f" {title}")
        logger.info(separator)
    
    def write_subsection_header(self, title: str, logger_name: str = 'BIPD'):
        """로그 파일에 서브섹션 헤더 작성"""
        logger = self.get_logger(logger_name)
        separator = "-" * 50
        logger.info(separator)
        logger.info(f" {title}")
        logger.info(separator)

# 전역 로거 인스턴스
logger_instance = BIPDLogger()
system_logger = logger_instance.get_system_logger()