"""
로깅 유틸리티 모듈

로깅 시스템 설정 및 관리 기능을 제공합니다.
파일 및 콘솔 출력에 대한 다양한 로깅 레벨과 포맷을 설정하는 함수를 포함합니다.
"""

import logging
import sys
import os

def setup_logger(run_dir):
    """
    지정된 실행 디렉토리 내에 로그 파일을 생성하도록 로깅 시스템을 설정합니다.
    파일 핸들러는 INFO 레벨 이상, 콘솔 핸들러는 WARNING 레벨 이상만 출력합니다.
    
    Args:
        run_dir (str): 로그 파일을 포함할 실행별 결과 디렉토리 경로.
        
    Returns:
        logging.Logger: 설정된 로거 객체
    """
    # os.makedirs(log_dir, exist_ok=True) # main에서 생성하므로 제거
    log_file = os.path.join(run_dir, "training.log")  # 로그 파일 경로 수정

    logger = logging.getLogger("PortfolioRL")
    logger.setLevel(logging.DEBUG)

    # 기존 핸들러 제거
    if logger.hasHandlers():
        logger.handlers.clear()

    # 파일 핸들러 설정 (INFO 레벨 이상, 모든 정보 기록)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)  # 파일에는 INFO 레벨 이상 기록
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러 설정 (WARNING 레벨 이상, 간략 정보)
    console_formatter = logging.Formatter(
        "%(levelname)s: %(message)s"
    )  # 레벨 이름 포함
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # 콘솔에는 WARNING 레벨 이상만 출력
    console_handler.setFormatter(console_formatter)
    # console_handler.addFilter(lambda record: not record.getMessage().startswith('=== 에피소드')) # 필터 제거, 레벨로 제어
    logger.addHandler(console_handler)

    # logger.info -> logger.debug 로 변경 (초기화 메시지는 파일에만 기록)
    logger.debug(f"로거 초기화 완료. 로그 파일: {log_file}")
    return logger


def print_memory_stats(logger):
    """현재 GPU 메모리 사용량 및 캐시 상태를 로깅합니다."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(
            f"GPU 메모리 사용량: {allocated:.2f} MB / 예약됨: {reserved:.2f} MB"
        ) 