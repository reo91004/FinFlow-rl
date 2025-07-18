# utils/logger.py

import sys
import os
from datetime import datetime
from pathlib import Path


class TeeOutput:
    """터미널 출력을 파일에도 동시에 저장"""

    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 즉시 파일에 쓰기

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.close()
            sys.stdout = self.terminal


def setup_logging(output_dir):
    """로깅 설정 - 터미널 출력을 파일로도 저장"""

    log_file_path = os.path.join(output_dir, "training.log")

    # 기존 로그 파일이 있으면 백업
    if os.path.exists(log_file_path):
        backup_path = f"{log_file_path}.backup_{datetime.now().strftime('%H%M%S')}"
        os.rename(log_file_path, backup_path)

    # TeeOutput으로 stdout 리다이렉트
    tee = TeeOutput(log_file_path)
    sys.stdout = tee

    print(f"로깅 시작: {log_file_path}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    return tee


def stop_logging(tee_output):
    """로깅 종료"""
    if tee_output:
        print("-" * 80)
        print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        tee_output.close()
