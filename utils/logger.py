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
        # tqdm 진행률 표시줄 처리
        # \r (캐리지 리턴)으로 시작하는 라인도 로그에 기록
        if message.strip():  # 빈 문자열이 아닌 경우만
            # 터미널에는 원본 그대로 출력
            self.terminal.write(message)
            
            # 로그 파일에 기록
            # tqdm의 경우 \r을 \n으로 변환하여 각 업데이트를 새 줄로 기록
            log_message = message
            if '\r' in message and not message.endswith('\n'):
                # tqdm 진행률 표시줄인 경우
                log_message = message.replace('\r', '') + '\n'
            self.log_file.write(log_message)
            self.log_file.flush()  # 즉시 파일에 쓰기
        else:
            # 빈 줄은 터미널에만 출력
            self.terminal.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.close()
            sys.stdout = self.terminal
    
    def isatty(self):
        """tqdm이 터미널 환경을 올바르게 감지하도록 지원"""
        return self.terminal.isatty()


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
