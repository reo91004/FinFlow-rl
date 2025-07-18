# utils/logger.py

import sys
import os
from datetime import datetime
from pathlib import Path


class TeeOutput:
    """터미널 출력을 파일에도 동시에 저장"""

    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.terminal_err = sys.stderr
        self.log_file = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        # 터미널에는 원본 그대로 출력
        self.terminal.write(message)
        
        # 로그 파일에도 출력 기록 (선택적)
        if message and message.strip():  # 빈 문자열이 아닌 경우
            # tqdm 진행률 표시줄 처리
            if '\r' in message and not message.endswith('\n'):
                # tqdm 진행률: 주요 이정표만 로그에 기록
                clean_message = message.replace('\r', '').strip()
                if clean_message and ('100%' in clean_message or '%|' in clean_message):
                    # 완료률이 있는 경우만 기록 (너무 빈번하지 않게)
                    percentage = None
                    if '%' in clean_message:
                        try:
                            # 진행률 추출
                            parts = clean_message.split('%')
                            if parts:
                                percentage = float(parts[0].split()[-1])
                        except:
                            pass
                    
                    # 10% 단위로만 기록
                    if percentage is None or percentage % 10 == 0 or percentage >= 99:
                        self.log_file.write(clean_message + '\n')
            else:
                # 일반 메시지는 그대로 기록
                self.log_file.write(message)
            
            self.log_file.flush()  # 즉시 파일에 쓰기

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.close()
            sys.stdout = self.terminal
            sys.stderr = self.terminal_err
    
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

    # TeeOutput으로 stdout, stderr 리다이렉트
    tee = TeeOutput(log_file_path)
    sys.stdout = tee
    sys.stderr = tee  # tqdm은 stderr에 출력하므로 추가

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
