# utils/logger.py

import sys
import os
import logging
from datetime import datetime
from pathlib import Path


class TeeOutput:
    """터미널 출력을 파일에도 동시에 저장하면서 logging 시스템과 통합"""

    def __init__(self, file_path, logger=None):
        self.terminal = sys.stdout
        self.terminal_err = sys.stderr
        self.log_file = open(file_path, "w", encoding="utf-8")
        self.logger = logger
        self._encoding = getattr(sys.stdout, "encoding", "utf-8")

    def write(self, message):
        # 터미널에는 원본 그대로 출력 (tqdm 진행률 막대 포함)
        self.terminal.write(message)
        self.terminal.flush()  # 즉시 플러시로 tqdm 시각화 개선

        # 로그 파일에도 출력 기록 (기존 로직 유지)
        if message and message.strip():
            # tqdm 진행률 표시줄 처리 (기존 로직)
            if "\r" in message and not message.endswith("\n"):
                clean_message = message.replace("\r", "").strip()
                if clean_message and ("100%" in clean_message or "%|" in clean_message):
                    percentage = None
                    if "%" in clean_message:
                        try:
                            parts = clean_message.split("%")
                            if parts:
                                percentage = float(parts[0].split()[-1])
                        except:
                            pass

                    # 10% 단위로만 기록 (기존 로직)
                    if percentage is None or percentage % 10 == 0 or percentage >= 99:
                        # 터미널 제어 문자 제거 후 기록
                        clean_for_log = self._clean_terminal_chars(clean_message)
                        self.log_file.write(clean_for_log + "\n")
                        # logging 시스템에도 기록
                        if self.logger:
                            self.logger.debug(f"Progress: {clean_for_log}")
            else:
                # 일반 메시지는 터미널 제어 문자 제거 후 기록
                clean_message = self._clean_terminal_chars(message)
                self.log_file.write(clean_message)

                # logging 시스템에도 기록 (레벨 자동 감지)
                if self.logger and clean_message.strip():
                    self._log_to_system(clean_message.strip())

            self.log_file.flush()

    def _clean_terminal_chars(self, message):
        """터미널 제어 문자 제거 (로그 파일용)"""
        import re

        # ANSI 이스케이프 시퀀스 제거
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", message)

    def _log_to_system(self, message):
        """메시지 내용에 따라 적절한 logging 레벨로 기록"""
        if not self.logger:
            return

        message_lower = message.lower()

        # 오류 관련 키워드 감지
        if any(
            keyword in message_lower
            for keyword in ["error", "오류", "failed", "실패", "exception"]
        ):
            self.logger.error(message)
        # 경고 관련 키워드 감지
        elif any(keyword in message_lower for keyword in ["warning", "경고", "warn"]):
            self.logger.warning(message)
        # 중요 정보 키워드 감지
        elif any(
            keyword in message_lower
            for keyword in ["완료", "complete", "성공", "success", "저장", "saved"]
        ):
            self.logger.info(message)
        # 기타는 debug 레벨
        else:
            self.logger.debug(message)

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

    @property
    def encoding(self):
        """tqdm이 인코딩 정보를 올바르게 감지하도록 지원"""
        return self._encoding

    def fileno(self):
        """tqdm이 파일 디스크립터를 올바르게 감지하도록 지원"""
        try:
            return self.terminal.fileno()
        except:
            return 1  # stdout의 기본 file descriptor


def setup_logging(output_dir):
    """
    통합 로깅 시스템 설정
    - 파일 핸들러: INFO 레벨 이상, 상세 정보 기록
    - 콘솔 핸들러: WARNING 레벨 이상, 중요 정보만 출력
    - TeeOutput: 모든 print 출력을 파일에도 저장
    """

    # 로그 파일 경로 설정
    log_file_path = os.path.join(output_dir, "training.log")

    # 기존 로그 파일이 있으면 백업
    if os.path.exists(log_file_path):
        backup_path = f"{log_file_path}.backup_{datetime.now().strftime('%H%M%S')}"
        os.rename(log_file_path, backup_path)

    # 표준 logging 시스템 설정
    logger = logging.getLogger("BIPD_System")
    logger.setLevel(logging.DEBUG)

    # 기존 핸들러 제거
    if logger.hasHandlers():
        logger.handlers.clear()

    # 파일 핸들러 설정 (INFO 레벨 이상, 상세 정보)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    file_handler = logging.FileHandler(
        os.path.join(output_dir, "system.log"), encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러 설정 (WARNING 레벨 이상, 간략 정보)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # TeeOutput으로 stdout, stderr 리다이렉트 (기존 기능 유지)
    tee = TeeOutput(log_file_path, logger)
    sys.stdout = tee
    sys.stderr = tee

    # 초기 로그 메시지
    logger.info(f"BIPD 시스템 로깅 초기화 완료")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"통합 로그 파일: {log_file_path}")
    logger.info(f"시스템 로그 파일: {os.path.join(output_dir, 'system.log')}")

    print(f"로깅 시작: {log_file_path}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    return tee, logger


def stop_logging(tee_output, logger=None):
    """로깅 종료"""
    if tee_output:
        print("-" * 80)
        print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        tee_output.close()

    if logger:
        logger.info("BIPD 시스템 로깅 종료")
        # 핸들러 정리
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def get_logger(name="BIPD_System"):
    """설정된 로거 인스턴스 반환"""
    return logging.getLogger(name)


def log_memory_stats(logger=None):
    """GPU 메모리 사용량 로깅"""
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2

            memory_msg = (
                f"GPU 메모리 사용량: {allocated:.2f} MB / 예약됨: {reserved:.2f} MB"
            )

            if logger:
                logger.info(memory_msg)
            else:
                print(f"[메모리] {memory_msg}")
        else:
            if logger:
                logger.debug("GPU를 사용할 수 없습니다.")
    except ImportError:
        if logger:
            logger.debug("PyTorch를 사용할 수 없습니다.")


def log_system_info(logger=None):
    """시스템 정보 로깅"""
    import platform
    import psutil

    try:
        system_info = {
            "OS": f"{platform.system()} {platform.release()}",
            "Python": platform.python_version(),
            "CPU": f"{psutil.cpu_count()} cores",
            "Memory": f"{psutil.virtual_memory().total / 1024**3:.1f} GB",
        }

        # GPU 정보 추가
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                system_info["GPU"] = gpu_name
        except ImportError:
            pass

        info_msg = "시스템 정보: " + ", ".join(
            [f"{k}: {v}" for k, v in system_info.items()]
        )

        if logger:
            logger.info(info_msg)
        else:
            print(f"[시스템] {info_msg}")

    except Exception as e:
        if logger:
            logger.warning(f"시스템 정보 수집 실패: {e}")


def log_checkpoint_info(checkpoint_path, logger=None):
    """체크포인트 저장 정보 로깅"""
    checkpoint_msg = f"체크포인트 저장: {checkpoint_path}"

    if logger:
        logger.info(checkpoint_msg)
    else:
        print(f"[체크포인트] {checkpoint_msg}")


def log_performance_metrics(metrics_dict, logger=None):
    """성과 지표 로깅"""
    metrics_msg = "성과 지표: " + ", ".join(
        [
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics_dict.items()
        ]
    )

    if logger:
        logger.info(metrics_msg)
    else:
        print(f"[성과] {metrics_msg}")


# 편의 함수들 (기존 코드와의 호환성 유지)
def setup_logging_legacy(output_dir):
    """기존 방식과 호환되는 래퍼 함수"""
    tee, logger = setup_logging(output_dir)
    return tee


def stop_logging_legacy(tee_output):
    """기존 방식과 호환되는 래퍼 함수"""
    stop_logging(tee_output)
