# constant.py

import os
from datetime import datetime

# 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# 디렉토리 생성 함수
def create_directories():
    """필요한 디렉토리들을 생성한다"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def create_timestamped_directory(base_dir, prefix="run"):
    """타임스탬프 기반 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir


# 기본 설정값들
DEFAULT_LOOKBACK = 20
DEFAULT_N_TCELLS = 3
DEFAULT_N_BCELLS = 5
DEFAULT_MEMORY_SIZE = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_UPDATE_FREQUENCY = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPSILON = 0.3
DEFAULT_EPSILON_DECAY = 0.995
DEFAULT_MIN_EPSILON = 0.05

# 임계값 설정
RISK_THRESHOLDS = {"low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9}

# 특성 크기 설정
FEATURE_SIZE = 12
EXPECTED_FEATURES = 12
