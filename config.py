# bipd/config.py

import numpy as np
import torch
import os

# 시드 설정
GLOBAL_SEED = 42


# Device 설정 (GPU 사용 가능시 자동 선택)
def get_best_device():
    """사용 가능한 최적의 디바이스 자동 선택"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_best_device()


def set_seed(seed=GLOBAL_SEED):
    """재현 가능한 결과를 위한 시드 설정"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 멀티 GPU 지원
        # CUDA 최적화 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     # MPS는 별도의 시드 설정이 필요하지 않음
    #     # torch.manual_seed로 충분
    #     pass


def get_device_info():
    """현재 사용 중인 device 정보 반환"""
    if DEVICE.type == "cuda":
        return f"CUDA GPU: {torch.cuda.get_device_name(0)} (Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB)"
    # elif DEVICE.type == "mps":
    #     return "Apple MPS GPU (Metal Performance Shaders)"
    else:
        return "CPU"


# 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# 데이터 설정
SYMBOLS = [
    "MMM",
    "AXP",
    "AMGN",
    "AMZN",
    "AAPL",
    "BA",
    "CAT",
    "CVX",
    "CSCO",
    "KO",
    "DIS",
    "GS",
    "HD",
    "HON",
    "IBM",
    "JNJ",
    "JPM",
    "MCD",
    "MRK",
    "MSFT",
    "NKE",
    "NVDA",
    "PG",
    "CRM",
    "SHW",
    "TRV",
    "UNH",
    "VZ",
    "V",
    "WMT",
]
TRAIN_START = "2008-01-01"
TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"
TEST_END = "2024-12-31"

# 시장 특성
FEATURE_DIM = 12
LOOKBACK_WINDOW = 20

# T-Cell 설정 (위기 감지)
TCELL_CONTAMINATION = 0.1  # Isolation Forest 이상치 비율
TCELL_SENSITIVITY = 1.0  # 위기 감지 민감도

# B-Cell 설정 (강화학습)
STATE_DIM = FEATURE_DIM + 1 + len(SYMBOLS)  # features + crisis + prev_weights
ACTION_DIM = len(SYMBOLS)  # 포트폴리오 가중치
HIDDEN_DIM = 128

# 강화학습 하이퍼파라미터
ACTOR_LR = float(3e-4)
CRITIC_LR = float(1e-4)  # Phase 4: 안정성을 위해 감소
ALPHA_LR = float(3e-4)
GAMMA = float(0.99)
TAU = float(0.001)
BATCH_SIZE = int(32)
BUFFER_SIZE = int(10000)
UPDATE_FREQUENCY = int(4)

# SAC 엔트로피 설정 (개선된 안정화 파라미터)
TARGET_ENTROPY_SCALE = float(0.25)  # 기존 호환성 유지

# SAC 안정화 파라미터 (Phase 1: 핵심 안정화)
ALPHA_MIN = float(1e-4)  # 온도 최소값
ALPHA_MAX = float(1.0)   # 온도 최대값
CONCENTRATION_MIN = float(0.2)   # Dirichlet 농도 파라미터 최소값
CONCENTRATION_MAX = float(100.0) # Dirichlet 농도 파라미터 최대값
WEIGHT_EPSILON = float(1e-6)     # 포트폴리오 가중치 보호 임계값

# 그래디언트 안정화
MAX_GRAD_NORM = float(1.0)  # 그래디언트 클리핑 최대 norm
HUBER_DELTA = float(1.0)    # Huber loss delta 파라미터

# CQL 정규화 (Phase 2)
CQL_ALPHA = float(0.1)      # CQL 정규화 강도

# Memory Cell 설정
MEMORY_CAPACITY = int(500)
EMBEDDING_DIM = int(32)
MEMORY_K = int(5)

# 환경 설정
INITIAL_CAPITAL = int(1000000)
TRANSACTION_COST = float(0.001)
MAX_STEPS = int(252)  # 1년 거래일

# 학습 설정
N_EPISODES = int(500)
LOG_INTERVAL = int(10)
SAVE_INTERVAL = int(100)

# 초기 탐험 설정
INITIAL_EXPLORATION_STEPS = int(1000)
MIN_BUFFER_SIZE = int(500)

# 보상 함수 설정
SHARPE_WINDOW = int(10)
SHARPE_SCALE = float(0.15)
REWARD_BUFFER_SIZE = int(1000)  # 3-sigma 필터링용
REWARD_OUTLIER_SIGMA = float(3.0)
REWARD_CLIP_MIN = float(-2.0)
REWARD_CLIP_MAX = float(2.0)

# 위기 임계값
CRISIS_HIGH = float(0.7)
CRISIS_MEDIUM = float(0.4)
