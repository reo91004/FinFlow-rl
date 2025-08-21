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

# 강화학습 하이퍼파라미터 (최적화된 설정)
ACTOR_LR = float(3e-4)  # 1e-4 → 3e-4 (학습 속도 향상)
CRITIC_LR = float(3e-4)  # 1e-4 → 3e-4 (학습 속도 향상)
ALPHA_LR = float(3e-4)  # 1e-4 → 3e-4 (엔트로피 자동 조정 강화)
GAMMA = float(0.99)
TAU = float(0.001)  # 0.005 → 0.001 (타겟 네트워크 안정성 향상)
BATCH_SIZE = int(32)  # 64 → 32 (더 빠른 학습)
BUFFER_SIZE = int(10000)
UPDATE_FREQUENCY = int(4)

# 탐험-활용
EPSILON_START = float(0.9)
EPSILON_END = float(0.05)
EPSILON_DECAY = float(0.995)

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

# 초기 탐험 강화 설정 (신규)
INITIAL_EXPLORATION_STEPS = int(1000)  # 초기 랜덤 탐험 스텝
MIN_BUFFER_SIZE = int(500)  # 최소 버퍼 크기 (학습 시작 조건)

# 보상 설정 (간소화된 로그 수익률 기반)
VOLATILITY_PENALTY_WEIGHT = float(2.0)  # 변동성 페널티 가중치
CONCENTRATION_THRESHOLD = float(0.5)  # 집중도 임계값
CONCENTRATION_PENALTY_WEIGHT = float(1.0)  # 집중도 페널티 가중치
REWARD_CLIP_MIN = float(-5.0)  # 보상 최소값
REWARD_CLIP_MAX = float(5.0)  # 보상 최대값

# 위기 임계값
CRISIS_HIGH = float(0.7)
CRISIS_MEDIUM = float(0.4)
