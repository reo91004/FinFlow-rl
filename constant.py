# constant.py
"""
BIPD 시스템 통합 상수 정의
실제 작동하는 강화학습 시스템을 위한 현실적 파라미터 설정
"""

import os
from datetime import datetime


# ===== 파일 시스템 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def create_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def create_timestamped_directory(base_dir, prefix="run"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir


# ===== 데이터 설정 =====
STOCK_SYMBOLS = [
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

TRAIN_START_DATE = "2008-01-02"
TRAIN_END_DATE = "2020-12-31"
TEST_START_DATE = "2021-01-01"
TEST_END_DATE = "2024-12-31"
ANALYSIS_START_DATE = "2021-01-01"
ANALYSIS_END_DATE = "2021-06-30"


# ===== 에이전트 시스템 =====
DEFAULT_N_TCELLS = 3
DEFAULT_N_BCELLS = 5
DEFAULT_MEMORY_SIZE = 20
DEFAULT_LOOKBACK = 20
FEATURE_SIZE = 12
EXPECTED_FEATURES = 12


# ===== 강화학습 핵심 파라미터 (현실화) =====
# 학습률
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_ACTOR_LR = 0.0001
DEFAULT_CRITIC_LR = 0.0002
DEFAULT_ATTENTION_LR = 0.00005
DEFAULT_META_LR = 0.0001
DEFAULT_MEMORY_LR = 0.0001

# RL 파라미터
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.005
DEFAULT_BATCH_SIZE = 256
DEFAULT_UPDATE_FREQUENCY = 1000

# 탐험-활용
DEFAULT_EPSILON = 0.3
DEFAULT_EPSILON_DECAY = 0.995
DEFAULT_MIN_EPSILON = 0.05


# ===== 학습 프로세스 (현실화) =====
EPISODE_LENGTH = 252
PRETRAIN_EPISODES = 5000
TOTAL_EPISODES = 50000
CURRICULUM_TOTAL_EPISODES = 100000

CURRICULUM_MIN_EPISODES = {
    0: 20000,
    1: 30000,
    2: 50000,
}


# ===== Experience Replay 및 Target Network =====
EXPERIENCE_BUFFER_SIZE = 100000
TARGET_UPDATE_FREQUENCY = 100


# ===== 네트워크 아키텍처 =====
ACTOR_HIDDEN_SIZE = 128
CRITIC_HIDDEN_SIZE = 256
BCELL_ACTOR_HIDDEN_SIZE = 64
BCELL_CRITIC_HIDDEN_SIZE = 64
ATTENTION_HIDDEN_SIZE = 64
ATTENTION_HIDDEN_DIM = 32
META_CONTROLLER_HIDDEN_SIZE = 512
META_CONTROLLER_DEFAULT_HIDDEN = 128
MEMORY_EMBEDDING_DIM = 64
MEMORY_HIDDEN_DIM = 64
HIDDEN_LAYER_DIVISOR = 2
ATTENTION_HIDDEN_RATIO = 0.5


# ===== 메모리 버퍼 =====
# B-Cell 버퍼
BCELL_SPECIALIZATION_BUFFER_SIZE = 1000
BCELL_GENERAL_BUFFER_SIZE = 500
BCELL_PERFORMANCE_BUFFER_SIZE = 50
BCELL_DECISION_BUFFER_SIZE = 100

# T-Cell 버퍼
TCELL_TRAINING_BUFFER_SIZE = 200
TCELL_ACTIVATION_BUFFER_SIZE = 200
TCELL_PATTERN_BUFFER_SIZE = 100
TCELL_REWARD_BUFFER_SIZE = 50

# 시스템 레벨 버퍼
MEMORY_PERFORMANCE_BUFFER_SIZE = 50
REWARD_HISTORY_BUFFER_SIZE = 252
REWARD_REGIME_BUFFER_SIZE = 5
REWARD_MEMORY_BUFFER_SIZE = 1000
HIERARCHICAL_EXPERIENCE_BUFFER_SIZE = 1000
HIERARCHICAL_EXPERT_PERFORMANCE_SIZE = 100
HIERARCHICAL_SELECTION_HISTORY_SIZE = 200
CURRICULUM_SITUATION_BUFFER_SIZE = 100


# ===== 임계값 =====
DEFAULT_ACTIVATION_THRESHOLD = 0.5
MEMORY_SIMILARITY_THRESHOLD = 0.8
BASE_RISK_THRESHOLD = 0.5
EXTREME_MARKET_THRESHOLD = 3.0
FEATURE_VALIDATION_THRESHOLD = 100
SPECIALIZATION_STRENGTH_INCREMENT = 0.005

CURRICULUM_RETURN_THRESHOLDS = {
    "high_positive": 0.001,
    "low_negative": -0.001,
    "very_high_positive": 0.002,
    "very_low_negative": -0.002,
}

RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
    "critical": 0.9,
}

ACTIVATION_THRESHOLDS = {
    "tcell": [0.8, 0.6, 0.4, 0.2],
    "bcell": 0.5,
    "memory": 0.3,
}


# ===== 거래 및 비용 =====
DEFAULT_TRANSACTION_COST_RATE = 0.001
COST_PENALTY_MULTIPLIER = 1000
DRAWDOWN_PENALTY_BASE = -10
DRAWDOWN_PENALTY_MULTIPLIER = 100


# ===== 보상 시스템 (단일 클리핑) =====
REWARD_CLIPPING_RANGE = (-5.0, 5.0)
RETURN_CLIPPING_RANGE = (-0.2, 0.2)
RETURN_SCALING_FACTOR = 20
TARGET_VOLATILITY = 0.15

REWARD_COMPONENT_WEIGHTS = {
    "return": 0.4,
    "risk_adjusted": 0.3,
    "target": 0.2,
    "adaptation": 0.1,
}

# 기존 리스트 형태도 호환성 유지
REWARD_COMPONENT_WEIGHTS_LIST = [0.3, 0.3, 0.2, 0.1, 0.05, 0.05]
REWARD_NORMALIZATION_THRESHOLD = 100
CONCENTRATION_PENALTY_THRESHOLD = 0.4


# ===== T-Cell 파라미터 =====
DEFAULT_SENSITIVITY = 0.1
TCELL_ACTIVATION_THRESHOLDS = [0.8, 0.6, 0.4, 0.2]

RISK_BOOST_FACTORS = {
    "volatility": 0.2,
    "stress": 0.15,
    "correlation": 0.1,
}


# ===== Memory Cell 파라미터 =====
MEMORY_DEFAULT_EMBEDDING_DIM = 64
MEMORY_MAX_MEMORIES = 100
MEMORY_DECAY_RATE = 0.05
MEMORY_BLENDING_ALPHA = 0.3
MEMORY_MIN_STRENGTH = 0.1
MEMORY_SHORT_TERM_BUFFER = 50
MEMORY_LONG_TERM_BUFFER = 50


# ===== B-Cell 파라미터 =====
INITIAL_PORTFOLIO_WEIGHTS = {
    "primary": 0.3,
    "secondary": 0.25,
}

CONFIDENCE_BOOST_FACTOR = 0.5
CRISIS_LEVEL_THRESHOLD = 0.5


# ===== 네트워크 학습 세부사항 =====
DEFAULT_DROPOUT_RATE = 0.1
SCHEDULER_FACTOR = 0.8
SCHEDULER_PATIENCE = 15
ENTROPY_BONUS = 0.01
SPECIALIST_WEIGHT = 3.0
EXPLORATION_STRENGTH_SPECIALTY = 0.05
EXPLORATION_STRENGTH_GENERAL = 0.1


# ===== 시스템 리소스 관리 =====
CPU_DEFAULT_MEMORY = 4000
GPU_MEMORY_THRESHOLD_1GB = 1000
GPU_MEMORY_THRESHOLD_2GB = 2000

DYNAMIC_BATCH_SIZES = {
    "low_memory": 16,
    "medium_memory": 32,
    "high_memory": 64,
    "full_memory": 256,
}

MEMORY_BATCH_DIVISOR = 100
BUFFER_TRIM_SIZE = 1000
REWARD_HISTORY_TRIM_SIZE = 100


# ===== 체크포인트 관리 =====
CHECKPOINT_SAVE_INTERVAL = 100
MAX_CHECKPOINTS = 5
CHECKPOINT_HISTORY_SIZE = 100


# ===== 테스트 및 검증 (현실화) =====
DEFAULT_N_RUNS = 50
DEFAULT_BASE_SEED = 42
TEST_EPISODES_PER_RUN = 1000

PERFORMANCE_THRESHOLDS = {
    "buy_hold_improvement": 0.15,
    "sharpe_improvement": 0.1,
    "convergence_episodes": 20000,
    "stability_cv": 0.3,
}


# ===== 실험 및 재현성 =====
USE_FIXED_SEED = False
GLOBAL_SEED = 42
RANDOM_SEED_MODULO = 10000
RUN_SEED_MULTIPLIER = 1000


# ===== 백테스터 및 시뮬레이션 =====
EXTREME_RETURN_THRESHOLD = 3.0
MAX_SIMULATION_ITERATIONS = 200

SIMULATION_PARAMS = {
    "base_drift": 0.2,
    "base_volatility": 0.1,
    "noise_scale": 0.3,
    "trend_strength": 0.5,
}


# ===== 기술적 분석 =====
RSI_SCALE = 100
MOVING_AVERAGE_WINDOWS = {
    "short": 20,
    "long": 100,
}

MOVING_AVERAGE_WINDOW = 50
MIN_EPISODES_FOR_STATS = 10


# ===== 시각화 =====
PLOT_STYLE = "default"
FIGURE_SIZE = (16, 12)
DPI = 100
HIGH_DPI = 300
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 12

GRID_SPACING = {
    "hspace": 0.3,
    "wspace": 0.3,
}

ALPHA_VALUES = {
    "light": 0.3,
    "medium": 0.5,
    "dark": 0.7,
    "opaque": 1.0,
}

TRACKER_GRID_ROWS = 4
TRACKER_GRID_COLS = 3


# ===== 로깅 =====
LOGGING_LEVELS = {
    "full": "전체 로깅",
    "sample": "샘플 로깅",
    "minimal": "최소 로깅",
}

LOGGING_LEVELS_ENUM = {
    "DEBUG": "디버깅 정보",
    "INFO": "일반 정보",
    "WARNING": "경고",
    "ERROR": "오류",
}


# ===== XAI 분석 =====
REACTION_TIME_THRESHOLD = 6
JSON_INDENT = 2


# ===== 출력 포맷 =====
FORMAT_PRECISION = {
    "percentage": ".2%",
    "float_2": ".2f",
    "float_3": ".3f",
    "float_4": ".4f",
}

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"


# ===== 초기화 값 =====
WEIGHT_INIT_RANGE = (-0.1, 0.1)
BIAS_INIT_VALUE = 0.0


# ===== 기능 플래그 =====
ENABLE_ALL_FEATURES = True
