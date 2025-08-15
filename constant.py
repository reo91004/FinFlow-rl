# constant.py
"""
BIPD 시스템 통합 상수 정의
강화학습 시스템 파라미터 설정
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
ANALYSIS_END_DATE = "2024-12-31"


# ===== 에이전트 시스템 =====
# core/system.py에서 면역 시스템 초기화 시 사용
DEFAULT_N_TCELLS = 3  # T-Cell 에이전트 개수 (위험 감지용)
DEFAULT_N_BCELLS = 5  # B-Cell 에이전트 개수 (포트폴리오 결정용)
DEFAULT_MEMORY_SIZE = 20  # agents/memory_cell.py에서 메모리 버퍼 크기
DEFAULT_LOOKBACK = 20  # 과거 데이터 참조 윈도우 크기
FEATURE_SIZE = 12  # 시장 특성 벡터 차원 (core/backtester.py에서 사용)
EXPECTED_FEATURES = 12  # 예상 특성 개수 (검증용)


# ===== 강화학습 파라미터 =====
# 학습률 대폭 상향 (빠른 학습을 위해 30배 증가)
DEFAULT_LEARNING_RATE = 0.001  # 일반적 학습률 (agents/tcell.py 등에서 사용)
DEFAULT_ACTOR_LR = 0.003  # B-Cell Actor 네트워크 학습률 (agents/bcell.py)
DEFAULT_CRITIC_LR = 0.003  # B-Cell Critic 네트워크 학습률 (agents/bcell.py)
DEFAULT_ATTENTION_LR = 0.001  # Attention 메커니즘 학습률 (agents/attention.py)
DEFAULT_META_LR = 0.001  # 메타 컨트롤러 학습률 (core/meta_controller.py)
DEFAULT_MEMORY_LR = 0.001  # 메모리 셀 학습률 (agents/memory_cell.py)

# RL 파라미터 - 빠른 학습과 안정성 균형
DEFAULT_GAMMA = 0.95  # 할인 팩터 (agents/bcell.py에서 미래 보상 계산용)
DEFAULT_TAU = 0.005  # 타겟 네트워크 소프트 업데이트 비율 (agents/bcell.py)
DEFAULT_BATCH_SIZE = 8  # 학습 배치 크기 (agents/bcell.py) - 빠른 학습 시작
DEFAULT_UPDATE_FREQUENCY = 5  # 기본 학습 주기 (agents/bcell.py에서 N일마다 학습)
INTERMEDIATE_LEARNING_FREQUENCY = (
    3  # 중간 학습 주기 (core/backtester.py에서 에피소드 중 학습)
)

# 탐험-활용 균형 조정 (agents/bcell.py에서 사용)
DEFAULT_EPSILON = 0.2  # 초기 탐험 확률 (랜덤 행동 선택 비율)
DEFAULT_EPSILON_DECAY = 0.995  # 에피소드마다 epsilon 감소 비율
DEFAULT_MIN_EPSILON = 0.05  # 최소 탐험 확률 (완전히 0으로 떨어지지 않도록)


# ===== 학습 프로세스 (core/backtester.py, core/curriculum.py에서 사용) =====
EPISODE_LENGTH = 252  # 1 에피소드 = 252일 (1년 거래일)
PRETRAIN_EPISODES = 500  # 사전 훈련 에피소드 수 (빠른 시작을 위해 감축)
TOTAL_EPISODES = 5  # 기본 훈련 총 에피소드 수
CURRICULUM_TOTAL_EPISODES = 10  # 커리큘럼 학습 총 에피소드 수

# 커리큘럼 학습 레벨별 최소 에피소드 수 (core/curriculum.py)
CURRICULUM_MIN_EPISODES = {
    0: 2,  # 레벨 0 (안정적 시장): 최소 2000 에피소드
    1: 3,  # 레벨 1 (중간 변동성): 최소 3000 에피소드
    2: 5,  # 레벨 2 (위기 상황): 최소 5000 에피소드
}


# ===== Experience Replay 및 Target Network (agents/bcell.py에서 사용) =====
EXPERIENCE_BUFFER_SIZE = 50000  # 경험 재생 버퍼 크기 (메모리 효율화)
TARGET_UPDATE_FREQUENCY = 20  # 타겟 네트워크 업데이트 주기 (안정성 향상)


# ===== 네트워크 아키텍처 (각 신경망 모델의 히든 레이어 크기) =====
ACTOR_HIDDEN_SIZE = 128  # 일반 Actor 네트워크 히든 크기
CRITIC_HIDDEN_SIZE = 256  # 일반 Critic 네트워크 히든 크기
BCELL_ACTOR_HIDDEN_SIZE = 256  # B-Cell Actor 네트워크 히든 크기 (agents/bcell.py)
BCELL_CRITIC_HIDDEN_SIZE = 256  # B-Cell Critic 네트워크 히든 크기 (agents/bcell.py)
ATTENTION_HIDDEN_SIZE = 64  # Attention 메커니즘 히든 크기 (agents/attention.py)
ATTENTION_HIDDEN_DIM = 32  # Attention 차원 크기
META_CONTROLLER_HIDDEN_SIZE = 512  # 메타 컨트롤러 히든 크기 (core/meta_controller.py)
META_CONTROLLER_DEFAULT_HIDDEN = 128  # 메타 컨트롤러 기본 히든 크기
MEMORY_EMBEDDING_DIM = 64  # 메모리 임베딩 차원 (agents/memory_cell.py)
MEMORY_HIDDEN_DIM = 64  # 메모리 히든 차원
HIDDEN_LAYER_DIVISOR = 2  # 히든 레이어 크기 분할자 (네트워크 설계용)
ATTENTION_HIDDEN_RATIO = 0.5  # Attention 히든 크기 비율


# ===== 메모리 버퍼 (각 에이전트의 경험 저장 버퍼 크기) =====
# B-Cell 전용 버퍼 (agents/bcell.py)
BCELL_SPECIALIZATION_BUFFER_SIZE = 1000  # 전문화 경험 버퍼 크기
BCELL_GENERAL_BUFFER_SIZE = 500  # 일반 경험 버퍼 크기
BCELL_PERFORMANCE_BUFFER_SIZE = 50  # 성과 추적 버퍼 크기
BCELL_DECISION_BUFFER_SIZE = 100  # 결정 이력 버퍼 크기

# T-Cell 전용 버퍼 (agents/tcell.py)
TCELL_TRAINING_BUFFER_SIZE = 200  # T-Cell 훈련 데이터 버퍼
TCELL_ACTIVATION_BUFFER_SIZE = 200  # T-Cell 활성화 이력 버퍼
TCELL_PATTERN_BUFFER_SIZE = 100  # 위험 패턴 인식 버퍼
TCELL_REWARD_BUFFER_SIZE = 50  # T-Cell 보상 추적 버퍼

# 시스템 레벨 버퍼 (core/system.py, agents/memory_cell.py)
MEMORY_PERFORMANCE_BUFFER_SIZE = 50  # 메모리 성과 추적 버퍼
REWARD_HISTORY_BUFFER_SIZE = 252
REWARD_REGIME_BUFFER_SIZE = 5
REWARD_MEMORY_BUFFER_SIZE = 1000
HIERARCHICAL_EXPERIENCE_BUFFER_SIZE = 1000
HIERARCHICAL_EXPERT_PERFORMANCE_SIZE = 100
HIERARCHICAL_SELECTION_HISTORY_SIZE = 200
CURRICULUM_SITUATION_BUFFER_SIZE = 100


# ===== 임계값 (각종 판단 기준값들) =====
DEFAULT_ACTIVATION_THRESHOLD = 0.5  # 기본 활성화 임계값 (agents/bcell.py)
MEMORY_SIMILARITY_THRESHOLD = 0.8  # 메모리 유사도 임계값 (agents/memory_cell.py)
BASE_RISK_THRESHOLD = 0.5  # 기본 위험도 임계값 (core/system.py)
EXTREME_MARKET_THRESHOLD = 3.0  # 극단적 시장 상황 임계값 (core/backtester.py)
FEATURE_VALIDATION_THRESHOLD = 100  # 특성 검증 임계값
SPECIALIZATION_STRENGTH_INCREMENT = 0.005  # 전문화 강도 증분 (agents/bcell.py)

# 커리큘럼 학습 수익률 임계값 (core/curriculum.py)
CURRICULUM_RETURN_THRESHOLDS = {
    "high_positive": 0.001,  # 높은 양의 수익률
    "low_negative": -0.001,  # 낮은 음의 수익률
    "very_high_positive": 0.002,  # 매우 높은 양의 수익률
    "very_low_negative": -0.002,  # 매우 낮은 음의 수익률
}

# 위험도 수준별 임계값 (agents/tcell.py, core/system.py)
RISK_THRESHOLDS = {
    "low": 0.3,  # 낮은 위험
    "medium": 0.5,  # 중간 위험
    "high": 0.7,  # 높은 위험
    "critical": 0.9,  # 임계적 위험
}

# 에이전트별 활성화 임계값 (각 에이전트 파일에서 사용)
ACTIVATION_THRESHOLDS = {
    "tcell": [0.8, 0.6, 0.4, 0.2],  # T-Cell 다단계 임계값
    "bcell": 0.1,  # B-Cell 임계값 (시장 특성 평균 0.2-0.3에 맞춰 낮춤)
    "memory": 0.3,  # Memory Cell 임계값
}


# ===== 거래 및 비용 =====
DEFAULT_TRANSACTION_COST_RATE = 0.001
COST_PENALTY_MULTIPLIER = 1000
DRAWDOWN_PENALTY_BASE = -10
DRAWDOWN_PENALTY_MULTIPLIER = 100


# ===== 보상 시스템 (core/reward_calculator.py에서 사용) =====
REWARD_CLIPPING_RANGE = (-10.0, 10.0)  # 보상 클리핑 범위 (학습 안정성 위해)
RETURN_CLIPPING_RANGE = (-0.2, 0.2)  # 수익률 클리핑 범위 (이상치 제거)
RETURN_SCALING_FACTOR = 20  # 수익률 스케일링 팩터
TARGET_VOLATILITY = 0.15  # 목표 변동성 (위험 조정 보상 계산용)

# 보상 구성 요소별 가중치 (core/reward_calculator.py)
REWARD_COMPONENT_WEIGHTS = {
    "return": 0.4,  # 수익률 가중치
    "risk_adjusted": 0.3,  # 위험 조정 가중치
    "target": 0.2,  # 목표 달성 가중치
    "adaptation": 0.1,  # 적응성 가중치
}

# 기존 리스트 형태도 호환성 유지
REWARD_COMPONENT_WEIGHTS_LIST = [0.3, 0.3, 0.2, 0.1, 0.05, 0.05]
REWARD_NORMALIZATION_THRESHOLD = 100  # 보상 정규화 임계값
CONCENTRATION_PENALTY_THRESHOLD = 0.4  # 집중도 페널티 임계값 (분산투자 유도)


# ===== T-Cell 파라미터 (agents/tcell.py에서 사용) =====
DEFAULT_SENSITIVITY = 0.1  # T-Cell 기본 민감도 (위험 감지 민감도)
TCELL_ACTIVATION_THRESHOLDS = [0.8, 0.6, 0.4, 0.2]  # T-Cell 활성화 임계값 리스트

# 위험 유형별 부스트 팩터 (agents/tcell.py)
RISK_BOOST_FACTORS = {
    "volatility": 0.2,  # 변동성 위험 부스트
    "stress": 0.15,  # 스트레스 상황 부스트
    "correlation": 0.1,  # 상관관계 위험 부스트
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
DEFAULT_N_RUNS = 5
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
