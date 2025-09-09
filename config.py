# bipd/config.py

import numpy as np
import torch
import os

# =============================================================================
# 1. 시스템 기본 설정 (System Basic Settings)
# =============================================================================

# 시드 설정 - 재현 가능한 실험을 위한 전역 시드
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


# =============================================================================
# 2. 디렉토리 및 파일 경로 설정 (Directory & File Path Settings)
# =============================================================================

# 기본 디렉토리 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")         # 시장 데이터 저장 디렉토리
MODELS_DIR = os.path.join(BASE_DIR, "models")     # 학습된 모델 저장 디렉토리  
LOGS_DIR = os.path.join(BASE_DIR, "logs")         # 실행 로그 저장 디렉토리

# =============================================================================
# 3. 시장 데이터 설정 (Market Data Settings)
# =============================================================================

# 거래 대상 주식 심볼 (다우존스 30 종목)
SYMBOLS = [
    "MMM",   # 3M Company
    "AXP",   # American Express
    "AMGN",  # Amgen
    "AMZN",  # Amazon
    "AAPL",  # Apple
    "BA",    # Boeing
    "CAT",   # Caterpillar
    "CVX",   # Chevron
    "CSCO",  # Cisco Systems
    "KO",    # Coca-Cola
    "DIS",   # Walt Disney
    "GS",    # Goldman Sachs
    "HD",    # Home Depot
    "HON",   # Honeywell
    "IBM",   # IBM
    "JNJ",   # Johnson & Johnson
    "JPM",   # JPMorgan Chase
    "MCD",   # McDonald's
    "MRK",   # Merck
    "MSFT",  # Microsoft
    "NKE",   # Nike
    "NVDA",  # NVIDIA
    "PG",    # Procter & Gamble
    "CRM",   # Salesforce
    "SHW",   # Sherwin-Williams
    "TRV",   # Travelers
    "UNH",   # UnitedHealth Group
    "VZ",    # Verizon
    "V",     # Visa
    "WMT",   # Walmart
]

# 데이터 기간 설정 (Yahoo Finance API 호환 형식)
TRAIN_START = "2008-01-01"  # 학습 시작일 (금융위기 포함)
TRAIN_END = "2020-12-31"    # 학습 종료일
TEST_START = "2021-01-01"   # 테스트 시작일
TEST_END = "2024-12-31"     # 테스트 종료일

# =============================================================================
# 4. 특성 추출 설정 (Feature Extraction Settings)
# =============================================================================

# 시장 특성 차원 설정 (data/features.py에서 사용)
FEATURE_DIM = 12         # 추출되는 시장 특성의 차원 수
LOOKBACK_WINDOW = 20     # 특성 계산을 위한 과거 데이터 윈도우 크기 (거래일 기준)

# =============================================================================
# 5. T-Cell (위기 감지) 설정 (T-Cell Crisis Detection Settings)
# =============================================================================

# Isolation Forest 기본 파라미터 (agents/tcell.py에서 사용)
TCELL_CONTAMINATION = 0.1    # Isolation Forest 이상치 비율 (전체 데이터 중 10%)
TCELL_SENSITIVITY = 1.0      # 위기 감지 민감도 (1.0 = 기본값)

# 적응적 임계값 관리 설정 (agents/tcell.py에서 사용) - 안정화 최적화
THRESHOLD_WINDOW_SIZE = int(1000)           # 임계값 적응을 위한 윈도우 크기 (안정성 향상)
VOLATILITY_CRISIS_QUANTILE = float(0.85)    # 변동성 위기 감지 분위수 (민감도 완화)
CORRELATION_CRISIS_QUANTILE = float(0.87)   # 상관성 위기 감지 분위수 (민감도 완화)
VOLUME_CRISIS_QUANTILE = float(0.86)        # 거래량 위기 감지 분위수 (민감도 완화)
OVERALL_CRISIS_QUANTILE = float(0.88)       # 전체 위기 감지 분위수 (민감도 완화)

# 목표 위기 감지율 (과도한 위기 감지 방지) - 균형 조정
VOLATILITY_CRISIS_RATE = float(0.18)    # 변동성 위기 목표 감지율 (18%)
CORRELATION_CRISIS_RATE = float(0.15)   # 상관성 위기 목표 감지율 (15%)
VOLUME_CRISIS_RATE = float(0.17)        # 거래량 위기 목표 감지율 (17%)
OVERALL_CRISIS_RATE = float(0.20)       # 전체 위기 목표 감지율 (20%)

# 위기 레벨 분류 임계값 (core/environment.py, core/system.py에서 사용)
CRISIS_HIGH = float(0.7)      # 고위기 임계값 (0.7 이상)
CRISIS_MEDIUM = float(0.4)    # 중위기 임계값 (0.4-0.7)

# =============================================================================
# 6. B-Cell (강화학습 에이전트) 설정 (B-Cell Reinforcement Learning Settings)
# =============================================================================

# 상태-행동 공간 설정
STATE_DIM = FEATURE_DIM + 1 + len(SYMBOLS)  # 상태 차원: 시장특성(12) + 위기레벨(1) + 이전가중치(30) = 43
ACTION_DIM = len(SYMBOLS)                   # 행동 차원: 포트폴리오 가중치(30)
HIDDEN_DIM = 128                           # 신경망 은닉층 차원

# SAC 하이퍼파라미터 (agents/bcell.py에서 사용) - 안정성 우선 v5.2
LR_ACTOR = float(3e-5)      # Actor 네트워크 학습률 (안정성 우선, 보수적 설정)
LR_CRITIC = float(3e-5)     # Critic 네트워크 학습률 (Actor와 동일하게 설정)
WEIGHT_DECAY = float(1e-6)  # 가중치 감쇠 (L2 정규화, 약한 정규화 추가)
GAMMA = float(0.99)         # 할인율 (표준값으로 안정화)
TAU = float(0.005)          # 타겟 네트워크 EMA 업데이트 비율 (보수적 설정)
BATCH_SIZE = int(256)       # 미니배치 크기 (효율성과 안정성 균형)
BUFFER_SIZE = int(20000)    # 경험 재생 버퍼 크기 (다양성 증대)
REWARD_SCALE = float(0.5)   # Q 폭주 방지용 보상 스케일링 (강화)
UPDATE_FREQUENCY = int(4)   # 네트워크 업데이트 주기 (안정성 우선)

# SAC 온도 파라미터 (temperature α - 스칼라) - 안정성 우선
INIT_ALPHA_TEMP = float(0.05)               # SAC temperature 초기값 (보수적 탐험)
TARGET_ENTROPY_PER_DIM = float(-0.8)        # SAC 목표 엔트로피 (안정적 범위)
ALPHA_TEMP_LEARN = True                     # temperature 자동 학습 활성화
ALPHA_LR = float(3e-4)                      # temperature 학습률 (안정적 적응)

# Dirichlet 정책 분포 파라미터 (심플렉스 가중치용 - concentration α 벡터)
DIRICHLET_MIN_CONC = float(0.02)            # concentration 하한 (one-hot 붕괴 방지)
DIRICHLET_MAX_CONC = float(5.0)             # concentration 상한 (과도 집중 방지)
ENTROPY_REG_COEF = float(0.0)               # 별도 엔트로피 보너스 (SAC temperature가 있으므로 0)
PORTFOLIO_WEIGHT_MIN = float(1e-4)          # 포트폴리오 최소 가중치

# 그래디언트 안정화 설정
CRITIC_GRAD_NORM = float(1.0)          # Critic 그래디언트 클리핑 (완화)
ACTOR_GRAD_NORM = float(1.0)           # Actor 그래디언트 클리핑 (완화)  
ALPHA_GRAD_NORM = float(10.0)          # Alpha 그래디언트 클리핑 (관대)
ENHANCED_HUBER_DELTA = float(2.0)      # 강화된 Huber loss delta

# CQL (Conservative Q-Learning) 정규화 설정 - 성능 최적화
ENABLE_CQL = True                       # CQL 정규화 활성화 플래그
CQL_ALPHA_START = float(0.01)           # CQL 정규화 시작 강도 (부드러운 시작)
CQL_ALPHA_END = float(0.05)             # CQL 정규화 최종 강도 (과추정 방지, 완화)
CQL_NUM_SAMPLES = int(8)                # CQL LogSumExp 샘플 수 (효율성)

# KL 정규화 설정 (Q-value 발산 방지) - 성능 최적화
ENABLE_KL_REGULARIZATION = True         # KL 정규화 활성화 플래그
KL_PENALTY_WEIGHT = float(0.005)        # KL 페널티 가중치 (완화)

# 통계 추적 설정 (agents/bcell.py에서 사용)
ROLLING_STATS_WINDOW = int(100)     # 성능 통계 슬라이딩 윈도우 크기

# =============================================================================
# 7. Memory Cell 설정 (Memory Cell Settings)
# =============================================================================

# 경험 저장 및 검색 설정 (agents/memory.py에서 사용)
MEMORY_CAPACITY = int(500)      # 메모리 셀 저장 용량 (경험 개수)
EMBEDDING_DIM = int(32)         # 상태 임베딩 차원
MEMORY_K = int(5)               # 유사 경험 검색 개수 (K-nearest neighbors)

# =============================================================================
# 8. 거래 환경 설정 (Trading Environment Settings)
# =============================================================================

# 기본 거래 설정 (core/environment.py에서 사용)
INITIAL_CAPITAL = int(1000000)      # 초기 자본금 ($1M)
TRANSACTION_COST = float(0.001)     # 거래 수수료 비율 (0.1%)
MAX_STEPS = int(252)                # 최대 거래일 수 (1년)

# 포트폴리오 최적화 설정
REWARD_EMPIRICAL_MEAN = float(0.002)    # 일일 평균 수익률 추정 (0.2%)
REWARD_EMPIRICAL_STD = float(0.02)      # 일일 변동성 추정 (2%)
VOLATILITY_TARGET = float(0.15)         # 연간 목표 변동성 (15% - 완화)
VOLATILITY_WINDOW = int(20)             # 변동성 추정 윈도우 (20거래일)

# 레버리지 및 거래 제약 설정 (외생화)
MIN_LEVERAGE = float(0.75)              # 최소 레버리지 (75% 투자 - 완화)
MAX_LEVERAGE = float(1.5)               # 최대 레버리지 (150% 투자 - 완화)
NO_TRADE_BAND = float(0.02)             # 노-트레이드 밴드 (2% 이하 변화시 거래 안함)
MAX_TURNOVER = float(0.5)               # 최대 일일 턴오버 (50%)

# 리스크 제약 및 페널티 가중치 (CVaR 추가) - MDD ≤25% 목표
CVAR_ALPHA = float(0.05)                # CVaR 분위수 (5%)
LAMBDA_DD = float(0.5)                  # 최대낙폭 페널티 (MDD 제한 강화)
LAMBDA_VOL = float(0.2)                 # 다운사이드 변동성 페널티 (리스크 관리 강화)
LAMBDA_TURN = float(0.005)              # 턴오버 페널티 (과도한 거래 방지)
LAMBDA_HHI = float(0.01)                # HHI 집중도 페널티 (분산투자 유도)

# 환경 모니터링 설정
WEIGHT_VALIDATION_WINDOW = int(100)     # 가중치 검증 윈도우
CORRELATION_CHECK_INTERVAL = int(100)   # 상관성 체크 간격
DEBUG_LOG_INTERVAL = int(50)            # 디버그 로그 출력 간격

# =============================================================================
# 9. 학습 프로세스 설정 (Training Process Settings)
# =============================================================================

# 학습 에피소드 및 주기 설정 (core/trainer.py, main.py에서 사용)
N_EPISODES = int(500)                   # 총 학습 에피소드 수
LOG_INTERVAL = int(10)                  # 로그 출력 간격
SAVE_INTERVAL = int(100)                # 모델 저장 간격

# MoE 게이팅 시스템 설정
SWITCH_DWELL_STEPS = int(5)             # 최소 전문가 체류 시간

# 초기 탐험 설정
INITIAL_EXPLORATION_STEPS = int(1000)   # 초기 랜덤 탐험 스텝 수
MIN_BUFFER_SIZE = int(500)              # 학습 시작을 위한 최소 버퍼 크기

# =============================================================================
# 10. 보상 함수 설정 (Reward Function Settings)
# =============================================================================

# Sharpe 비율 계산 설정 (utils/metrics.py, core/environment.py에서 사용) - Sharpe >1.0 목표
SHARPE_WINDOW = int(20)                 # Sharpe 비율 계산 윈도우 (길게 조정)
SHARPE_SCALE = float(0.5)               # Sharpe 비율 보상 스케일링 (강화)

# 보상 정규화 및 이상치 처리
REWARD_TANH_BOUND = True                     # tanh 바운딩 활성화
REWARD_TANH_R_MAX = float(0.05)              # tanh 바운딩 최대값
REWARD_TANH_TAU = float(0.05)                # tanh 바운딩 스케일링 파라미터
REWARD_BUFFER_SIZE = int(1000)               # 보상 정규화를 위한 버퍼 크기
REWARD_OUTLIER_SIGMA = float(3.0)            # 이상치 감지 시그마 배수 (3-sigma rule)
REWARD_CLIP_MIN = float(-2.0)                # 보상 하한 클리핑
REWARD_CLIP_MAX = float(2.0)                 # 보상 상한 클리핑

# Q-value 안정화 설정 
Q_TARGET_HARD_CLIP_MIN = float(-50.0)       # Q 타깃 하드 클리핑 하한
Q_TARGET_HARD_CLIP_MAX = float(50.0)        # Q 타깃 하드 클리핑 상한
Q_VALUE_STABILITY_CHECK = float(100.0)      # Q-value 안정성 체크 임계값
Q_MONITOR_WINDOW_SIZE = int(1000)           # Q-value 모니터링 윈도우 크기
Q_EXTREME_THRESHOLD = float(0.3)            # 극단치 비율 경고 임계값 (30%)

# EMA/히스테리시스 설정 (T-Cell 안정화용)
CRISIS_EMA_BETA = float(0.95)               # 위기 점수 EMA 계수
CRISIS_ENTER_THRESHOLD = float(0.7)         # 위기 진입 임계값
CRISIS_EXIT_THRESHOLD = float(0.3)          # 위기 해제 임계값

# 메모리 다양성 설정
MEMORY_SIMILARITY_THRESHOLD = float(0.7)    # 유사도 임계값 (0.95에서 완화)
MEMORY_EXTREME_RATIO = float(0.3)           # 극단 상황 리플레이 비율