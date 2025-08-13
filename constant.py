# constant.py
"""
상수 파일 - 각 섹션은 관련된 기능별로 그룹화
"""

import os
from datetime import datetime

# ===== 1. 파일 시스템 및 디렉토리 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def create_directories():
    """필요한 디렉토리들 생성"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def create_timestamped_directory(base_dir, prefix="run"):
    """타임스탬프 기반 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir


# ===== 2. 데이터 설정 =====
# 분석할 주식 종목 리스트
STOCK_SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "GOOGL",  # Google
    "AMD",  # AMD
    "TSLA",  # Tesla
    "JPM",  # JP Morgan
    "JNJ",  # Johnson & Johnson
    "PG",  # Procter & Gamble
    "V",  # Visa
]

# 학습 및 테스트 기간
TRAIN_START_DATE = "2008-01-02"  # 학습 시작일
TRAIN_END_DATE = "2020-12-31"  # 학습 종료일
TEST_START_DATE = "2021-01-01"  # 테스트 시작일
TEST_END_DATE = "2024-12-31"  # 테스트 종료일

# 분석 기간 (백테스터 리포트용)
ANALYSIS_START_DATE = "2021-01-01"  # 상세 분석 시작일
ANALYSIS_END_DATE = "2021-06-30"  # 상세 분석 종료일


# ===== 3. 에이전트 시스템 설정 =====
# 면역 시스템 에이전트 수
DEFAULT_N_TCELLS = 3  # T-Cell 수 (위험 감지)
DEFAULT_N_BCELLS = 5  # B-Cell 수 (전략 실행)
DEFAULT_MEMORY_SIZE = 20  # 메모리 셀 크기

# 데이터 처리
DEFAULT_LOOKBACK = 20  # 과거 데이터 참조 기간
FEATURE_SIZE = 12  # 입력 특성 차원
EXPECTED_FEATURES = 12  # 예상 특성 수


# ===== 4. 강화학습 하이퍼파라미터 =====
# 학습률 설정
DEFAULT_LEARNING_RATE = 0.0003  # 기본 학습률
DEFAULT_ACTOR_LR = 0.0003  # Actor 네트워크 학습률
DEFAULT_CRITIC_LR = 0.001  # Critic 네트워크 학습률
DEFAULT_ATTENTION_LR = 0.0002  # 어텐션 메커니즘 학습률
DEFAULT_META_LR = 0.0005  # 메타 컨트롤러 학습률
DEFAULT_MEMORY_LR = 0.0005  # 메모리 네트워크 학습률

# 강화학습 파라미터
DEFAULT_GAMMA = 0.99  # 할인 인수
DEFAULT_TAU = 0.005  # 소프트 업데이트 파라미터
DEFAULT_BATCH_SIZE = 64  # 미니배치 크기
DEFAULT_UPDATE_FREQUENCY = 100  # 네트워크 업데이트 주기

# 탐험-활용 파라미터
DEFAULT_EPSILON = 0.2  # 초기 탐험 확률
DEFAULT_EPSILON_DECAY = 0.999  # 에피소드마다 감소율
DEFAULT_MIN_EPSILON = 0.01  # 최소 탐험 확률


# ===== 5. 학습 프로세스 설정 =====
# 에피소드 설정
TOTAL_EPISODES = 2000  # 전체 학습 에피소드
PRETRAIN_EPISODES = 500  # 사전 학습 에피소드
EPISODE_LENGTH = 252  # 에피소드당 스텝 수 (1년 거래일)

# 커리큘럼 학습 설정
CURRICULUM_MIN_EPISODES = {
    0: 400,  # Level 0: 쉬운 시장
    1: 600,  # Level 1: 중간 난이도
    2: 1000,  # Level 2: 어려운 시장
}


# ===== 6. 네트워크 아키텍처 =====
# 메타 컨트롤러 상태 차원
META_STATE_DIM = 29  # base(12) + crisis(4) + expert_perf(5) + selection(5) + temporal(3)
# Actor-Critic 네트워크
ACTOR_HIDDEN_SIZE = 128  # Actor 기본 은닉층
CRITIC_HIDDEN_SIZE = 256  # Critic 기본 은닉층
BCELL_ACTOR_HIDDEN_SIZE = 64  # B-Cell Actor 은닉층
BCELL_CRITIC_HIDDEN_SIZE = 64  # B-Cell Critic 은닉층

# 어텐션 및 메타 학습
ATTENTION_HIDDEN_SIZE = 64  # 어텐션 메커니즘 은닉층
ATTENTION_HIDDEN_DIM = 32  # 어텐션 차원
META_CONTROLLER_HIDDEN_SIZE = 512  # 메타 컨트롤러 은닉층
META_CONTROLLER_DEFAULT_HIDDEN = 128  # 메타 컨트롤러 기본 크기

# 메모리 네트워크
MEMORY_EMBEDDING_DIM = 64  # 메모리 임베딩 차원
MEMORY_HIDDEN_DIM = 64  # 메모리 은닉층 크기


# ===== 7. 메모리 버퍼 설정 =====
"""
deque의 maxlen 파라미터: 버퍼가 가득 차면 오래된 데이터부터 자동 삭제
각 버퍼는 특정 유형의 경험이나 데이터를 저장
"""

# B-Cell 메모리 버퍼
BCELL_SPECIALIZATION_BUFFER_SIZE = 1000  # 전문화된 시장 상황별 경험
BCELL_GENERAL_BUFFER_SIZE = 500  # 일반적인 거래 경험
BCELL_PERFORMANCE_BUFFER_SIZE = 50  # 최근 성과 추적 (단기 성과)
BCELL_DECISION_BUFFER_SIZE = 100  # 의사결정 이력

# T-Cell 메모리 버퍼
TCELL_ACTIVATION_BUFFER_SIZE = 200  # 위험 감지 활성화 패턴
TCELL_PATTERN_BUFFER_SIZE = 100  # 인식된 위험 패턴
TCELL_REWARD_BUFFER_SIZE = 50  # 보상 피드백 추적

# 시스템 레벨 버퍼
MEMORY_PERFORMANCE_BUFFER_SIZE = 50  # 메모리 시스템 성능
REWARD_HISTORY_BUFFER_SIZE = 252  # 1년치 거래일 보상 (252 거래일)
REWARD_REGIME_BUFFER_SIZE = 5  # 시장 체제 변화 감지
REWARD_MEMORY_BUFFER_SIZE = 1000  # 전체 보상 메모리
HIERARCHICAL_EXPERIENCE_BUFFER_SIZE = 1000  # 계층적 학습 경험
HIERARCHICAL_EXPERT_PERFORMANCE_SIZE = 100  # 각 전문가 성과
HIERARCHICAL_SELECTION_HISTORY_SIZE = 200  # 전문가 선택 패턴
CURRICULUM_SITUATION_BUFFER_SIZE = 100  # 학습 난이도별 상황


# ===== 8. 임계값 및 한계값 =====
# 활성화 및 의사결정 임계값
DEFAULT_ACTIVATION_THRESHOLD = 0.5  # 에이전트 활성화 최소값
MEMORY_SIMILARITY_THRESHOLD = 0.8  # 유사 경험 판단 기준
BASE_RISK_THRESHOLD = 0.5  # 기본 위험 수준

# 시장 상황 판단
EXTREME_MARKET_THRESHOLD = 3.0  # 극단적 시장 (3 표준편차)
CURRICULUM_RETURN_THRESHOLDS = {  # 수익률 기준 시장 분류
    "high_positive": 0.001,  # +0.1% 이상: 양호
    "low_negative": -0.001,  # -0.1% 이하: 경미한 손실
    "very_high_positive": 0.002,  # +0.2% 이상: 매우 양호
    "very_low_negative": -0.002,  # -0.2% 이하: 심각한 손실
}

# 시스템 검증
FEATURE_VALIDATION_THRESHOLD = 100  # 특성값 이상치 감지
SPECIALIZATION_STRENGTH_INCREMENT = 0.005  # B-Cell 전문화 증가 단위

# 위험 수준 분류
RISK_THRESHOLDS = {
    "low": 0.3,  # 저위험
    "medium": 0.5,  # 중위험
    "high": 0.7,  # 고위험
    "critical": 0.9,  # 위기
}


# ===== 9. 거래 및 비용 설정 =====
# 보상 함수 가중치
REWARD_RETURN_WEIGHT = 0.3      # 수익률 비중
REWARD_RISK_WEIGHT = 0.3        # 위험 조정 성과 비중  
REWARD_TARGET_WEIGHT = 0.2      # 목표 달성 비중
REWARD_ADAPTATION_WEIGHT = 0.1  # 시장 적응성 비중
REWARD_TRANSACTION_WEIGHT = 0.05 # 거래 비용 비중
REWARD_CONCENTRATION_WEIGHT = 0.05 # 집중도 패널티 비중

# 보상 스케일링 팩터
REWARD_RETURN_SCALE = 10        # 수익률 스케일링
REWARD_RISK_SCALE = 5           # 위험 조정 보상 스케일링
REWARD_TRANSACTION_SCALE = 1000 # 거래 비용 스케일링
REWARD_VOLATILITY_SCALE = 50    # 변동성 스케일링
REWARD_DRAWDOWN_SCALE = 100     # 낙폭 스케일링
REWARD_CRISIS_SCALE = 15        # 위기 적응 스케일링
REWARD_CONCENTRATION_SCALE = 20 # 집중도 스케일링

# B-Cell 전문화 보상 스케일링
SPECIALIST_REWARD_BOOST = 1.5   # 전문 상황 보상 증폭
SPECIALIST_PENALTY_FACTOR = 0.8 # 비전문 상황 페널티
GENERAL_REWARD_BOOST = 2.0      # 일반 상황 보상 증폭

# B-Cell 전문화 세부 팩터
SPECIALIZATION_CONFIDENCE_FACTOR = 0.5  # 전문성 신뢰도 팩터
SPECIALIZATION_LIQUIDITY_FACTOR = 0.6   # 유동성 전문화 팩터  
SPECIALIZATION_MACRO_FACTOR = 0.7       # 거시경제 전문화 팩터

# 면역 시스템 전략 조정 팩터
RISK_REDUCTION_FACTOR = 0.3         # 위험 감소 팩터
DIVERSIFICATION_FACTOR = 0.2        # 분산투자 팩터
NEUTRAL_ADJUSTMENT_FACTOR = 0.25    # 중성 조정 팩터
LARGE_CAP_FACTOR = 0.2              # 대형주 선호 팩터
DEFENSIVE_FACTOR = 0.3              # 방어적 전략 팩터

# 메모리 시스템 팩터
MEMORY_STRENGTH_FACTOR = 0.1        # 메모리 강도 팩터

# 수치 안정성 상수
EPSILON_SMALL = 1e-8               # Division by zero 방지용
WEIGHT_NORMALIZATION_MIN = 1e-8    # 가중치 정규화 최소값

# 특성 스케일링 팩터
MARKET_VOLATILITY_SCALE = 5     # 시장 변동성 스케일링
MARKET_RETURN_SCALE = 10        # 시장 수익률 스케일링
VIX_PROXY_SCALE = 3             # VIX 프록시 스케일링
BOLLINGER_SCALE = 2             # 볼린저 밴드 스케일링
PRICE_RANGE_SCALE = 2           # 가격 범위 스케일링
DEFAULT_TRANSACTION_COST_RATE = 0.001  # 거래 수수료 (0.1%)

# 보상 계산 파라미터
COST_PENALTY_MULTIPLIER = 1000  # 거래비용을 보상에 반영할 때 사용
DRAWDOWN_PENALTY_BASE = -10  # 낙폭 기본 페널티
DRAWDOWN_PENALTY_MULTIPLIER = 100  # 낙폭 페널티 증폭


# ===== 10. 기술적 분석 설정 =====
RSI_SCALE = 100  # RSI 지표 범위 (0-100)
MOVING_AVERAGE_WINDOWS = {
    "short": 20,  # 단기 이동평균 (1개월)
    "long": 100,  # 장기 이동평균 (5개월)
}


# ===== 11. 시스템 리소스 관리 =====
# 메모리 관리
CPU_DEFAULT_MEMORY = 1000  # CPU 모드 기본 메모리 (MB)
MEMORY_BATCH_DIVISOR = 100  # 동적 배치 크기 계산 (메모리/100)
BUFFER_TRIM_SIZE = 1000  # 버퍼 정리 시 남길 크기
REWARD_HISTORY_TRIM_SIZE = 100  # 보상 이력 정리 크기

# 체크포인트
CHECKPOINT_SAVE_INTERVAL = 100  # 저장 주기 (에피소드)
MAX_CHECKPOINTS = 5  # 최대 보관 개수
CHECKPOINT_HISTORY_SIZE = 100  # 체크포인트 성과 이력


# ===== 12. 실험 및 재현성 =====
# 랜덤 시드
USE_FIXED_SEED = False  # True: 재현 가능, False: 랜덤
GLOBAL_SEED = 42  # 고정 시드값
RANDOM_SEED_MODULO = 10000  # 시드 생성 범위 (0-9999)
RUN_SEED_MULTIPLIER = 1000  # 다중 실행 시드 간격

# 백테스트
DEFAULT_N_RUNS = 3  # 기본 반복 실행 횟수
DEFAULT_BASE_SEED = 42  # 다중 실행 기본 시드


# ===== 13. 시각화 및 로깅 =====
# 플롯 설정
PLOT_STYLE = "seaborn-v0_8-darkgrid"  # matplotlib 스타일
FIGURE_SIZE = (16, 12)  # 기본 그림 크기
DPI = 100  # 해상도

# 로깅 간격 설정
MEMORY_CLEANUP_INTERVAL = 200   # 메모리 정리 간격
LOG_INTERVAL_DETAILED = 10      # 상세 로깅 간격
LOG_INTERVAL_NORMAL = 20        # 일반 로깅 간격
LOG_INTERVAL_SPARSE = 50        # 희소 로깅 간격

# 로깅 레벨
LOGGING_LEVELS = {
    "full": "전체 로깅",  # 모든 정보
    "sample": "샘플 로깅",  # 주요 정보만
    "minimal": "최소 로깅",  # 최소한의 정보
}


# ===== 14. 기능 플래그 =====
ENABLE_ALL_FEATURES = True  # 모든 고급 기능 활성화
