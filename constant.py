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
DEFAULT_LEARNING_RATE = 0.001  # 기본 학습률
DEFAULT_ACTOR_LR = 0.001  # Actor 네트워크 학습률
DEFAULT_CRITIC_LR = 0.002  # Critic 네트워크 학습률 (일반적으로 Actor의 2배)
DEFAULT_ATTENTION_LR = 0.0005  # 어텐션 메커니즘 학습률 (Actor의 0.5배)
DEFAULT_META_LR = 0.001  # 메타 컨트롤러 학습률
DEFAULT_MEMORY_LR = 0.001  # 메모리 네트워크 학습률

# 강화학습 파라미터
DEFAULT_GAMMA = 0.95  # 할인 인수 (미래 보상의 현재 가치)
DEFAULT_TAU = 0.001  # 소프트 업데이트 파라미터
DEFAULT_BATCH_SIZE = 32  # 미니배치 크기
DEFAULT_UPDATE_FREQUENCY = 10  # 네트워크 업데이트 주기

# 탐험-활용 파라미터
DEFAULT_EPSILON = 0.3  # 초기 탐험 확률
DEFAULT_EPSILON_DECAY = 0.995  # 에피소드마다 감소율
DEFAULT_MIN_EPSILON = 0.05  # 최소 탐험 확률


# ===== 5. 학습 프로세스 설정 =====
# 에피소드 설정
TOTAL_EPISODES = 1000  # 전체 학습 에피소드
PRETRAIN_EPISODES = 300  # 사전 학습 에피소드
EPISODE_LENGTH = 60  # 에피소드당 스텝 수

# 커리큘럼 학습 설정
CURRICULUM_MIN_EPISODES = {
    0: 200,  # Level 0: 쉬운 시장 (안정적)
    1: 300,  # Level 1: 중간 난이도 (변동성 있음)
    2: 500,  # Level 2: 어려운 시장 (위기 상황)
}


# ===== 6. 네트워크 아키텍처 =====
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
PLOT_STYLE = "default"  # matplotlib 스타일 (호환성을 위해 기본값 사용)
FIGURE_SIZE = (16, 12)  # 기본 그림 크기
DPI = 100  # 해상도

# 로깅 레벨
LOGGING_LEVELS = {
    "full": "전체 로깅",  # 모든 정보
    "sample": "샘플 로깅",  # 주요 정보만
    "minimal": "최소 로깅",  # 최소한의 정보
}


# ===== 14. 기능 플래그 =====
ENABLE_ALL_FEATURES = True  # 모든 고급 기능 활성화


# ===== 15. 네트워크 학습 세부 설정 =====
# 드롭아웃 및 정규화
DEFAULT_DROPOUT_RATE = 0.1  # 기본 드롭아웃 비율
HIDDEN_LAYER_DIVISOR = 2  # 은닉층 크기 분할 계수
ATTENTION_HIDDEN_RATIO = 0.5  # 어텐션 은닉층 비율

# 학습률 스케줄러
SCHEDULER_FACTOR = 0.8  # 학습률 감소 비율
SCHEDULER_PATIENCE = 15  # 성능 개선 없을 시 대기 에피소드

# 엔트로피 및 탐험
ENTROPY_BONUS = 0.01  # 엔트로피 정규화 계수
SPECIALIST_WEIGHT = 3.0  # 전문가 모델 가중치
EXPLORATION_STRENGTH_SPECIALTY = 0.05  # 전문 영역 탐험 강도
EXPLORATION_STRENGTH_GENERAL = 0.1  # 일반 영역 탐험 강도


# ===== 16. B-Cell 특화 파라미터 =====
# 포트폴리오 가중치 초기화
INITIAL_PORTFOLIO_WEIGHTS = {
    "primary": 0.3,  # 주요 자산 초기 가중치
    "secondary": 0.25,  # 보조 자산 초기 가중치
}

# 신뢰도 및 위기 관리
CONFIDENCE_BOOST_FACTOR = 0.5  # 신뢰도 부스팅 계수
CRISIS_LEVEL_THRESHOLD = 0.5  # 위기 판단 임계값


# ===== 17. T-Cell 위험 감지 파라미터 =====
# 기본 민감도
DEFAULT_SENSITIVITY = 0.1  # 기본 위험 감지 민감도
TCELL_TRAINING_BUFFER_SIZE = 200  # T-Cell 훈련 데이터 버퍼

# 활성화 임계값 (위험도 레벨별)
TCELL_ACTIVATION_THRESHOLDS = [
    0.8,  # Level 1: 매우 높은 위험
    0.6,  # Level 2: 높은 위험
    0.4,  # Level 3: 중간 위험
    0.2,  # Level 4: 낮은 위험
]

# 위험 요소별 부스트 계수
RISK_BOOST_FACTORS = {
    "volatility": 0.2,  # 변동성 부스트
    "stress": 0.15,  # 스트레스 부스트
    "correlation": 0.1,  # 상관관계 부스트
}


# ===== 18. Memory Cell 파라미터 =====
# 메모리 시스템 기본값
MEMORY_DEFAULT_EMBEDDING_DIM = 64  # 임베딩 차원
MEMORY_MAX_MEMORIES = 100  # 최대 메모리 저장 수
MEMORY_DECAY_RATE = 0.05  # 메모리 감쇠율
MEMORY_BLENDING_ALPHA = 0.3  # 메모리 블렌딩 계수
MEMORY_MIN_STRENGTH = 0.1  # 최소 메모리 강도

# 메모리 버퍼 (Memory Cell 전용)
MEMORY_SHORT_TERM_BUFFER = 50  # 단기 메모리 버퍼
MEMORY_LONG_TERM_BUFFER = 50  # 장기 메모리 버퍼


# ===== 19. 보상 시스템 세부 설정 =====
# 보상 계산 파라미터
RETURN_CLIPPING_RANGE = (-0.2, 0.2)  # 수익률 클리핑 범위
RETURN_SCALING_FACTOR = 20  # 수익률 스케일링 계수
REWARD_CLIPPING_RANGE = (-2.0, 2.0)  # 최종 보상 클리핑 범위

# 다중 목표 보상 가중치
REWARD_COMPONENT_WEIGHTS = [
    0.3,   # 수익률 가중치
    0.3,   # 샤프 비율 가중치
    0.2,   # 최대 낙폭 가중치
    0.1,   # 거래 비용 가중치
    0.05,  # 변동성 가중치
    0.05,  # 기타 요소 가중치
]

# 보상 정규화 및 페널티
REWARD_NORMALIZATION_THRESHOLD = 100  # 정규화 시작 에피소드
CONCENTRATION_PENALTY_THRESHOLD = 0.4  # HHI 집중도 페널티 임계값
TARGET_VOLATILITY = 0.15  # 목표 변동성 수준


# ===== 20. 백테스터 및 시뮬레이션 =====
# 조기 종료 조건
EXTREME_RETURN_THRESHOLD = 3.0  # 극단적 수익률 표준편차
MAX_SIMULATION_ITERATIONS = 200  # 최대 시뮬레이션 반복

# 시뮬레이션 파라미터
SIMULATION_PARAMS = {
    "base_drift": 0.2,  # 기본 드리프트
    "base_volatility": 0.1,  # 기본 변동성
    "noise_scale": 0.3,  # 노이즈 스케일
    "trend_strength": 0.5,  # 트렌드 강도
}


# ===== 21. 시각화 세부 설정 =====
# 그래프 레이아웃
GRID_SPACING = {
    "hspace": 0.3,  # 수직 간격
    "wspace": 0.3,  # 수평 간격
}

# 폰트 및 스타일
TITLE_FONTSIZE = 16  # 제목 폰트 크기
LABEL_FONTSIZE = 12  # 라벨 폰트 크기
HIGH_DPI = 300  # 고해상도 출력용 DPI

# 투명도 설정
ALPHA_VALUES = {
    "light": 0.3,  # 연한 투명도
    "medium": 0.5,  # 중간 투명도
    "dark": 0.7,  # 진한 투명도
    "opaque": 1.0,  # 불투명
}


# ===== 22. XAI 분석 파라미터 =====
# 시간 임계값
REACTION_TIME_THRESHOLD = 6  # 반응 시간 임계값 (시간)
JSON_INDENT = 2  # JSON 출력 들여쓰기


# ===== 23. RL Tracker 설정 =====
# 추적 윈도우
MOVING_AVERAGE_WINDOW = 50  # 이동평균 윈도우 크기
MIN_EPISODES_FOR_STATS = 10  # 통계 계산 최소 에피소드

# 시각화 그리드
TRACKER_GRID_ROWS = 4  # 추적기 그리드 행 수
TRACKER_GRID_COLS = 3  # 추적기 그리드 열 수


# ===== 24. 출력 포맷 설정 =====
# 숫자 출력 정밀도
FORMAT_PRECISION = {
    "percentage": ".2%",  # 백분율: 소수점 2자리
    "float_2": ".2f",  # 실수: 소수점 2자리
    "float_3": ".3f",  # 실수: 소수점 3자리
    "float_4": ".4f",  # 실수: 소수점 4자리
}

# 시간 포맷
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"  # 기본 시간 포맷
DATE_FORMAT = "%Y-%m-%d"  # 날짜 포맷


# ===== 25. 초기화 값 =====
# 네트워크 가중치 초기화
WEIGHT_INIT_RANGE = (-0.1, 0.1)  # 가중치 초기화 범위
BIAS_INIT_VALUE = 0.0  # 편향 초기화 값
