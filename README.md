# BIPD: Behavioral Immune Portfolio Defense

생물학적 면역 시스템을 모방한 **설명 가능한 강화학습** 기반 포트폴리오 관리 시스템

## 🎯 프로젝트 개요

BIPD는 기존 강화학습 포트폴리오 최적화(PPO, A2C 등)와 달리 **생물학적 면역 시스템**의 동작 원리를 모방하여:

- **T-Cell**: 시장 위기 감지 (Isolation Forest)
- **B-Cell**: 위험별 전문화된 대응 전략 (Actor-Critic)
- **Memory Cell**: 과거 경험 기반 의사결정 지원

각 컴포넌트의 의사결정 과정이 **설명 가능**하여 금융 실무에서 활용 가능함을 목적으로 하고 있습니다.

## 🏗️ 시스템 아키텍처

```
Market Data → Feature Extraction (12D) → T-Cell (Crisis Detection)
                                              ↓
Memory Cell ← Portfolio Weights ← B-Cell Selection ← Crisis Level
     ↓                              ↑
 Historical                   Volatility/Correlation/Momentum
 Experience                        Specialists
```

## 📦 설치 및 실행

### 1. 환경 설정

```bash
cd bipd
pip install -r requirements.txt
```

### 2. 기본 실행

```bash
python main.py
```

### 3. 설정 변경

`config.py`에서 다음 항목들을 조정 가능:

- `SYMBOLS`: 투자 대상 주식
- `N_EPISODES`: 훈련 에피소드 수
- `ACTOR_LR`, `CRITIC_LR`: 학습률
- `INITIAL_CAPITAL`: 초기 자본

## 📊 12차원 시장 특성

1. **수익률 통계** (3개): 최근 수익률, 평균 수익률, 변동성
2. **기술적 지표** (4개): RSI, MACD, 볼린저 밴드, 거래량 비율
3. **시장 구조** (3개): 자산간 상관관계, 시장 베타, 최대 낙폭
4. **모멘텀** (2개): 단기 모멘텀, 장기 모멘텀

## 🧬 면역 시스템 컴포넌트

### T-Cell (위기 감지)

```python
# Isolation Forest 기반 이상 탐지
crisis_level = tcell.detect_crisis(market_features)
explanation = tcell.get_anomaly_explanation(market_features)
```

### B-Cell (전문화 전략)

```python
# 위험 유형별 Actor-Critic 네트워크
bcells = {
    'volatility': BCell('volatility', state_dim, action_dim),    # 고위기 특화
    'correlation': BCell('correlation', state_dim, action_dim),  # 중위기 특화
    'momentum': BCell('momentum', state_dim, action_dim)         # 저위기 특화
}
```

### Memory Cell (경험 활용)

```python
# 코사인 유사도 기반 과거 경험 회상
similar_experiences = memory.recall(current_state, crisis_level, k=5)
guidance = memory.get_memory_guidance(current_state, crisis_level)
```

## 🎛️ 하이퍼파라미터

주요 하이퍼파라미터들 (일반적 강화학습 권장값 적용):

```python
# 강화학습
ACTOR_LR = 3e-4          # Actor 학습률
CRITIC_LR = 6e-4         # Critic 학습률
GAMMA = 0.99             # 할인 팩터
BATCH_SIZE = 64          # 배치 크기
BUFFER_SIZE = 10000      # Experience Replay 크기

# 탐험-활용
EPSILON_START = 0.9      # 초기 탐험률
EPSILON_END = 0.05       # 최소 탐험률
EPSILON_DECAY = 0.995    # 탐험률 감소

# 면역 시스템
TCELL_CONTAMINATION = 0.1    # T-Cell 이상치 비율
MEMORY_CAPACITY = 500        # Memory Cell 용량
CRISIS_HIGH = 0.7           # 고위기 임계값
CRISIS_MEDIUM = 0.4         # 중위기 임계값
```

## 📈 성과 평가

시스템은 다음 메트릭으로 평가됩니다:

- **샤프 비율**: 위험 대비 수익률
- **최대 낙폭**: 최대 손실 구간
- **변동성**: 포트폴리오 리스크
- **벤치마크 대비 성과**: 동일가중 포트폴리오 vs BIPD

## 🔍 설명 가능성 (XAI)

### 의사결정 설명

```python
explanation = immune_system.get_system_explanation(state)
```

**출력 예시:**

```json
{
	"crisis_detection": {
		"crisis_level": 0.73,
		"is_anomaly": true,
		"top_anomaly_features": [2, 7, 11]
	},
	"strategy_selection": {
		"selected_strategy": "volatility",
		"specialization_scores": {
			"volatility": 0.73,
			"correlation": 0.41,
			"momentum": 0.27
		}
	},
	"memory_system": {
		"memory_count": 342,
		"avg_reward": 0.15
	}
}
```

## 📁 프로젝트 구조

```
bipd/
├── main.py              # 메인 실행 스크립트
├── config.py            # 설정 파일
├── requirements.txt     # 의존성
├── agents/              # 면역 세포
│   ├── tcell.py        # T-Cell (위기 감지)
│   ├── bcell.py        # B-Cell (전략 실행)
│   └── memory.py       # Memory Cell (경험 저장)
├── core/                # 핵심 시스템
│   ├── environment.py  # 포트폴리오 환경
│   ├── system.py       # 통합 면역 시스템
│   └── trainer.py      # 강화학습 훈련
├── data/                # 데이터 처리
│   ├── loader.py       # 시장 데이터 로드
│   └── features.py     # 특성 추출
└── utils/               # 유틸리티
    ├── logger.py       # 로깅 시스템
    └── metrics.py      # 성과 측정
```

## 🚀 실행 결과

성공적인 실행 시 다음 결과물을 생성:

1. **모델 파일**: `models/bipd_final_model_*`
2. **로그 파일**: `logs/bipd_*.log`
3. **시각화**: `models/visualizations/training_results_*.png`
4. **성과 보고서**: 콘솔 출력으로 제공

## 🔬 연구 기여도

- **새로운 메타포**: 생물학적 면역 시스템 → 포트폴리오 관리
- **전문화 메커니즘**: 위험 유형별 특화된 에이전트
- **설명 가능성**: 각 의사결정의 근거 제공
- **메모리 활용**: 과거 경험 기반 적응적 학습
