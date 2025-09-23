# API Reference

FinFlow-RL의 주요 클래스와 함수에 대한 API 레퍼런스

## 목차
- [Agents](#agents)
  - [BCell](#bcell)
  - [TCell](#tcell)
  - [MemoryCell](#memorycell)
- [Core](#core)
  - [FinFlowTrainer](#finflowtrainer)
  - [PortfolioEnv](#portfolioenv)
  - [IQL](#iql)
- [Analysis](#analysis)
  - [XAIAnalyzer](#xaianalyzer)
  - [RealisticBacktester](#realisticbacktester)

---

## Agents

### BCell

**경로**: `src/algorithms/online/b_cell.py`

REDQ 또는 TQC 알고리즘을 사용하는 적응형 포트폴리오 전략 에이전트

#### 클래스 정의
```python
class BCell:
    def __init__(
        self,
        state_dim: int,          # 상태 공간 차원 (43)
        action_dim: int,         # 행동 공간 차원 (30)
        config: Dict[str, Any],  # 설정 딕셔너리
        device: torch.device     # 연산 디바이스
    )
```

#### 주요 메서드

##### `act(state: torch.Tensor, deterministic: bool = False) -> np.ndarray`
주어진 상태에서 행동 선택

**매개변수**:
- `state`: 현재 시장 상태 (shape: [batch_size, 43])
- `deterministic`: True면 평균 행동, False면 샘플링

**반환값**: 포트폴리오 가중치 (shape: [batch_size, 30])

##### `update_iql(batch: Dict) -> Dict[str, float]`
IQL 오프라인 사전학습 업데이트

**매개변수**:
- `batch`: 오프라인 데이터 배치
  - `states`: 상태 (shape: [batch_size, 43])
  - `actions`: 행동 (shape: [batch_size, 30])
  - `rewards`: 보상 (shape: [batch_size])
  - `next_states`: 다음 상태
  - `dones`: 종료 플래그

**반환값**: 손실 딕셔너리 (`q_loss`, `v_loss`, `actor_loss`)

##### `update_sac(batch: Dict) -> Dict[str, float]`
Distributional SAC 온라인 업데이트

**매개변수**:
- `batch`: 경험 리플레이 배치

**반환값**: 손실 딕셔너리 (`critic_loss`, `actor_loss`, `alpha_loss`, `cql_loss`)

##### `save(path: str)`
SafeTensors 형식으로 모델 저장

##### `load(path: str)`
SafeTensors 모델 로드

#### 사용 예시
```python
from src.agents.b_cell import BCell

# 초기화
bcell = BCell(
    state_dim=43,
    action_dim=30,
    config=config['bcell']
)

# 행동 선택
state = torch.randn(1, 43)
action = bcell.act(state, deterministic=False)

# IQL 학습
iql_losses = bcell.update_iql(offline_batch)

# SAC 학습
sac_losses = bcell.update_sac(online_batch)
```

---

### TCell

**경로**: `src/agents/t_cell.py`

Isolation Forest 기반 위기 감지 시스템

#### 클래스 정의
```python
class TCell:
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        config: Dict = None
    )
```

#### 주요 메서드

##### `detect_crisis(features: np.ndarray) -> Tuple[float, np.ndarray]`
위기 수준 감지

**매개변수**:
- `features`: 시장 특징 (shape: [batch_size, n_features])

**반환값**:
- `crisis_level`: 위기 수준 (0.0 ~ 1.0)
- `shap_values`: SHAP 설명 값

##### `explain_anomaly(features: np.ndarray) -> Dict`
이상치 원인 분석

**반환값**:
- `top_features`: 주요 위기 요인
- `contribution`: 각 요인의 기여도

#### 사용 예시
```python
from src.agents.t_cell import TCell

tcell = TCell(n_estimators=100, contamination=0.1)

# 위기 감지
crisis_level, shap_values = tcell.detect_crisis(market_features)

# 위기 설명
explanation = tcell.explain_anomaly(market_features)
print(f"위기 수준: {crisis_level:.2f}")
print(f"주요 원인: {explanation['top_features']}")
```

---

### MemoryCell

**경로**: `src/agents/memory.py`

k-NN 기반 경험 검색 시스템

#### 클래스 정의
```python
class MemoryCell:
    def __init__(
        self,
        capacity: int = 50000,
        embedding_dim: int = 32,
        k_neighbors: int = 5
    )
```

#### 주요 메서드

##### `store(experience: Dict)`
경험 저장

##### `retrieve(query_state: np.ndarray, k: int = 5) -> List[Dict]`
유사한 경험 검색

**매개변수**:
- `query_state`: 쿼리 상태
- `k`: 검색할 이웃 수

**반환값**: k개의 유사한 경험

##### `update_priorities(indices: List[int], td_errors: np.ndarray)`
우선순위 업데이트

#### 사용 예시
```python
from src.agents.memory import MemoryCell

memory = MemoryCell(capacity=50000, k_neighbors=5)

# 경험 저장
memory.store({
    'state': state,
    'action': action,
    'reward': reward,
    'metadata': {'regime': 'bull_market'}
})

# 유사 경험 검색
similar_experiences = memory.retrieve(current_state, k=5)
```

---

## Core

### FinFlowTrainer

**경로**: `src/core/trainer.py`

통합 학습 파이프라인 관리자

#### 클래스 정의
```python
class FinFlowTrainer:
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[FinFlowLogger] = None
    )
```

#### 주요 메서드

##### `train() -> Dict`
전체 학습 파이프라인 실행 (IQL → SAC)

**반환값**: 학습 결과 메트릭

##### `train_iql(dataset: OfflineDataset) -> Dict`
IQL 오프라인 사전학습

##### `train_online(env: PortfolioEnv) -> Dict`
B-Cell 온라인 미세조정

##### `save_checkpoint(path: str, metrics: Dict)`
체크포인트 저장

##### `load_checkpoint(path: str)`
체크포인트 로드

#### 사용 예시
```python
from src.core.trainer import FinFlowTrainer

trainer = FinFlowTrainer(config)

# 전체 파이프라인
results = trainer.train()

# 개별 단계
iql_results = trainer.train_iql(offline_dataset)
sac_results = trainer.train_online(env)
```

---

### PortfolioEnv

**경로**: `src/core/env.py`

T+1 결제 포트폴리오 거래 환경

#### 클래스 정의
```python
class PortfolioEnv(gym.Env):
    def __init__(
        self,
        config: Dict,
        mode: str = 'train'  # 'train' or 'test'
    )
```

#### 주요 속성
- `observation_space`: Box(43,) - 상태 공간
- `action_space`: Box(30,) - 행동 공간

#### 주요 메서드

##### `reset() -> np.ndarray`
환경 초기화

**반환값**: 초기 상태

##### `step(action: np.ndarray) -> Tuple`
행동 실행

**매개변수**:
- `action`: 포트폴리오 가중치

**반환값**:
- `next_state`: 다음 상태
- `reward`: 보상
- `done`: 에피소드 종료 여부
- `info`: 추가 정보

#### 사용 예시
```python
from src.core.env import PortfolioEnv

env = PortfolioEnv(config, mode='train')

state = env.reset()
for _ in range(100):
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    if done:
        break
    state = next_state
```

---

### IQL

**경로**: `src/core/iql.py`

Implicit Q-Learning 구현

#### 클래스 정의
```python
class IQL:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        expectile: float = 0.7,
        temperature: float = 3.0
    )
```

#### 주요 메서드

##### `compute_value_loss(batch: Dict) -> torch.Tensor`
가치 함수 손실 계산

##### `compute_q_loss(batch: Dict) -> torch.Tensor`
Q 함수 손실 계산

##### `compute_policy_loss(batch: Dict) -> torch.Tensor`
정책 손실 계산

#### 사용 예시
```python
from src.core.iql import IQL

iql = IQL(
    state_dim=43,
    action_dim=30,
    expectile=0.7
)

# 손실 계산
v_loss = iql.compute_value_loss(batch)
q_loss = iql.compute_q_loss(batch)
p_loss = iql.compute_policy_loss(batch)
```

---

## Analysis

### XAIAnalyzer

**경로**: `src/analysis/xai.py`

SHAP 기반 설명 가능한 AI 분석기

#### 클래스 정의
```python
class XAIAnalyzer:
    def __init__(
        self,
        model: BCell,
        config: Dict
    )
```

#### 주요 메서드

##### `analyze_decision(state: np.ndarray, action: np.ndarray) -> Dict`
개별 의사결정 분석

**반환값**:
- `shap_values`: SHAP 값
- `feature_importance`: 피처 중요도
- `decision_card`: 의사결정 설명 카드

##### `generate_counterfactual(state: np.ndarray) -> Dict`
반사실적 분석

**반환값**:
- `counterfactual_state`: 대안 상태
- `counterfactual_action`: 대안 행동
- `impact`: 예상 영향

##### `portfolio_attribution(returns: pd.DataFrame) -> Dict`
포트폴리오 수익 귀속

**반환값**:
- `asset_contribution`: 자산별 기여도
- `factor_attribution`: 팩터별 귀속

#### 사용 예시
```python
from src.analysis.xai import XAIAnalyzer

analyzer = XAIAnalyzer(model, config)

# 의사결정 분석
explanation = analyzer.analyze_decision(state, action)
print(f"주요 요인: {explanation['feature_importance']}")

# 반사실적 분석
counterfactual = analyzer.generate_counterfactual(state)
print(f"대안 행동: {counterfactual['counterfactual_action']}")

# 수익 귀속
attribution = analyzer.portfolio_attribution(returns)
```

---

### RealisticBacktester

**경로**: `src/analysis/backtest.py`

현실적 거래 비용을 반영한 백테스터

#### 클래스 정의
```python
class RealisticBacktester:
    def __init__(
        self,
        config: Dict = None
    )
```

#### 주요 메서드

##### `backtest(strategy: Callable, data: pd.DataFrame, initial_capital: float) -> Dict`
백테스트 실행

**매개변수**:
- `strategy`: 전략 함수
- `data`: 가격 데이터
- `initial_capital`: 초기 자본

**반환값**:
- `metrics`: 성과 지표
- `equity_curve`: 자산 가치 곡선
- `trades`: 거래 내역

##### `calculate_costs(trade_value: float, volume: float) -> float`
거래 비용 계산

#### 사용 예시
```python
from src.analysis.backtest import RealisticBacktester

backtester = RealisticBacktester(config)

def my_strategy(data, positions, t):
    # 전략 로직
    return new_weights

results = backtester.backtest(
    strategy=my_strategy,
    data=price_data,
    initial_capital=1000000
)

print(f"Sharpe: {results['metrics']['sharpe']:.2f}")
print(f"총 비용: ${results['metrics']['total_costs']:,.2f}")
```

---

## 유틸리티

### FinFlowLogger

**경로**: `src/utils/logger.py`

세션 기반 로깅 시스템

#### 사용 예시
```python
from src.utils.logger import FinFlowLogger

logger = FinFlowLogger("my_session")
logger.info("학습 시작")
logger.log_metrics({
    'episode': 100,
    'sharpe': 1.82,
    'cvar': -0.015
})
```

### set_seed

**경로**: `src/utils/seed.py`

재현 가능한 실험을 위한 시드 설정

```python
from src.utils.seed import set_seed

set_seed(42)  # 모든 랜덤 시드 고정
```

---

## 설정 구조

### TrainingConfig

**경로**: `src/core/trainer.py`

```python
@dataclass
class TrainingConfig:
    # 데이터
    symbols: List[str]
    start_date: str
    end_date: str

    # IQL
    iql_epochs: int = 100
    iql_batch_size: int = 256
    iql_expectile: float = 0.7

    # SAC
    sac_episodes: int = 1000
    sac_batch_size: int = 256

    # 모델
    hidden_dim: int = 256
    n_quantiles: int = 32

    # 학습
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
```

---

## 에러 처리

모든 API는 다음과 같은 표준 예외를 발생시킬 수 있음:

- `ValueError`: 잘못된 매개변수
- `RuntimeError`: 실행 중 오류
- `FileNotFoundError`: 체크포인트 파일 없음
- `MemoryError`: 메모리 부족

예외 처리 예시:
```python
try:
    model = BCell.load("checkpoint.pt")
except FileNotFoundError:
    logger.error("체크포인트 파일을 찾을 수 없음")
    model = BCell(state_dim=43, action_dim=30, config=config)
```

---

*더 자세한 구현 예시는 [TRAINING.md](TRAINING.md)와 [EVALUATION.md](EVALUATION.md) 참조*