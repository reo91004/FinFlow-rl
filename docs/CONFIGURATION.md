# Configuration Guide

FinFlow-RL의 상세 설정 가이드 및 파라미터 튜닝

## 목차
- [설정 파일 구조](#설정-파일-구조)
- [데이터 설정](#데이터-설정)
- [모델 아키텍처](#모델-아키텍처)
- [학습 파라미터](#학습-파라미터)
- [목적함수 설정](#목적함수-설정)
- [환경 설정](#환경-설정)
- [모니터링 설정](#모니터링-설정)
- [파라미터 튜닝 가이드](#파라미터-튜닝-가이드)
- [시장별 최적 설정](#시장별-최적-설정)

---

## 설정 파일 구조

### 기본 설정 파일
`configs/default.yaml`

```yaml
# 최상위 섹션
seed: 42              # 재현성을 위한 시드
device: auto          # cpu|cuda|mps|auto

data:                 # 데이터 관련
bcell:               # B-Cell 모델
tcell:               # T-Cell 위기 감지
memory:              # Memory Cell
objectives:          # 목적함수
train:               # 학습 설정
env:                 # 환경 설정
eval:                # 평가 설정
monitoring:          # 모니터링
backtest:            # 백테스팅
xai:                 # XAI 설정
```

### 설정 우선순위

1. 명령행 인자 (최우선)
2. 환경 변수
3. 설정 파일
4. 기본값

```bash
# 명령행이 설정 파일을 오버라이드
python main.py --config configs/default.yaml --lr 1e-4
```

---

## 데이터 설정

### 종목 선택

```yaml
data:
  # 방법 1: 직접 지정
  symbols: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

  # 방법 2: 인덱스 사용
  index: "DOW30"  # SP500, NASDAQ100, DOW30

  # 방법 3: 섹터별
  sectors:
    technology: ["AAPL", "MSFT", "NVDA"]
    healthcare: ["JNJ", "UNH", "PFE"]
    finance: ["JPM", "BAC", "GS"]
```

### 기간 설정

```yaml
data:
  # 학습 데이터
  start: "2015-01-01"      # 시작일
  end: "2020-12-31"        # 종료일 (train+val)

  # 테스트 데이터
  test_start: "2021-01-01"
  test_end: "2024-12-31"

  # 검증 데이터
  val_ratio: 0.2           # train 데이터에서 val 비율 (0.2 = 20%)
```

### 데이터 전처리

```yaml
data:
  interval: "1d"            # 1m, 5m, 1h, 1d
  cache_dir: "data/cache"   # 캐시 디렉토리

  # 전처리 옵션
  preprocessing:
    handle_missing: "forward_fill"  # forward_fill, interpolate, drop
    remove_outliers: true
    outlier_threshold: 5.0          # IQR 배수
    normalize: true
    normalization_method: "zscore"  # zscore, minmax, robust
```

### 피처 엔지니어링

```yaml
features:
  window: 20               # 룩백 윈도우

  # 피처 차원
  dimensions:
    returns: 3             # 수익률 관련
    technical: 4          # 기술적 지표
    structure: 3          # 시장 구조
    momentum: 2           # 모멘텀

  # 기술적 지표
  indicators:
    - "returns"           # 단순 수익률
    - "log_returns"       # 로그 수익률
    - "volatility"        # 역사적 변동성
    - "rsi"              # RSI
    - "macd"             # MACD
    - "bollinger"        # 볼린저 밴드
    - "volume"           # 거래량
    - "correlation"      # 상관관계
```

---

## 모델 아키텍처

### B-Cell 설정

```yaml
bcell:
  # 네트워크 구조
  actor_hidden: [256, 256]     # Actor 은닉층
  critic_hidden: [256, 256]    # Critic 은닉층
  n_quantiles: 32              # 분위수 개수

  # 활성화 함수
  activation: "relu"           # relu, tanh, elu, selu
  output_activation: "softmax" # 포트폴리오 가중치용

  # 정규화
  use_batch_norm: false
  use_layer_norm: true
  dropout: 0.1
```

### IQL 파라미터

```yaml
bcell:
  # IQL 오프라인 학습
  offline_algo: "iql"
  iql_expectile: 0.7          # 0.5-0.9, 클수록 보수적
  iql_temperature: 3.0        # 1.0-10.0, 클수록 다양한 행동
  iql_clip_score: 100.0       # 그래디언트 클리핑
```

### SAC 파라미터

```yaml
bcell:
  # SAC 온라인 학습
  online_algo: "dist_sac_cql"

  # 엔트로피 정규화
  alpha_init: 0.75            # 초기 엔트로피 계수
  alpha_min: 5.0e-4          # 최소값
  alpha_max: 0.5             # 최대값
  target_entropy_ratio: 0.5   # 목표 엔트로피 비율

  # 타겟 네트워크
  tau: 0.005                 # Polyak 평균 계수
  update_frequency: 1        # 업데이트 빈도
```

### CQL 설정

```yaml
bcell:
  # Conservative Q-Learning
  cql_alpha_start: 5.0       # 시작 강도
  cql_alpha_end: 10.0        # 종료 강도
  cql_num_samples: 8         # 샘플 수
  cql_include_current: true  # 현재 정책 포함
  cql_temp: 1.0             # CQL 온도
```

### Multi-Expert 설정

```yaml
bcell:
  # Soft MoE
  n_experts: 5               # 전문가 수
  expert_types:
    - "volatility"          # 변동성 전문가
    - "correlation"         # 상관관계 전문가
    - "momentum"            # 모멘텀 전문가
    - "defensive"           # 방어 전문가
    - "growth"              # 성장 전문가

  # Gating Network
  gating_hidden_dim: 128
  gating_temperature: 1.0    # 소프트맥스 온도
  gating_noise_std: 0.01    # 탐험용 노이즈
```

---

## 학습 파라미터

### 기본 학습 설정

```yaml
train:
  # 학습률
  actor_lr: 3.0e-4
  critic_lr: 3.0e-4
  alpha_lr: 3.0e-4

  # 학습률 스케줄러
  lr_scheduler: "cosine"     # none, step, exponential, cosine
  lr_decay_rate: 0.9
  lr_decay_steps: 1000

  # 옵티마이저
  optimizer: "adam"          # adam, sgd, rmsprop
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_eps: 1e-8
  weight_decay: 0.0
```

### 오프라인 학습

```yaml
train:
  # 데이터 수집
  offline_episodes: 500           # 수집 에피소드
  force_recollect_offline: false  # 강제 재수집

  # IQL 학습
  offline_training_epochs: 50     # 학습 에폭
  offline_steps_per_epoch: 1000   # 에폭당 스텝
  offline_batch_size: 512         # 배치 크기
```

### 온라인 학습

```yaml
train:
  # 에피소드 설정
  online_episodes: 1000           # 총 에피소드
  max_steps_per_episode: 252      # 에피소드당 최대 스텝 (1년)

  # 배치 설정
  online_batch_size: 512
  min_buffer_size: 256            # 학습 시작 최소 버퍼

  # 업데이트 설정
  updates_per_step: 1              # 스텝당 업데이트 횟수
  gradient_steps: 1                # 그래디언트 스텝
```

### Experience Replay

```yaml
train:
  # 버퍼 설정
  buffer_size: 100000              # 리플레이 버퍼 크기
  prioritized_replay: true         # 우선순위 리플레이 사용

  # PER 파라미터
  per_alpha: 0.6                   # 우선순위 지수
  per_beta: 0.4                    # 중요도 샘플링 지수
  per_beta_end: 1.0                # 베타 최종값
  per_eps: 1e-6                    # 우선순위 최소값
```

### 체크포인팅

```yaml
train:
  # 저장 설정
  checkpoint_interval: 100         # 체크포인트 간격
  save_best: true                  # 최고 성능 저장
  save_latest: true                # 최신 저장
  max_checkpoints: 5               # 최대 보관 수

  # 평가 설정
  eval_interval: 50                # 평가 간격
  eval_episodes: 10                # 평가 에피소드 수
```

---

## 목적함수 설정

### Differential Sharpe

```yaml
objectives:
  # Sharpe Ratio 최적화
  sharpe_beta: 1.0                 # Sharpe 가중치
  sharpe_ema_alpha: 0.99           # EMA 계수
  sharpe_epsilon: 1.0e-8          # 수치 안정성

  # 계산 방법
  use_differential: true           # Differential Sharpe 사용
  sharpe_window: 252               # 계산 윈도우
```

### 리스크 제약

```yaml
objectives:
  # CVaR 제약
  cvar_alpha: 0.05                 # 5% CVaR
  cvar_target: -0.01               # 목표 CVaR
  lambda_cvar: 5.0                 # CVaR 페널티 가중치

  # VaR 제약
  var_alpha: 0.05                  # 5% VaR
  var_limit: -0.02                 # VaR 한계

  # 최대 낙폭
  max_drawdown_limit: 0.25         # 25% 한계
  lambda_dd: 0.0                   # 낙폭 페널티
```

### 거래 비용

```yaml
objectives:
  # 회전율 페널티
  lambda_turn: 0.1                 # 회전율 페널티 가중치
  turnover_limit: 2.0              # 연간 회전율 한계

  # 거래 비용
  transaction_cost: 0.001           # 10 bps
  slippage: 0.0005                 # 5 bps
```

### 보상 설계

```yaml
objectives:
  # 보상 구성
  reward_type: "sharpe"             # simple, sharpe, sortino, custom

  # 보상 정규화
  r_clip: 5.0                      # 보상 클리핑
  reward_ema_alpha: 0.99           # 보상 EMA
  normalize_rewards: true          # 정규화 여부
```

---

## 환경 설정

### 포트폴리오 환경

```yaml
env:
  # 초기 설정
  initial_capital: 1000000         # 초기 자본
  currency: "USD"                  # 통화

  # 포지션 제약
  max_leverage: 1.0                # 최대 레버리지
  allow_short: false               # 공매도 허용
  max_position_size: 0.3           # 단일 자산 최대 비중
  min_position_size: 0.0           # 최소 비중
```

### 거래 제약

```yaml
env:
  # 거래 빈도
  max_turnover: 0.9                # 일일 최대 회전율
  no_trade_band: 0.0005           # 무거래 구간 (0.05%)

  # 거래 시간
  trading_hours: "regular"         # regular, extended, 24/7
  settlement: "T+1"                # T+0, T+1, T+2
```

---

## 모니터링 설정

### 안정성 모니터링

```yaml
monitoring:
  stability:
    enabled: true
    window_size: 100               # 모니터링 윈도우

    # 이상치 감지
    n_sigma: 3.0                   # 시그마 임계값
    max_weight_change: 0.2         # 최대 가중치 변화
    min_effective_assets: 3        # 최소 유효 자산

    # Q값 체크
    q_value_check: true
    q_value_threshold: 100.0       # Q값 상한

    # 엔트로피 체크
    entropy_check: true
    entropy_min: 0.1               # 최소 엔트로피
```

### 로깅 설정

```yaml
monitoring:
  # 로깅 레벨
  log_level: "INFO"                # DEBUG, INFO, WARNING, ERROR
  log_interval: 1                  # 로그 간격 (에피소드)

  # 메트릭 추적
  track_metrics: true
  metrics_file: "metrics.jsonl"

  # 시각화
  use_tensorboard: true
  use_wandb: false
  wandb_project: "finflow-rl"
```

### 자동 개입

```yaml
monitoring:
  # 자동 조정
  auto_intervention: true
  intervention_threshold: 3.0      # 3-시그마 이상

  # 조정 방법
  rollback_on_divergence: true    # 발산 시 롤백
  reduce_lr_on_plateau: true      # 정체 시 학습률 감소
  early_stopping: true             # 조기 종료
  patience: 50                     # 인내 에피소드
```

---

## 파라미터 튜닝 가이드

### 중요도 순위

#### 🔴 매우 중요 (큰 영향)
1. **alpha_init** (0.2-1.0): 탐험 vs 활용
2. **cql_alpha** (1.0-10.0): 보수성
3. **lr** (1e-4 ~ 1e-3): 학습 속도
4. **batch_size** (128-512): 안정성

#### 🟡 중요 (중간 영향)
5. **iql_expectile** (0.5-0.9): 오프라인 보수성
6. **gamma** (0.95-0.99): 장기 계획
7. **tau** (0.001-0.01): 타겟 업데이트
8. **n_quantiles** (8-64): 분포 정밀도

#### 🟢 덜 중요 (미세 조정)
9. **hidden_dim** (128-512): 모델 용량
10. **dropout** (0.0-0.3): 정규화
11. **gradient_clip** (0.5-5.0): 안정성

### 그리드 서치 예시

```python
param_grid = {
    'alpha_init': [0.5, 0.75, 1.0],
    'cql_alpha': [5.0, 7.5, 10.0],
    'lr': [1e-4, 3e-4, 1e-3],
    'batch_size': [256, 512]
}

# Optuna로 자동 튜닝
study = optuna.create_study(direction='maximize')
study.optimize(
    lambda trial: train_and_evaluate(
        alpha_init=trial.suggest_float('alpha_init', 0.2, 1.0),
        cql_alpha=trial.suggest_float('cql_alpha', 1.0, 10.0),
        lr=trial.suggest_loguniform('lr', 1e-4, 1e-2)
    ),
    n_trials=100
)
```

---

## 시장별 최적 설정

### 상승장 (Bull Market)

```yaml
# 공격적 설정
bcell:
  alpha_init: 0.5          # 낮은 탐험
  cql_alpha: 3.0          # 약한 보수성

objectives:
  lambda_cvar: 1.0        # CVaR 완화
  lambda_turn: 0.05       # 회전율 허용

train:
  online_batch_size: 256   # 빠른 적응
```

### 하락장 (Bear Market)

```yaml
# 방어적 설정
bcell:
  alpha_init: 1.0         # 높은 탐험
  cql_alpha: 10.0        # 강한 보수성

objectives:
  lambda_cvar: 10.0      # CVaR 강화
  lambda_turn: 0.2       # 회전율 제한

env:
  max_position_size: 0.1  # 포지션 제한
```

### 횡보장 (Sideways Market)

```yaml
# 균형 설정
bcell:
  alpha_init: 0.75        # 중간 탐험
  cql_alpha: 5.0         # 중간 보수성

objectives:
  sharpe_beta: 2.0       # Sharpe 중시
  lambda_turn: 0.15      # 적당한 회전율
```

### 고변동성 (High Volatility)

```yaml
# 안정성 중시
bcell:
  n_quantiles: 64         # 정밀한 분포
  cql_num_samples: 16     # 더 많은 샘플

monitoring:
  stability:
    n_sigma: 2.0          # 엄격한 이상치
    auto_intervention: true

train:
  gradient_clip: 0.5      # 강한 클리핑
```

---

## 커스텀 설정

### 설정 상속

```yaml
# configs/aggressive.yaml
# 기본 설정 상속 후 오버라이드
extends: default.yaml

bcell:
  alpha_init: 0.3
  cql_alpha: 1.0

objectives:
  lambda_cvar: 0.5
```

### 환경 변수

```bash
# 환경 변수로 오버라이드
export FINFLOW_LR=1e-4
export FINFLOW_BATCH_SIZE=512

python main.py --mode train
```

### 프로그래밍 방식

```python
from src.core.trainer import FinFlowTrainer
import yaml

# 설정 로드
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

# 프로그래밍 방식으로 수정
config['bcell']['alpha_init'] = 0.8
config['train']['online_episodes'] = 500

# 학습 실행
trainer = FinFlowTrainer(config)
trainer.train()
```

---

## 검증 및 테스트

### 설정 검증

```python
from src.utils.config_validator import validate_config

# 설정 유효성 검사
is_valid, errors = validate_config(config)
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

### A/B 테스트

```python
# 두 설정 비교
config_a = load_config('configs/conservative.yaml')
config_b = load_config('configs/aggressive.yaml')

results_a = train_and_evaluate(config_a)
results_b = train_and_evaluate(config_b)

# 통계적 비교
from scipy import stats
t_stat, p_value = stats.ttest_ind(
    results_a['returns'],
    results_b['returns']
)
```

---

## 베스트 프랙티스

1. **단계적 조정**: 한 번에 하나씩 파라미터 변경
2. **로그 스케일**: 학습률은 로그 스케일로 탐색
3. **시드 고정**: 재현성을 위해 시드 고정
4. **교차 검증**: 여러 기간으로 검증
5. **민감도 분석**: 파라미터 민감도 확인

---

*더 자세한 내용은 [TRAINING.md](TRAINING.md) 참조*

---

*Last Updated: 2025-01-22*