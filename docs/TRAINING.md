# Training Guide

FinFlow-RL 학습 파이프라인 상세 가이드

## 목차
- [개요](#개요)
- [학습 파이프라인](#학습-파이프라인)
- [Phase 1: IQL 오프라인 사전학습](#phase-1-iql-오프라인-사전학습)
- [Phase 2: B-Cell 온라인 미세조정](#phase-2-b-cell-온라인-미세조정)
- [학습 모니터링](#학습-모니터링)
- [하이퍼파라미터 튜닝](#하이퍼파라미터-튜닝)
- [체크포인팅](#체크포인팅)
- [트러블슈팅](#트러블슈팅)
- [성능 최적화](#성능-최적화)

---

## 개요

FinFlow-RL은 2단계 학습 파이프라인을 사용한다:

1. **IQL 오프라인 사전학습**: 과거 데이터로 안정적인 가치 함수 학습
2. **B-Cell 온라인 미세조정**: Distributional SAC + CQL로 실시간 적응

이 접근법의 장점:
- Cold-start 문제 해결
- 샘플 효율성 향상
- 안정적인 학습 시작점

## 학습 파이프라인

### 전체 파이프라인 실행

#### 방법 1: main.py 사용 (권장)
```bash
python main.py --mode train \
    --config configs/default.yaml \
    --iql-epochs 100 \
    --sac-episodes 1000
```

#### 방법 2: scripts/train.py 사용
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --mode full  # full = IQL + SAC
```

### 파이프라인 흐름도

```
데이터 수집 → IQL 사전학습 → B-Cell 초기화 → 온라인 미세조정 → 모델 저장
     ↓            ↓              ↓               ↓              ↓
  오프라인     가치함수      정책 전이      실시간 적응    SafeTensors
   데이터       학습                         + CQL
```

---

## Phase 1: IQL 오프라인 사전학습

### 1.1 오프라인 데이터 수집

#### 자동 수집 (기본)
```python
# trainer.py 내부에서 자동 처리
trainer = FinFlowTrainer(config)
trainer.collect_offline_data()  # 자동으로 캐싱됨
```

#### 수동 수집
```bash
# 별도로 오프라인 데이터 수집
python scripts/pretrain_iql.py \
    --collect-episodes 1000 \
    --save-path data/offline/dataset.npz
```

### 1.2 IQL 알고리즘

IQL(Implicit Q-Learning)의 핵심 개념:

```python
# 가치 함수 학습 (Expectile Regression)
def value_loss(v, q, expectile=0.7):
    diff = q - v
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * diff**2).mean()

# Q 함수 학습 (TD Learning)
def q_loss(q, r, next_v, gamma=0.99):
    target = r + gamma * next_v
    return F.mse_loss(q, target.detach())

# 정책 학습 (Advantage Weighted)
def policy_loss(log_prob, q, v, temperature=3.0):
    advantage = q - v
    weight = torch.exp(advantage / temperature)
    return -(weight.detach() * log_prob).mean()
```

### 1.3 학습 설정

#### 중요 하이퍼파라미터
```yaml
# configs/default.yaml
train:
  offline_episodes: 500        # 데이터 수집 에피소드
  offline_training_epochs: 50  # IQL 학습 에폭
  offline_batch_size: 512      # 배치 크기

bcell:
  iql_expectile: 0.7           # 기댓값 회귀 파라미터
  iql_temperature: 3.0         # 정책 온도
```

### 1.4 실행 예시

```python
from src.core.trainer import FinFlowTrainer

# IQL만 학습
trainer = FinFlowTrainer(config)
iql_results = trainer.train_iql()

print(f"IQL V-Loss: {iql_results['v_loss']:.4f}")
print(f"IQL Q-Loss: {iql_results['q_loss']:.4f}")
print(f"IQL Policy Loss: {iql_results['policy_loss']:.4f}")
```

---

## Phase 2: B-Cell 온라인 미세조정

### 2.1 B-Cell 아키텍처

B-Cell은 5개의 전문 전략을 가진 Multi-Expert System:

1. **Volatility Expert**: 변동성 관리
2. **Correlation Expert**: 상관관계 최적화
3. **Momentum Expert**: 추세 추종
4. **Defensive Expert**: 방어적 포지셔닝
5. **Growth Expert**: 성장 추구

### 2.2 Distributional SAC

분위수 기반 분포적 강화학습:

```python
# Quantile Critic 네트워크
class QuantileCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles=32):
        self.n_quantiles = n_quantiles
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_quantiles)
        )

    def forward(self, state, action):
        # Returns: [batch_size, n_quantiles]
        return self.network(torch.cat([state, action], dim=-1))
```

### 2.3 CQL 정규화

Conservative Q-Learning으로 과대평가 방지:

```python
def cql_loss(q_values, batch_actions, cql_alpha=5.0):
    # 현재 정책의 Q값
    current_q = q_values.gather(-1, batch_actions)

    # 모든 행동의 logsumexp
    logsumexp = torch.logsumexp(q_values, dim=-1)

    # CQL 페널티
    cql_penalty = (logsumexp - current_q).mean()

    return cql_alpha * cql_penalty
```

### 2.4 학습 루프

```python
# 온라인 미세조정
for episode in range(sac_episodes):
    state = env.reset()
    episode_return = 0

    for step in range(max_steps):
        # 행동 선택
        action = bcell.act(state)

        # 환경 상호작용
        next_state, reward, done, info = env.step(action)

        # 버퍼에 저장
        buffer.push(state, action, reward, next_state, done)

        # B-Cell 업데이트
        if len(buffer) > min_buffer_size:
            batch = buffer.sample(batch_size)
            losses = bcell.update_sac(batch)

        state = next_state
        episode_return += reward

        if done:
            break

    logger.log_metrics({
        'episode': episode,
        'return': episode_return,
        'sharpe': info['sharpe']
    })
```

---

## 학습 모니터링

### 실시간 메트릭 추적

```python
# src/utils/monitoring.py
monitor = StabilityMonitor(config)

# 안정성 체크
monitor.check_q_values(q_values)  # Q값 폭발 감지
monitor.check_entropy(entropy)     # 엔트로피 급락 감지
monitor.check_rewards(rewards)     # 보상 이상치 감지
```

### TensorBoard 연동

```bash
# TensorBoard 실행
tensorboard --logdir logs/

# 브라우저에서 확인
http://localhost:6006
```

### 주요 모니터링 지표

| 지표 | 정상 범위 | 경고 임계값 |
|-----|----------|------------|
| Q-value | -10 ~ 10 | > 100 |
| Entropy | > 0.1 | < 0.01 |
| Actor Loss | < 10 | > 50 |
| Critic Loss | < 5 | > 20 |
| Sharpe Ratio | > 0 | < -0.5 |

---

## 하이퍼파라미터 튜닝

### 자동 튜닝 (Optuna)

```bash
python src/core/tuning.py \
    --config configs/default.yaml \
    --n-trials 100 \
    --objective sharpe
```

### 중요도별 파라미터

#### 🔴 매우 중요
- `alpha_init`: SAC 엔트로피 계수 (0.75 권장)
- `cql_alpha`: CQL 정규화 강도 (5.0-10.0)
- `lr`: 학습률 (3e-4)
- `batch_size`: 배치 크기 (256-512)

#### 🟡 중요
- `iql_expectile`: IQL 기댓값 (0.7)
- `iql_temperature`: IQL 온도 (3.0)
- `gamma`: 할인율 (0.99)
- `tau`: 타겟 네트워크 업데이트 (0.005)

#### 🟢 덜 중요
- `n_quantiles`: 분위수 개수 (32)
- `hidden_dim`: 은닉층 차원 (256)
- `memory_capacity`: 메모리 용량 (50000)

### 시장 상황별 권장값

#### 변동성 높은 시장
```yaml
bcell:
  alpha_init: 1.0      # 높은 탐험
  cql_alpha: 10.0      # 강한 보수성

objectives:
  lambda_cvar: 10.0    # CVaR 중시
  lambda_turn: 0.5     # 회전율 페널티 증가
```

#### 안정적인 시장
```yaml
bcell:
  alpha_init: 0.5      # 낮은 탐험
  cql_alpha: 5.0       # 중간 보수성

objectives:
  lambda_cvar: 5.0     # CVaR 완화
  lambda_turn: 0.1     # 회전율 허용
```

---

## 체크포인팅

### SafeTensors 형식

```python
# 모델 저장
bcell.save("checkpoint.safetensors")

# 메타데이터 포함 저장
save_checkpoint({
    'model': bcell.state_dict(),
    'optimizer': optimizer.state_dict(),
    'episode': episode,
    'metrics': metrics
}, "checkpoint_full.pt")
```

### 체크포인트 전략

```yaml
train:
  checkpoint_interval: 100    # 에피소드마다
  save_best: true            # 최고 성능 저장
  save_latest: true          # 최신 저장
  max_checkpoints: 5         # 최대 보관 수
```

### 재개 방법

```bash
# 특정 체크포인트에서 재개
python main.py --mode train \
    --resume logs/20250122_120000/models/checkpoint_best.pt
```

---

## 트러블슈팅

### 문제 1: 학습 불안정

**증상**: 손실이 발산하거나 진동

**해결책**:
```yaml
# 학습률 감소
bcell:
  actor_lr: 1e-4  # 3e-4 → 1e-4
  critic_lr: 1e-4

# CQL 강화
  cql_alpha: 10.0  # 5.0 → 10.0

# 배치 크기 증가
train:
  online_batch_size: 512  # 256 → 512
```

### 문제 2: 과적합

**증상**: 학습 성능은 좋으나 평가 성능 나쁨

**해결책**:
```python
# 드롭아웃 추가
class PolicyNetwork(nn.Module):
    def __init__(self, dropout=0.1):
        self.dropout = nn.Dropout(dropout)

# 데이터 증강
augmented_state = state + torch.randn_like(state) * 0.01
```

### 문제 3: 메모리 부족

**증상**: CUDA out of memory

**해결책**:
```bash
# 배치 크기 감소
python main.py --batch-size 64

# 그래디언트 체크포인팅
python main.py --gradient-checkpointing

# Mixed Precision Training
python main.py --mixed-precision
```

---

## 성능 최적화

### GPU 가속

```python
# Multi-GPU 학습
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Mixed Precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(batch)
scaler.scale(loss).backward()
```

### 데이터 로딩 최적화

```python
# 비동기 데이터 로딩
dataloader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=4,      # 멀티프로세싱
    pin_memory=True,    # GPU 전송 가속
    prefetch_factor=2   # 프리페치
)
```

### 프로파일링

```bash
# PyTorch 프로파일러
python -m torch.utils.bottleneck main.py --mode train

# 메모리 프로파일링
python -m memory_profiler main.py --mode train
```

---

## 실전 팁

### 1. 단계별 접근
```bash
# Step 1: 작은 데이터로 테스트
python main.py --mode demo

# Step 2: 오프라인 학습만 (IQL/TD3BC)
python scripts/train.py --mode offline

# Step 3: 전체 파이프라인
python scripts/train.py --mode full
```

### 2. 로그 분석
```python
# 메트릭 시각화
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json('logs/*/metrics.jsonl', lines=True)
df[['sharpe', 'cvar']].plot()
plt.show()
```

### 3. 앙상블 학습
```python
# 여러 모델 학습 후 앙상블
models = [train_model(seed=i) for i in range(5)]
ensemble_action = np.mean([m.act(state) for m in models], axis=0)
```

---

## 다음 단계

학습이 완료되면:
1. [EVALUATION.md](EVALUATION.md) - 모델 평가 방법
2. [XAI.md](XAI.md) - 의사결정 설명 방법
3. [CONFIGURATION.md](CONFIGURATION.md) - 고급 설정 옵션

---

*Last Updated: 2025-01-22*