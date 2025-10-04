# IRT (Immune Replicator Transport) 설명서

## 📋 목차

1. [개요](#개요)
2. [IRT의 3가지 핵심 메커니즘](#irt의-3가지-핵심-메커니즘)
3. [효과 및 성능 목표](#효과-및-성능-목표)
4. [다른 알고리즘과의 비교](#다른-알고리즘과의-비교)
5. [하이퍼파라미터 가이드](#하이퍼파라미터-가이드)
6. [사용 예시](#사용-예시)
7. [참고 문헌](#참고-문헌)

---

## 개요

### IRT란 무엇인가?

IRT (Immune Replicator Transport)는 **면역학적 메커니즘에서 영감을 받은** 포트폴리오 관리 알고리즘이다. 시장 위기 상황에서 적응력을 높이기 위해 설계되었으며, 두 가지 서로 다른 전략을 혼합한다:

1. **과거 성공 경험 활용** (Replicator Dynamics)
2. **현재 상황에 맞는 전략 탐색** (Optimal Transport)

### 핵심 공식

```
w_t = (1-α)·Replicator(w_{t-1}, f_t) + α·Transport(E_t, K, C_t)
```

**해석**:
- `w_t`: 현재 시점의 프로토타입 혼합 가중치
- `α`: OT-Replicator 혼합 비율 (0~1 사이, 기본 0.3)
- `Replicator(w_{t-1}, f_t)`: 과거 성공 전략에 기반한 가중치
- `Transport(E_t, K, C_t)`: 현재 상태에 최적화된 가중치

### 왜 위기 적응에 효과적인가?

일반적인 강화학습 알고리즘은 **단일 정책**을 학습한다. 하지만 금융 시장은 **정상 구간**과 **위기 구간**에서 완전히 다른 특성을 보인다:

- **정상 구간**: 낮은 변동성, 예측 가능한 패턴
- **위기 구간**: 높은 변동성, 급격한 변화, 상관관계 붕괴

IRT는 **여러 전문가 전략(프로토타입)**을 학습하고, 현재 상황에 따라 동적으로 혼합한다:

```
위기 감지 → T-Cell이 위기 레벨 출력
         → Replicator 가열 (빠른 적응)
         → OT가 위기 신호와 정렬된 프로토타입 선택
         → 위기 적응형 포트폴리오 구성
```

---

## IRT의 3가지 핵심 메커니즘

### 1. Optimal Transport (OT)

**개념**: 현재 상태(에피토프)와 전문가 전략(프로토타입) 간의 **최적 매칭**을 찾는다.

**수학적 배경**:
- Cuturi (2013)의 엔트로피 정규화 최적수송
- Sinkhorn 알고리즘으로 효율적 계산 (O(1/ε) 수렴)

**동작 방식**:
```
1. 현재 시장 상태 → 6개 에피토프 토큰 (E_t)
2. 8개 프로토타입 전략 (K)
3. 비용 행렬 계산: C_ij = d(E_i, K_j) - 면역학적 조정
4. Sinkhorn으로 최적 수송 계획 P* 계산
5. 프로토타입별 수송 질량 → w_ot
```

**직관**:
- 위기 상황 → 위기 신호와 정렬된 프로토타입의 비용 ↓
- OT가 자동으로 위기 대응 전략 선택

### 2. Replicator Dynamics

**개념**: 과거에 성공한 전략을 **선호**하는 메커니즘 (진화 게임 이론)

**수학적 배경**:
- Hofbauer & Sigmund (1998)의 복제자 동역학
- 균형점은 Nash 균형, 안정점은 ESS

**동작 방식**:
```
1. 이전 가중치 w_{t-1} 기억
2. 각 프로토타입의 적합도 f_j 계산 (Critic Q-value 기반)
3. Advantage: A_j = f_j - 평균 적합도
4. 위기 가열: η(c) = η_0 + η_1·crisis_level
5. 업데이트: w_rep ∝ w_{t-1}·exp(η·A)
```

**직관**:
- 성공한 프로토타입 → 가중치 ↑
- 위기 시 → η ↑ (빠른 적응)
- 시간 메모리 → 일관성 유지

### 3. 면역학적 비용 함수

**개념**: 도메인 지식을 비용 함수에 **내장**하여 더 나은 의사결정 유도

**구성 요소**:

```
C_ij = distance - γ·co_stimulation + λ·tolerance + ρ·checkpoint
```

#### 3.1 Co-stimulation (공자극)
- 위기 신호(danger embedding)와 정렬된 에피토프 선호
- `co_stim = <E_i, d_t>` (내적)
- 위기 시 위험 신호와 유사한 패턴 우선 선택

#### 3.2 Tolerance (내성)
- 과거 실패 패턴(self signatures)과 유사한 에피토프 억제
- `tolerance_penalty = ReLU(κ·max_similarity - ε_tol)`
- 반복적 실수 방지

#### 3.3 Checkpoint (체크포인트)
- 과신하는 프로토타입 억제
- `checkpoint_penalty = ρ·confidence_score`
- 과도한 집중 방지

**효과**:
- 단순 거리 기반 매칭보다 **의미 있는 매칭**
- 위기 대응, 실패 회피, 분산 투자 자동 유도

---

## 효과 및 성능 목표

### 핵심 목표

| 메트릭 | SAC Baseline | IRT 목표 | 개선율 |
|--------|--------------|---------|--------|
| **Sharpe Ratio** | 1.0-1.2 | 1.2-1.4 | **+10-15%** |
| **전체 Max Drawdown** | -30~-35% | -20~-25% | **-20-30%** |
| **위기 구간 MDD** | -40~-45% | -25~-30% | **-30-40%** |

### 위기 구간 집중

IRT의 진가는 **위기 구간**에서 발휘된다:

- **2020년 COVID-19**: MDD -40% → -25% (목표)
- **2022년 Fed 금리 인상**: MDD -35% → -22% (목표)
- **정상 구간**: SAC와 유사 (안정성 유지)

### 해석 가능성

IRT는 **블랙박스가 아니다**. 다음 정보를 제공한다:

1. **IRT 분해**: `w = (1-α)·w_rep + α·w_ot`
   - w_rep: 시간 메모리 기여도
   - w_ot: 구조적 매칭 기여도

2. **T-Cell 위기 감지**:
   - 위기 타입별 점수 (변동성, 유동성, 상관관계, 시스템)
   - 위기 레벨 (0~1)

3. **비용 행렬**:
   - 에피토프-프로토타입 간 면역학적 비용
   - 어떤 전략이 왜 선택되었는지 추적

4. **프로토타입 해석**:
   - 각 프로토타입이 선호하는 자산
   - 위기 vs 정상 구간 활성화 패턴

---

## 다른 알고리즘과의 비교

### 호환성 요약

| 알고리즘 | IRT 적용 | Fitness 계산 | Policy 타입 | 권장도 |
|---------|---------|-------------|------------|-------|
| **SAC** | ✅ 최적 | Q(s,a) | Stochastic | ⭐⭐⭐⭐⭐ |
| **TD3** | ✅ 가능 | Q(s,a) | Deterministic | ⭐⭐⭐⭐ |
| **DDPG** | ✅ 가능 | Q(s,a) | Deterministic | ⭐⭐⭐ |
| **PPO** | ⚠️ 수정 필요 | V(s) 기반 | Stochastic | ⭐⭐ |
| **A2C** | ⚠️ 수정 필요 | V(s) 기반 | Stochastic | ⭐⭐ |

### SAC (현재 사용) ⭐⭐⭐⭐⭐

**장점**:
- ✅ **Q-network 기반** → 프로토타입 fitness 계산 용이
- ✅ **Entropy regularization** → IRT exploration과 시너지
- ✅ **Off-policy** → 샘플 효율성 (과거 경험 재사용)
- ✅ **Stochastic policy** → Dirichlet 정책과 완벽 호환
- ✅ **2 Q-networks (ensemble)** → 안정성

**IRT와의 궁합**:
```python
# SAC의 entropy maximization
max E[Q(s,a)] + α_sac·H(π)

# IRT의 exploration
- Sinkhorn entropy (ε)
- Dirichlet concentration (α_k)

# 결과: 이중 exploration → 강건한 학습
```

**사용 예시**:
```python
from stable_baselines3 import SAC
from finrl.agents.irt import IRTPolicy

model = SAC(
    policy=IRTPolicy,
    env=env,
    policy_kwargs={"alpha": 0.3, "eps": 0.10}
)
```

### TD3 ⭐⭐⭐⭐

**장점**:
- ✅ **Q-network 기반** → fitness 계산 가능
- ✅ **Twin Q-networks** → Overestimation 완화
- ✅ **Off-policy** → 샘플 효율성

**차이점**:
- ⚠️ **Deterministic policy** → IRT의 `deterministic=True` 모드 사용
- ❌ **Entropy regularization 없음** → Exploration 약함

**적용 방법**:
```python
from stable_baselines3 import TD3

model = TD3(
    policy=IRTPolicy,  # 동일한 IRT Policy 사용 가능
    env=env,
    policy_kwargs={"alpha": 0.3}
)
# IRT 내부에서 deterministic 모드로 자동 전환
```

### DDPG ⭐⭐⭐

**장점**:
- ✅ **Q-network 기반** → fitness 계산 가능
- ✅ **Off-policy** → 샘플 효율성
- ✅ **단순 구조** → 빠른 학습

**단점**:
- ❌ **Single Q-network** → Overestimation 문제
- ❌ **불안정성** → 학습 발산 가능
- ⚠️ **Deterministic policy**

**권장사항**: TD3 사용 (DDPG 개선 버전)

### PPO ⭐⭐

**문제점**:
- ❌ **V(s) 기반 Critic** → Q(s,a) 없음
- ❌ **On-policy** → 과거 경험 재사용 불가
- ❌ **IRT의 시간 메모리 약화**

**대안** (구조 수정 필요):
```python
# Fitness를 Advantage로 근사
fitness[j] ≈ A(s, a_j) = r + γ·V(s') - V(s)

# 문제:
# 1. Episode 끝까지 기다려야 함 (즉시 계산 불가)
# 2. 분산 ↑ (Monte Carlo 추정)
# 3. On-policy → w_prev 메모리 효과 약화
```

**결론**: IRT와 궁합 나쁨. SAC/TD3 권장.

### A2C ⭐⭐

PPO와 동일한 문제 (V(s) 기반, On-policy).

**추가 단점**:
- Synchronous update → 느린 학습
- PPO의 clipping 없음 → 불안정

---

## 하이퍼파라미터 가이드

### 핵심 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| **alpha** | 0.3 | 0.1-0.5 | OT-Replicator 혼합 비율 |
| **eps** | 0.10 | 0.01-0.2 | Sinkhorn 엔트로피 (exploration) |
| **eta_0** | 0.05 | 0.03-0.08 | 기본 학습률 (Replicator) |
| **eta_1** | 0.15 | 0.05-0.20 | 위기 증가량 (Replicator) |
| **m_tokens** | 6 | 4-8 | 에피토프 토큰 수 |
| **M_proto** | 8 | 6-12 | 프로토타입 수 |
| **dirichlet_min** | 0.5 | 0.1-1.0 | Dirichlet concentration 최소값 |
| **dirichlet_max** | 50.0 | 20.0-100.0 | Dirichlet concentration 최대값 |

### 파라미터 효과

#### alpha (OT-Replicator 혼합)

```
α=0.0: Pure Replicator (시간 일관성 ↑, 구조 적응 ↓)
α=0.3: Balanced (권장)
α=1.0: Pure OT (구조 매칭 ↑, 시간 메모리 ↓)
```

**Ablation Study 예시**:
```bash
python scripts/train_irt.py --alpha 0.0 --episodes 200  # Pure Replicator
python scripts/train_irt.py --alpha 0.3 --episodes 200  # Balanced
python scripts/train_irt.py --alpha 1.0 --episodes 200  # Pure OT
```

#### eps (Sinkhorn 엔트로피)

```
ε ↑ → 수송 계획 P가 균등 분산 → exploration ↑
ε ↓ → 수송 계획 P가 집중 → exploitation ↑
```

**권장값**: 0.10 (Cuturi, 2013 권장 범위)

#### eta_1 (위기 가열)

```
η(c) = η_0 + η_1·crisis_level

η_1=0.05: 약한 가열 (안정적, 느린 적응)
η_1=0.15: 중간 가열 (권장)
η_1=0.20: 강한 가열 (빠른 적응, 불안정 가능)
```

**⚠️ 주의**: η_1 > 0.20은 학습 불안정 가능

#### dirichlet_min/max (Exploration)

```
min ↓ → 더 sparse한 포트폴리오 가능 (높은 exploration)
max ↑ → 더 집중된 포트폴리오 가능 (낮은 exploration)
```

**핸드오버 권장**: min=0.5, max=50.0 (무거래 루프 방지)

### 무거래 루프 방지

**문제**: Episode 전체에서 turnover ≈ 0, 균등 분배 정책 반복

**해결** (핸드오버 문서 기반):
```yaml
# 환경 레벨
lambda_turn: 0.01  # 0.1 → 0.01 (거래 유인)

# IRT 레벨
eps: 0.10          # 0.05 → 0.10 (OT 다양성)
eta_1: 0.15        # 0.10 → 0.15 (빠른 적응)
dirichlet_min: 0.5 # 1.0 → 0.5 (exploration)
```

---

## 사용 예시

### 기본 사용법

```python
from stable_baselines3 import SAC
from finrl.agents.irt import IRTPolicy

# IRT Policy 설정
policy_kwargs = {
    "emb_dim": 128,
    "m_tokens": 6,
    "M_proto": 8,
    "alpha": 0.3,
    "eps": 0.10,
    "eta_0": 0.05,
    "eta_1": 0.15
}

# SAC + IRT
model = SAC(
    policy=IRTPolicy,
    env=env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    verbose=1
)

# 학습
model.learn(total_timesteps=50000)

# 저장
model.save("irt_model.zip")
```

### CLI 사용법

```bash
# 기본 설정으로 학습 및 평가
python scripts/train_irt.py --mode both --episodes 200

# Alpha 조정 (Ablation Study)
python scripts/train_irt.py --alpha 0.5 --episodes 200

# 저장된 모델 평가
python scripts/evaluate.py \
  --model logs/irt/20251004_*/irt_final.zip \
  --method direct \
  --save-plot \
  --save-json
```

### 해석 가능성 활용

```python
# 평가 시 IRT 정보 수집
obs = env.reset()
action, _ = model.predict(obs, deterministic=True)

# IRT Policy에서 정보 추출 (내부 접근)
policy = model.policy
actor = policy.irt_actor

# Forward pass로 IRT 정보 획득
action, info = actor(obs_tensor, deterministic=True)

print(f"프로토타입 가중치: {info['w']}")
print(f"위기 레벨: {info['crisis_level']}")
print(f"IRT 분해 - Replicator: {info['w_rep']}")
print(f"IRT 분해 - OT: {info['w_ot']}")
```

---

## 참고 문헌

### 이론적 기초

1. **Optimal Transport**
   - Cuturi, M. (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
   - NIPS 2013

2. **Replicator Dynamics**
   - Hofbauer, J., & Sigmund, K. (1998). "Evolutionary Games and Population Dynamics"
   - Cambridge University Press

3. **면역학적 비용**
   - 프로젝트 독자 설계 (도메인 지식 기반)

### 구현 참조

4. **FinRL**
   - Liu, X. Y., et al. (2024). "FinRL: Financial Reinforcement Learning Framework"
   - NeurIPS Workshop

5. **Stable Baselines3**
   - Raffin, A., et al. (2021). "Stable-Baselines3: Reliable Reinforcement Learning Implementations"
   - JMLR

### 관련 연구

6. **정보기하학**
   - Amari, S. (2016). "Information Geometry and Its Applications"
   - Applied Mathematical Sciences

7. **Portfolio Optimization**
   - Markowitz, H. (1952). "Portfolio Selection"
   - Journal of Finance

---

## 추가 자료

- **프로젝트 문서**: [README.md](../README.md)
- **변경사항 이력**: [CHANGELOG.md](CHANGELOG.md)
- **스크립트 가이드**: [SCRIPTS.md](SCRIPTS.md)
- **FinRL 공식 문서**: [https://finrl.readthedocs.io/](https://finrl.readthedocs.io/)
- **Stable Baselines3 문서**: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)

---

**문의**: GitHub Issues 또는 Discussions 활용
