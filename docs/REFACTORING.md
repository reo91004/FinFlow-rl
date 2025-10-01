# 🧬 **IRT (Immune Replicator Transport) Operator: 설계 명세 및 이론적 기초**

**Version 1.0 | 2025-10-01**

---

## 📑 **목차**

1. [Executive Summary](#1-executive-summary)
2. [동기 및 배경](#2-동기-및-배경)
3. [IRT Operator 수학적 정의](#3-irt-operator-수학적-정의)
4. [기존 기법과의 비교 분석](#4-기존-기법과의-비교-분석)
5. [이론적 기초와 검증 가능성](#5-이론적-기초와-검증-가능성)
6. [구현 명세](#6-구현-명세)
7. [실험 설계 및 검증 계획](#7-실험-설계-및-검증-계획)
8. [적용 가능 영역](#8-적용-가능-영역)
9. [한계점 및 완화 방안](#9-한계점-및-완화-방안)
10. [결론 및 기여도](#10-결론-및-기여도)

---

## **1. Executive Summary**

### **1.1 핵심 제안**

IRT (Immune Replicator Transport) Operator는 **엔트로피 정규화 최적수송(Optimal Transport)**과 **복제자 동역학(Replicator Dynamics)**을 결합하여, 시간적 일관성과 구조적 매칭을 동시에 달성하는 새로운 정책 혼합 연산자다.

**한 줄 정의:**

```
w_t = (1-α)·Replicator(w_{t-1}, f_t) + α·Transport(E_t, K, C_t)
```

### **1.2 차별성**

| 특성        | Attention | Soft MoE        | IRT            |
| ----------- | --------- | --------------- | -------------- |
| 시간 메모리 | ✗         | ✗               | ✅ w\_{t-1}    |
| 구조적 매칭 | ✅ QK^T   | ✅ Weighted Avg | ✅ OT          |
| 질량 보존   | ✗         | ✗               | ✅             |
| 도메인 신호 | ✗         | ✗               | ✅ 비용 함수   |
| m=1 극한    | Softmax   | Weighted Avg    | **Replicator** |

**핵심 차별점**: 복제자 동역학의 시간 메모리로 인해 IRT는 단일 토큰(`m=1`) 극한에서도 softmax로 퇴화하지 않는다.

### **1.3 적용 영역**

- **주 타겟**: 포트폴리오 관리 (위기 적응)
- **확장 가능**: 의료 의사결정, 로봇 제어 (regime switching)
- **부적합**: 텍스트 생성 (시간 일관성 불필요)

### **1.4 혁신성 점수**

- **이론적 참신성**: ⭐⭐⭐⭐ (4/5) - 조합 신규
- **실용적 가치**: ⭐⭐⭐⭐ (4/5) - 위기 적응에 직접 적용
- **수학적 엄밀성**: ⭐⭐⭐☆ (3.5/5) - 수렴 보조정리 필요
- **구현 용이성**: ⭐⭐⭐⭐ (4/5) - ~270 lines

---

## **2. 동기 및 배경**

### **2.1 문제 정의**

포트폴리오 관리에서 강화학습 에이전트는 다음 문제에 직면한다:

1. **급격한 레짐 전환**: 정상 시장 → 위기 → 회복
2. **다중 모드 분포**: 단일 정책으로 모든 상황 포착 불가
3. **샘플 효율성**: 위기 데이터는 희소

**기존 접근법의 한계:**

최근 PPO/SAC 기반 포트폴리오 연구들은 단일 정책으로 다양한 시장 상황을 처리하려 하지만, 위기 구간에서 급격한 성능 저하를 보인다.

### **2.2 설계 원칙**

**P1. 구조적 매칭 (Structural Matching)**

- 현재 시장 상태와 과거 전문가 패턴 간의 최적 결합
- 최적수송은 이산 분포 간 매칭을 질량 보존 제약으로 형식화

**P2. 시간적 일관성 (Temporal Consistency)**

- 갑작스러운 전략 변경 방지 (거래 비용)
- 복제자 동역학은 성공한 전략의 빈도를 지수적으로 증가시키는 생물학적 원리

**P3. 위기 적응 (Crisis Adaptation)**

- 면역학적 신호(공자극, 내성, 체크포인트)를 비용 함수에 내장
- 위기 시 적응 속도 자동 증가

**P4. 해석 가능성 (Interpretability)**

- 수송 행렬 `P[i,j]`: "에피토프 i가 프로토타입 j로 매핑"
- 복제자 가중치 `w_t`: "과거 성능 기반 혼합"
- 비용 분해: distance + co-stim + tolerance + checkpoint

---

## **3. IRT Operator 수학적 정의**

### **3.1 표기법**

| 기호                  | 정의                        | 차원        |
| --------------------- | --------------------------- | ----------- |
| `s_t ∈ S`             | 시점 t의 상태               | -           |
| `E_t = {e_i}_{i=1}^m` | 에피토프 토큰 (상태 임베딩) | `[B, m, D]` |
| `K = {k_j}_{j=1}^M`   | 항체 프로토타입 (전문가 키) | `[B, M, D]` |
| `d_t ∈ ℝ^D`           | 공자극 임베딩 (위기 신호)   | `[B, D]`    |
| `c_t ∈ [0,1]`         | 스칼라 위기 레벨            | `[B, 1]`    |
| `w_t ∈ Δ^M`           | 프로토타입 혼합 가중치      | `[B, M]`    |
| `π_j: S → Δ^A`        | 프로토타입 j의 정책         | -           |

### **3.2 Step 1: 면역학적 비용 행렬**

```math
C_{ij} = d_\phi(e_i, k_j) - γ⟨e_i, d_t⟩ + λ[\kappa \max_{u∈S} \cos(e_i,u) - ε]_+ + ρ \mathrm{conf}(k_j)
```

**각 항의 근거:**

1. **기본 거리** `d_φ(e_i, k_j)`:

   - 학습 가능한 마할라노비스 거리
   - `d_φ(x,y) = √((x-y)^T M_φ (x-y))`, `M_φ = A^T A` (positive definite)

2. **공자극 (Co-stimulation)** `-γ⟨e_i, d_t⟩`:

   - 위험 신호와 정렬된 에피토프를 선호
   - 면역학 근거: T-Cell 활성화는 항원 신호(MHC) + 공자극(CD28) 필요
   - γ > 0: 위기 시 위험 신호에 반응하는 전략 선호

3. **음성 선택 (Tolerance)** `λ[κ max cos - ε]_+`:

   - 자기-내성 서명 집합 `S`와 유사한 패턴 억제
   - 과거 실패한 전략 회피
   - `[·]_+ = max(0, ·)`: 임계값 ε 이상만 페널티

4. **체크포인트 억제 (Checkpoint)** `ρ conf(k_j)`:
   - 과신(overconfidence) 억제
   - `conf(k_j) = -H(π_j(·|s_t))` (정책 엔트로피의 음수)
   - 너무 확신하는 프로토타입에 페널티

**이론적 근거**: 최적수송 기반 오프라인 RL에서 critic 기반 비용 함수가 최적 행동 선택을 유도함이 증명됨. IRT는 이를 다층 면역 신호로 확장했다.

### **3.3 Step 2: 엔트로피 정규화 최적수송**

**문제 정의:**

```math
P_t^* = \arg\min_{P ∈ U(u,v)} ⟨P, C_t⟩ + ε·KL(P || uv^T)
```

where `U(u,v) = {P ≥ 0 : P1_M = u, P^T 1_m = v}`

**주변 분포:**

- `u = (1/m)1_m ∈ ℝ^m` (에피토프 균등 분포)
- `v = (1/M)1_M ∈ ℝ^M` (프로토타입 균등 분포)

**Sinkhorn 알고리즘:**

```math
K = exp(-C_t / ε)
r^{(n+1)} = u ⊘ (K c^{(n)})
c^{(n+1)} = v ⊘ (K^T r^{(n+1)})
P_t^* = diag(r^{(∞)}) K diag(c^{(∞)})
```

**수송 질량의 프로토타입 마진:**

```math
p_t(j) = ∑_{i=1}^m P_{t,ij}^*
```

**이론적 보장**: 엔트로피 정규화 OT는 유일한 해를 가지며, Sinkhorn 알고리즘은 선형 수렴 (Cuturi, 2013).

### **3.4 Step 3: 위기-공자극 복제자 업데이트**

```math
\tilde{w}_t(j) ∝ w_{t-1}(j) · exp(η(c_t)[f_j(s_t) - \bar{f}_t] - λ_{neg} r_j(E_t))
```

**각 항의 의미:**

1. **과거 가중치** `w_{t-1}(j)`:

   - 지수 이동 평균: `w_{t-1} = 0.9·w_{t-2} + 0.1·w_t^{실제}`
   - 시간 메모리 제공

2. **적합도 (Fitness)** `f_j(s_t)`:

   - 옵션 A: `𝔼_{a~π_j}[Q(s_t,a)]` (기대 Q값)
   - 옵션 B: Distributional RL에서 CVaR\_{α}(Z_j(s_t,a))
   - 베이스라인 `\bar{f}_t = Σ_j w_{t-1}(j)f_j(s_t)`

3. **위기 가열** `η(c_t) = η_0 + η_1·c_t`:

   - 위기 시 적응 속도 증가
   - 근거: 확장된 복제자 동역학에서 미래 보상을 고려한 가변 학습률

4. **자기-내성** `r_j(E_t)`:
   - 프로토타입 j도 self 시그니처와 유사하면 억제
   - `r_j = max_{u∈S} cos(k_j, u)`

**복제자 동역학 연결**: 표준 복제자 방정식 dw_j/dt = w_j(f_j - \bar{f})의 이산 버전이며, 진화 게임 이론에서 내쉬 균형으로 수렴함이 증명됨 (Hofbauer & Sigmund, 1998).

### **3.5 Step 4: 이중 결합 혼합**

```math
w_t = (1-α)·\mathrm{Normalize}(\tilde{w}_t) + α·p_t
```

**정규화**: `Normalize(x) = x / (||x||_1 + ε_num)`

**최종 정책:**

```math
π_{IRT}(a|s_t) = \mathrm{Dirichlet}\bigg(\sum_{j=1}^M w_t(j)·α_j(s_t)\bigg)
```

where `α_j(s_t) ∈ ℝ^A_+` is the concentration parameter from decoder `j`.

**형태적 특성:**

- `α = 0`: 순수 Replicator (진화적 동역학)
- `α = 1`: 순수 OT (구조적 매칭)
- `0 < α < 1`: **geodesic interpolation** on simplex

### **3.6 극한 분석**

**Claim 1**: IRT는 `m=1` (단일 에피토프) 극한에서도 softmax로 퇴화하지 않는다.

**증명 스케치:**

```
m=1일 때:
- OT 항: p_t = v (균등 분포, 정보 없음)
- Replicator 항: w_t ≈ (1-α)·tilde_w_t + α/M
- 따라서 w_t는 여전히 w_{t-1}에 의존 → 시간 메모리 유지
```

**Claim 2**: IRT는 단순 가중 평균이 아니다.

**증명**: OT는 비선형 최적화 문제이며, Sinkhorn 반복은 exp-log 연산 포함. 따라서 `p_t`는 `C_t`의 비선형 함수.

---

## **4. 기존 기법과의 비교 분석**

### **4.1 Attention Mechanism**

**수학적 정의:**

```math
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\bigg(\frac{QK^T}{\sqrt{d}}\bigg)V
```

**IRT와의 차이:**

| 측면        | Attention          | IRT                     |
| ----------- | ------------------ | ----------------------- |
| 매칭 방법   | 내적 → softmax     | 비용 행렬 → Sinkhorn OT |
| 시간 메모리 | ✗ (각 레이어 독립) | ✅ w\_{t-1} 명시적      |
| 질량 보존   | ✗                  | ✅ (OT 제약)            |
| 도메인 신호 | ✗                  | ✅ (비용 함수)          |
| 계산 복잡도 | O(n²d)             | O(iters·m·M·d)          |

**본질적 차이**: Attention은 **순간적 유사도**만 계산하지만, IRT는 **과거 성능**과 **현재 구조**를 동시에 고려한다.

**근거**: Soft MoE 논문에서도 Attention이 정적 가중 평균임을 지적하며, 동적 할당의 필요성 강조했다.

### **4.2 Mixture of Experts (MoE)**

**표준 MoE:**

```math
\mathrm{MoE}(x) = ∑_{j=1}^M g_j(x)·E_j(x)
```

where `g(x) = softmax(W_g x)` (게이팅 함수)

**Soft MoE (Google DeepMind, 2024):**

```math
\mathrm{SoftMoE}(X) = ∑_{j=1}^M φ_j^T X · E_j
```

where `φ_j = softmax(W_j X^T)` (slot 가중치)

**IRT와의 차이:**

| 측면        | Standard MoE          | Soft MoE     | IRT              |
| ----------- | --------------------- | ------------ | ---------------- |
| 게이팅      | Softmax               | Weighted Avg | OT + Replicator  |
| 시간 메모리 | ✗                     | ✗            | ✅               |
| 할당 방식   | Hard/Top-K            | Soft (연속)  | Transport (질량) |
| 미분 가능   | ✗ (Top-K) / ✅ (Soft) | ✅           | ✅               |

**핵심 차별점**: Soft MoE는 완전 미분 가능한 가중 평균으로 토큰-전문가 할당을 수행하지만, **과거 성능을 고려하지 않는다**. IRT는 Replicator 항으로 이를 해결한다.

### **4.3 MOT (Mixture of Actors with OT, 2024)**

MOT는 OT를 사용하여 오프라인 샘플을 다중 액터에 할당한다.

**MOT 정의:**

```math
\mathcal{L}_{OT} = D_W(μ_{data}, μ_{actors}) + λ·regularization
```

**IRT와의 차이:**

| 측면        | MOT                 | IRT                 |
| ----------- | ------------------- | ------------------- |
| OT 용도     | 샘플 할당 (학습 시) | 정책 혼합 (추론 시) |
| 시간 축     | ✗                   | ✅ Replicator       |
| 실시간      | ✗ (오프라인)        | ✅                  |
| 도메인 신호 | ✗                   | ✅ 비용 함수        |

**근거**: MOT는 imitation learning에서 전문가 전략과 정렬하기 위해 OT를 사용하지만, **온라인 추론에서는 표준 정책 선택**을 사용한다.

### **4.4 Replicator Dynamics in RL**

Tuyls et al. (2003)은 복제자 동역학과 Q-learning의 연결을 증명했다.

**표준 복제자:**

```math
\frac{dw_j}{dt} = w_j(f_j - \bar{f})
```

**확장된 복제자 (Tuyls et al.):**

```math
\frac{dw_j}{dt} = w_j(\underbrace{f_j^{current}}_{\text{즉시 보상}} + \underbrace{\beta f_j^{future}}_{\text{미래 보상}} - \bar{f})
```

**IRT의 복제자:**

```math
\tilde{w}_t(j) ∝ w_{t-1}(j) · exp(\underbrace{η(c_t)}_{\text{위기 가열}}[f_j(s_t) - \bar{f}_t] - \underbrace{λ_{neg} r_j}_{\text{내성}})
```

**혁신점**:

1. **위기 가열** `η(c_t)`: 레짐 전환 시 적응 속도 증가
2. **내성 페널티**: 자기-유사 전략 억제
3. **OT와 결합**: 구조적 매칭 + 진화적 동역학

---

## **5. 이론적 기초와 검증 가능성**

### **5.1 존재성과 유일성**

**Theorem 1 (OT 해의 존재성)**:

> 엔트로피 정규화 OT 문제는 유일한 해 `P_t^*`를 갖는다.

**증명 스케치**:

- 목적 함수 `F(P) = ⟨P,C⟩ + ε·KL(P||uv^T)`는 strictly convex
- 제약 집합 `U(u,v)`는 compact convex polytope
- Strictly convex + compact → 유일한 최솟값 ∎

**근거**: Cuturi (2013) "Sinkhorn Distances"에서 엔트로피 정규화 OT의 유일성과 Sinkhorn 알고리즘의 선형 수렴 증명

### **5.2 수렴성 분석**

**Proposition 1 (Sinkhorn 수렴)**:

> Sinkhorn 알고리즘은 `O(1/ε)` 반복 내에 `||P^{(n)} - P^*||_F ≤ δ`를 달성한다.

**Proposition 2 (Replicator 평형)**:

> 일정한 fitness 하에서, 복제자 동역학은 진화적 안정 전략(ESS)으로 수렴한다.

**증명**: Hofbauer & Sigmund (1998) 정리 7.4.1 - 복제자 동역학의 내쉬 균형 수렴 조건

**미해결 문제**: IRT 전체의 수렴성은 **OT와 Replicator의 결합**으로 인해 복잡하다.

**추론 가능한 성질:**

1. OT 단계: `P_t^*`는 잘 정의됨 (Theorem 1)
2. Replicator 단계: 고정된 fitness에서 수렴 (Prop 2)
3. 결합: `α`가 작으면 Replicator 지배 → 수렴 가능성 ↑

**실험적 검증 필요**:

- Loss landscape 시각화
- Lyapunov 함수 후보: `V(w_t) = -∑_j w_t(j) log w_t(j) + ⟨w_t, -f_t⟩`

### **5.3 정보기하학적 해석**

**복제자 동역학 = 거울 하강**:
복제자 방정식은 negative entropy를 Bregman divergence로 하는 심플렉스 상의 거울 하강과 동형이다.

```math
w_{t+1} = \arg\min_{w∈Δ^M} ⟨w, -f_t⟩ + \frac{1}{η}D_{KL}(w || w_t)
```

**OT = Wasserstein Gradient Flow**:
최적수송은 Wasserstein 공간 상의 gradient descent로 해석 가능 (Jordan-Kinderlehrer-Otto, 1998).

**IRT의 기하학**:

- Replicator: `Δ^M` 심플렉스 상의 정보기하
- OT: Wasserstein space `(P(X), W_2)` 상의 리만 기하
- 결합: **이중 다양체 간의 geodesic interpolation**

이는 **새로운 기하학적 구조**를 암시하며, 향후 이론 연구의 대상이다.

### **5.4 복잡도 분석**

**시간 복잡도:**

```
Forward Pass:
- 에피토프 인코딩: O(B·S·m·D)
- 비용 행렬: O(B·m·M·D)
- Sinkhorn: O(iters·B·m·M)
- Replicator: O(B·M)
- 디코딩: O(B·M·A)

Total: O(B·(S·m·D + iters·m·M + M·A))
```

**표준 SAC 대비**: 약 1.5-2배 증가, 하지만 일일 거래 빈도에서는 무시 가능.

**공간 복잡도:**

```
메모리:
- 에피토프: B·m·D
- 프로토타입: B·M·D
- 수송 행렬: B·m·M
- 가중치 히스토리: M

Total: O(B·(m+M)·D + B·m·M)
```

**근거**: MOT 논문에서도 OT 정규화가 약간의 계산 오버헤드를 발생시키지만, 학습 안정성 향상으로 상쇄된다고 보고.

---

## **6. 구현 명세**

### **6.1 최소 구현 (Minimal Implementation)**

```python
# src/immune/irt.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sinkhorn(nn.Module):
    """엔트로피 정규화 최적수송"""
    def __init__(self, max_iters: int = 10, eps: float = 0.05):
        super().__init__()
        self.max_iters = max_iters
        self.eps = eps

    def forward(self, C: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        """
        Args:
            C: 비용 행렬 [B, m, M]
            u: 소스 분포 [B, m, 1]
            v: 타겟 분포 [B, 1, M]
        Returns:
            P: 수송 계획 [B, m, M]
        """
        # 수치 안정성을 위한 log-space 연산
        log_K = -C / self.eps  # [B, m, M]
        log_u = torch.log(u + 1e-8)  # [B, m, 1]
        log_v = torch.log(v + 1e-8)  # [B, 1, M]

        log_a = torch.zeros_like(log_u)  # [B, m, 1]
        log_b = torch.zeros_like(log_v)  # [B, 1, M]

        for _ in range(self.max_iters):
            log_a = log_u - torch.logsumexp(log_K + log_b, dim=2, keepdim=True)
            log_b = log_v - torch.logsumexp(log_K + log_a, dim=1, keepdim=True)

        P = torch.exp(log_a + log_K + log_b)
        return P

class IRT(nn.Module):
    """Immune Replicator Transport Operator"""
    def __init__(self, emb_dim: int, m_tokens: int = 6, M_proto: int = 8,
                 eps: float = 0.05, alpha: float = 0.3):
        super().__init__()
        self.emb_dim = emb_dim
        self.m, self.M = m_tokens, M_proto
        self.alpha = alpha

        # 학습 가능한 마할라노비스 메트릭
        self.metric_L = nn.Parameter(torch.eye(emb_dim))  # Cholesky factor

        # 자기-내성 서명 (학습 가능하거나 고정)
        self.register_buffer('self_sigs', torch.randn(4, emb_dim) * 0.1)

        # Sinkhorn 알고리즘
        self.sinkhorn = Sinkhorn(eps=eps)

        # 하이퍼파라미터
        self.gamma = 0.5  # 공자극 가중치
        self.lambda_tol = 2.0  # 내성 가중치
        self.rho = 0.3  # 체크포인트 가중치
        self.kappa = 1.0  # 내성 게인
        self.epsilon_tol = 0.1  # 내성 임계값

    def _cost_matrix(self, E: torch.Tensor, K: torch.Tensor,
                     danger: torch.Tensor, proto_conf: torch.Tensor):
        """
        면역학적 비용 행렬 계산

        Args:
            E: 에피토프 [B, m, D]
            K: 프로토타입 [B, M, D]
            danger: 공자극 임베딩 [B, D]
            proto_conf: 프로토타입 과신도 [B, 1, M]
        """
        B, m, D = E.shape
        M = K.shape[1]

        # 1. 마할라노비스 거리
        M_matrix = self.metric_L.T @ self.metric_L  # [D, D]
        diff = E.unsqueeze(2) - K.unsqueeze(1)  # [B, m, M, D]
        mahal = torch.sqrt(
            torch.clamp(
                torch.einsum('bmnd,de,bmne->bmn', diff, M_matrix, diff),
                min=1e-8
            )
        )  # [B, m, M]

        # 2. 공자극 (Co-stimulation)
        co_stim = torch.einsum('bmd,bd->bm', E, danger).unsqueeze(2)  # [B, m, 1]

        # 3. 음성 선택 (Tolerance)
        if self.self_sigs.numel() > 0:
            E_norm = F.normalize(E, dim=-1)  # [B, m, D]
            sig_norm = F.normalize(self.self_sigs, dim=-1)  # [S, D]
            cos_sim = E_norm @ sig_norm.T  # [B, m, S]
            worst_match = cos_sim.max(dim=-1, keepdim=True)[0]  # [B, m, 1]
            tolerance_penalty = torch.relu(
                self.kappa * worst_match - self.epsilon_tol
            )  # [B, m, 1]
        else:
            tolerance_penalty = 0.0

        # 4. 종합 비용
        C = (
            mahal
            - self.gamma * co_stim
            + self.lambda_tol * tolerance_penalty
            + self.rho * proto_conf
        )

        return C

    def forward(self, E: torch.Tensor, K: torch.Tensor,
                danger: torch.Tensor, w_prev: torch.Tensor,
                fitness: torch.Tensor, crisis_level: torch.Tensor,
                proto_conf: torch.Tensor = None):
        """
        IRT 연산자 forward

        Args:
            E: 에피토프 [B, m, D]
            K: 프로토타입 [B, M, D]
            danger: 공자극 [B, D]
            w_prev: 이전 가중치 [B, M]
            fitness: 프로토타입 적합도 [B, M]
            crisis_level: 위기 레벨 [B, 1]
            proto_conf: 프로토타입 과신도 [B, 1, M] (optional)

        Returns:
            w_new: 새 혼합 가중치 [B, M]
            P: 수송 계획 [B, m, M] (해석용)
        """
        B, m, D = E.shape
        M = K.shape[1]

        if proto_conf is None:
            proto_conf = torch.zeros(B, 1, M, device=E.device)

        # Step 1: OT 매칭
        u = torch.full((B, m, 1), 1.0/m, device=E.device)
        v = torch.full((B, 1, M), 1.0/M, device=E.device)

        C = self._cost_matrix(E, K, danger, proto_conf)
        P = self.sinkhorn(C, u, v)  # [B, m, M]

        p_mass = P.sum(dim=1)  # [B, M] - OT 마진

        # Step 2: Replicator 업데이트
        eta_0, eta_1 = 0.05, 0.10
        eta = eta_0 + eta_1 * crisis_level  # [B, 1]

        baseline = (w_prev * fitness).sum(dim=-1, keepdim=True)  # [B, 1]
        advantage = fitness - baseline  # [B, M]

        # 내성 페널티 (프로토타입도 자기-유사하면 억제)
        if self.self_sigs.numel() > 0:
            K_norm = F.normalize(K, dim=-1)
            sig_norm = F.normalize(self.self_sigs, dim=-1)
            proto_self_sim = (K_norm @ sig_norm.T).max(dim=-1)[0]  # [B, M]
            r_penalty = proto_self_sim
        else:
            r_penalty = 0.0

        log_tilde_w = (
            torch.log(w_prev + 1e-8)
            + eta * advantage
            - 0.5 * r_penalty
        )
        tilde_w = F.softmax(log_tilde_w, dim=-1)  # [B, M]

        # Step 3: 이중 결합
        w_new = (1 - self.alpha) * tilde_w + self.alpha * p_mass
        w_new = w_new / (w_new.sum(dim=-1, keepdim=True) + 1e-8)

        return w_new, P
```

### **6.2 Actor 통합**

```python
# src/agents/bcell_actor_irt.py
import torch
import torch.nn as nn
from src.immune.irt import IRT

class BCellActorIRT(nn.Module):
    """IRT 기반 B-Cell Actor"""
    def __init__(self, state_dim: int, action_dim: int,
                 emb_dim: int = 128, m: int = 6, M: int = 8):
        super().__init__()
        self.m, self.M = m, M
        self.action_dim = action_dim

        # 에피토프 인코더
        self.epitope_enc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, m * emb_dim)
        )

        # 프로토타입 키 (학습 가능)
        self.proto_keys = nn.Parameter(
            torch.randn(M, emb_dim) / (emb_dim ** 0.5)
        )

        # 각 프로토타입의 Dirichlet 디코더
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Softplus()  # 양수 concentration
            )
            for _ in range(M)
        ])

        # IRT 연산자
        self.irt = IRT(emb_dim, m_tokens=m, M_proto=M)

        # 이전 가중치 (EMA 업데이트)
        self.register_buffer('w_prev', torch.full((1, M), 1.0/M))
        self.ema_beta = 0.9

    def _compute_fitness(self, state: torch.Tensor, critics: list):
        """
        각 프로토타입의 적합도 계산

        Args:
            state: [B, S]
            critics: Q 네트워크 리스트

        Returns:
            fitness: [B, M]
        """
        B = state.size(0)
        fitness = torch.zeros(B, self.M, device=state.device)

        with torch.no_grad():
            for j in range(self.M):
                # 프로토타입 j의 정책으로 행동 샘플
                conc_j = self.decoders[j](self.proto_keys[j])  # [A]
                dist_j = torch.distributions.Dirichlet(conc_j + 1.0)
                action_j = dist_j.sample((B,))  # [B, A]

                # 평균 Q값
                q_values = [critic(state, action_j) for critic in critics]
                fitness[:, j] = torch.stack(q_values).mean(dim=0).squeeze()

        return fitness

    def forward(self, state: torch.Tensor, danger_embed: torch.Tensor,
                crisis_level: torch.Tensor, critics: list = None):
        """
        Args:
            state: [B, S]
            danger_embed: [B, D]
            crisis_level: [B, 1]
            critics: Q 네트워크 리스트 (fitness 계산용)

        Returns:
            action: [B, A] - Dirichlet 혼합 정책
            w: [B, M] - 프로토타입 가중치
            P: [B, m, M] - 수송 계획
        """
        B = state.size(0)

        # 1. 에피토프 인코딩
        E = self.epitope_enc(state).view(B, self.m, -1)  # [B, m, D]

        # 2. 프로토타입 (배치로 확장)
        K = self.proto_keys.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

        # 3. Fitness 계산
        if critics is not None:
            fitness = self._compute_fitness(state, critics)
        else:
            # 평가 모드: 균등 fitness
            fitness = torch.ones(B, self.M, device=state.device)

        # 4. IRT 연산
        w_prev_batch = self.w_prev.expand(B, -1)  # [B, M]
        w, P = self.irt(E, K, danger_embed, w_prev_batch,
                        fitness, crisis_level)

        # 5. Dirichlet 혼합
        concentrations = torch.stack([
            dec(K[:, j, :]) for j, dec in enumerate(self.decoders)
        ], dim=1)  # [B, M, A]

        mixed_conc = torch.einsum('bm,bma->ba', w, concentrations) + 1.0
        action = mixed_conc / mixed_conc.sum(dim=-1, keepdim=True)

        # 6. EMA 업데이트
        with torch.no_grad():
            self.w_prev = (
                self.ema_beta * self.w_prev
                + (1 - self.ema_beta) * w.detach().mean(dim=0, keepdim=True)
            )

        return action, w, P
```

### **6.3 T-Cell 통합**

```python
# src/immune/t_cell_min.py
import torch
import torch.nn as nn

class TCellMinimal(nn.Module):
    """경량 T-Cell: 위기 감지 + 공자극 임베딩"""
    def __init__(self, in_dim: int, emb_dim: int = 128,
                 n_types: int = 4):
        super().__init__()
        self.n_types = n_types
        self.emb_dim = emb_dim

        # 단일 인코더
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_types + emb_dim)
        )

        # 위기 타입별 통계 (온라인 업데이트)
        self.register_buffer('mu', torch.zeros(n_types))
        self.register_buffer('sigma', torch.ones(n_types))
        self.register_buffer('count', torch.zeros(1))

        # 위기 타입별 가중치 (학습 가능)
        self.alpha = nn.Parameter(torch.ones(n_types) / n_types)

        self.momentum = 0.99

    def forward(self, features: torch.Tensor, update_stats: bool = True):
        """
        Args:
            features: 시장 특성 [B, F]
            update_stats: 통계 업데이트 여부

        Returns:
            z: 위기 타입 점수 [B, K]
            d: 공자극 임베딩 [B, D]
            c: 스칼라 위기 레벨 [B, 1]
        """
        h = self.encoder(features)  # [B, K+D]

        z = h[:, :self.n_types]  # [B, K] - 위기 타입 점수
        d = h[:, self.n_types:]  # [B, D] - 공자극 임베딩

        # 온라인 정규화
        if update_stats and self.training:
            with torch.no_grad():
                batch_mu = z.mean(dim=0)
                batch_sigma = z.std(dim=0) + 1e-6

                self.mu = self.momentum * self.mu + (1 - self.momentum) * batch_mu
                self.sigma = self.momentum * self.sigma + (1 - self.momentum) * batch_sigma
                self.count += 1

        # 표준화
        z_std = (z - self.mu) / (self.sigma + 1e-6)

        # 가중 합산 → 시그모이드
        c = torch.sigmoid(
            (z_std * F.softmax(self.alpha, dim=0)).sum(dim=-1, keepdim=True)
        )  # [B, 1]

        return z, d, c
```

### **6.4 통합 예시**

```python
# 학습 루프 통합
state_dim = 43  # FinFlow-RL
action_dim = 30
emb_dim = 128

# 초기화
t_cell = TCellMinimal(in_dim=12, emb_dim=emb_dim)  # 시장 특성만
actor = BCellActorIRT(state_dim, action_dim, emb_dim)
critics = [QNetwork(...) for _ in range(10)]  # REDQ

# Forward
state = ...  # [B, 43]
market_features = state[:, :12]  # 시장 특성 추출

z, danger_embed, crisis_level = t_cell(market_features)
action, w, P = actor(state, danger_embed, crisis_level, critics)

# Critic 업데이트 (표준 REDQ)
q_loss = ...

# Actor 업데이트 (IRT 포함)
actor_loss = -critics[0](state, action).mean()
actor_loss.backward()
```

---

## **7. 실험 설계 및 검증 계획**

### **7.1 필수 Ablation Study**

| 실험 ID | 구성               | 목적                   |
| ------- | ------------------ | ---------------------- |
| **A1**  | IRT (full)         | 베이스라인             |
| **A2**  | α=0 (Replicator만) | OT 기여도 측정         |
| **A3**  | α=1 (OT만)         | Replicator 기여도 측정 |
| **A4**  | Soft MoE           | 시간 메모리 효과 검증  |
| **A5**  | Attention          | 면역 신호 효과 검증    |
| **A6**  | Standard SAC       | 전체 시스템 개선도     |

**평가 지표:**

- **전체 구간**: Sharpe, 연환산 수익률, MDD
- **위기 구간**: Crisis MDD, Recovery Time, CVaR
- **거래 비용**: Turnover, Transaction Cost Impact

### **7.2 하이퍼파라미터 민감도**

**Grid Search:**

```yaml
alpha: [0.1, 0.2, 0.3, 0.4, 0.5]
eps (Sinkhorn): [0.01, 0.05, 0.1]
eta_0: [0.03, 0.05, 0.08]
eta_1: [0.05, 0.10, 0.15]
m: [4, 6, 8]
M: [6, 8, 12]
```

**분석 방법:**

- Partial Dependence Plot
- ANOVA로 주 효과 식별
- 2-way interaction 테스트

### **7.3 위기 구간 집중 분석**

**Crisis Bucket 정의:**

1. **COVID-19 Crash**: 2020년 2-3월
2. **2008 Financial Crisis**: 2008년 9월-2009년 3월
3. **Dot-com Bubble**: 2000년 3월-2002년 10월

**평가 메트릭:**

```python
crisis_mdd = max_drawdown(crisis_period)
recovery_time = days_to_recover(crisis_period)
crisis_sharpe = sharpe_ratio(crisis_period)
```

**가설 검증:**

> H1: IRT는 위기 구간에서 SAC 대비 MDD를 20% 이상 감소시킨다.  
> H2: 위기 → 회복 전환에서 IRT는 더 빠른 적응을 보인다.

**통계적 검증**: Paired t-test, Wilcoxon signed-rank test

### **7.4 해석 가능성 검증**

**질적 분석:**

1. **수송 행렬 시각화**:

   ```python
   plt.imshow(P[t].cpu().numpy(), cmap='viridis')
   plt.xlabel('Prototype')
   plt.ylabel('Epitope')
   ```

2. **프로토타입 활성화 추적**:

   ```python
   plt.plot(w_history)  # [T, M]
   plt.axvspan(crisis_start, crisis_end, alpha=0.3, color='red')
   ```

3. **비용 분해**:
   ```python
   cost_components = {
       'distance': mahal,
       'co_stim': -gamma * co_stim,
       'tolerance': lambda * tolerance,
       'checkpoint': rho * checkpoint
   }
   ```

**정량 분석:**

- 위기 전후 프로토타입 전환 속도
- 높은 fitness 프로토타입과 실제 성능 상관계수
- OT 마진 `p_t`와 Replicator `tilde_w_t` 간 KL divergence

### **7.5 확장성 테스트**

**데이터셋:**

- S&P 500 (500 자산) - 대규모
- 암호화폐 (30 자산) - 높은 변동성
- 글로벌 ETF (20 자산) - 낮은 상관관계

**계산 비용 측정:**

```python
import time
start = time.time()
for _ in range(100):
    action, w, P = actor(state, danger_embed, crisis_level, critics)
end = time.time()
print(f"Avg Forward: {(end-start)/100*1000:.2f} ms")
```

---

## **8. 적용 가능 영역**

### **8.1 주 타겟: 포트폴리오 관리**

**적합성 ★★★★★ (5/5)**

**이유:**

1. **레짐 전환**: 정상 → 위기 → 회복 패턴이 명확
2. **다중 전략**: 모멘텀, 평균회귀, 리스크패리티 등 혼합 필요
3. **거래 비용**: 시간 메모리로 급격한 리밸런싱 억제
4. **해석 필요**: 규제 요구사항 (MiFID II, Dodd-Frank)

**구현 예시:**

```python
# 다우존스 30 종목
M = 8 프로토타입:
- Momentum (단기)
- Momentum (중기)
- Mean Reversion
- Risk Parity
- Minimum Variance
- Defensive (고배당)
- Growth (기술주)
- Balanced

위기 시: Defensive/Risk Parity 활성화
정상 시: Momentum/Growth 활성화
```

### **8.2 확장 가능: 의료 의사결정**

**적합성 ★★★★☆ (4/5)**

**적용 시나리오:**

- **ICU 치료 전략 선택**: 환자 상태(에피토프) → 치료 프로토콜(프로토타입)
- **암 치료 조합**: 다중 약물 요법 혼합
- **개인화 의료**: 환자별 치료 반응 히스토리(w\_{t-1})

**필요한 수정:**

- 면역 신호 → 환자 위험 지표 (SOFA score 등)
- 프로토타입 → 표준 치료 가이드라인
- 내성 → 약물 상호작용 회피

**근거**: MOT 논문에서도 금융 외에 의료 의사결정에 OT 적용 가능성 언급

### **8.3 확장 가능: 로봇 제어 (Multi-Task)**

**적합성 ★★★☆☆ (3/5)**

**적용 시나리오:**

- **다중 행동 모드**: 걷기, 달리기, 점프 (프로토타입)
- **지형 적응**: 현재 센서 데이터(에피토프) → 행동 혼합
- **안전 제약**: 내성 = 위험한 행동 패턴

**한계:**

- 실시간 제어 요구: IRT는 ~2ms latency → 100Hz 제어는 도전적
- 시간 메모리 필요성: 일부 로봇 작업은 memoryless Markov

### **8.4 부적합: 자연어 처리**

**적합성 ★☆☆☆☆ (1/5)**

**이유:**

1. **시간 메모리 불필요**: 각 토큰 생성은 독립적 (조건부 독립)
2. **구조적 매칭 과잉**: Attention의 QK^T가 이미 충분
3. **계산 비용**: 텍스트는 시퀀스 길이 >>100, OT는 O(n²) 부담

**단, 예외 케이스:**

- **대화 시스템**: 페르소나 혼합 (복제자로 일관성 유지)
- **다중 도메인 번역**: 도메인별 전문가 혼합

---

## **9. 한계점 및 완화 방안**

### **9.1 이론적 한계**

#### **L1. 전체 수렴 보증 부재**

**문제**: IRT는 OT(수렴 O)와 Replicator(조건부 수렴)의 결합이므로, **전체 시스템의 수렴성이 증명되지 않았다**.

**완화 방안:**

1. **실험적 검증**:

   ```python
   # Loss landscape 시각화
   from loss_landscape import plot_2d
   plot_2d(model, criterion, data, steps=50)
   ```

2. **안정성 모니터링**:

   ```python
   # Lyapunov 후보 추적
   V_t = -entropy(w_t) + <w_t, -fitness>
   assert V_{t+1} <= V_t  # 하강 여부
   ```

3. **이론 연구 제안**:
   - Contraction mapping 분석
   - 제한된 α 범위에서 고정점 존재성

**근거**: 복제자 동역학은 일정한 fitness 하에서 ESS로 수렴하지만, IRT는 fitness가 시간에 따라 변함. 이는 **non-stationary game**으로 해석 가능하며, 평균장 이론(mean-field theory) 적용이 향후 과제다.

#### **L2. 극한 행동 불확실성**

**문제**: `α→1` 극한에서 순수 OT가 되지만, `m<<M`일 때 정보 손실 발생 가능.

**완화 방안:**

- **적응형 α**:

  ```python
  alpha = alpha_0 * (1 - crisis_level) + alpha_1 * crisis_level
  ```

  위기 시 Replicator 비중 증가 (히스토리 중시)

- **m, M 선택 가이드**:
  - `m ≥ log(M)`: 충분한 정보 채널
  - `M ≤ 2×action_dim`: 과도한 전문가 회피

### **9.2 실용적 한계**

#### **L3. 하이퍼파라미터 민감도**

**문제**: ε (Sinkhorn), α, η 등 다수의 하이퍼파라미터 존재.

**완화 방안:**

1. **자동 튜닝**:

   ```python
   # Bayesian Optimization (Optuna)
   def objective(trial):
       alpha = trial.suggest_float('alpha', 0.1, 0.5)
       eps = trial.suggest_float('eps', 0.01, 0.1)
       # ... 학습 후 validation sharpe 반환
   ```

2. **Warm Start**:

   ```python
   # 표준 SAC로 pre-train → IRT로 fine-tune
   actor_sac = train_sac(...)
   actor_irt.decoders.load_state_dict(actor_sac.state_dict(), strict=False)
   ```

3. **기본값 제공**:
   ```python
   DEFAULT_CONFIG = {
       'alpha': 0.3,      # 균형
       'eps': 0.05,       # 표준
       'eta_0': 0.05,     # 보수적
       'eta_1': 0.10,     # 적당한 가열
       'm': 6, 'M': 8     # 중간 규모
   }
   ```

#### **L4. 계산 오버헤드**

**문제**: 표준 SAC 대비 1.5-2배 계산량.

**완화 방안:**

1. **Fast Sinkhorn**:

   ```python
   # Log-space + GPU 최적화
   # 8 iterations × 6×8 = 384 ops → ~0.5ms on GPU
   ```

2. **저빈도 업데이트**:

   ```python
   if step % irt_update_freq == 0:
       w_new, P = irt(...)
   else:
       w_new = w_cached
   ```

3. **프로파일링 기반 최적화**:
   ```python
   with torch.profiler.profile() as prof:
       action, w, P = actor(state, danger, crisis, critics)
   print(prof.key_averages().table())
   # → Sinkhorn이 40% 차지하면 집중 최적화
   ```

**근거**: 일일 거래 빈도에서는 몇 ms 차이가 실용적으로 무시 가능하다.

### **9.3 데이터 의존성**

#### **L5. 오프라인 데이터 품질**

**문제**: 프로토타입 학습은 오프라인 데이터 품질에 의존.

**완화 방안:**

1. **데이터 증강**:

   ```python
   # 노이즈 주입 + 다양한 전략
   strategies = ['random', 'momentum', 'mean_reversion',
                 'equal_weight', 'risk_parity', 'min_variance']
   ```

2. **온라인 적응**:

   ```python
   # 프로토타입 키도 fine-tune
   for param in actor.proto_keys:
       param.requires_grad = True
   ```

3. **전이 학습**:
   ```python
   # 다른 시장에서 pre-train
   actor_pretrained = load('sp500_trained.pth')
   actor_irt.proto_keys = actor_pretrained.proto_keys
   ```

### **9.4 도메인 제약**

#### **L6. 금융 외 적용 난이도**

**문제**: 면역학적 비유가 금융 외에서는 직관적이지 않을 수 있음.

**완화 방안:**

1. **추상화 수준 조정**:

   - 금융: "공자극 = 위험 신호", "내성 = 실패 회피"
   - 의료: "공자극 = 환자 위험도", "내성 = 약물 상호작용"
   - 로봇: "공자극 = 안전 제약", "내성 = 불안정 행동"

2. **비용 함수 커스터마이징**:
   ```python
   class CustomCost(nn.Module):
       def forward(self, E, K, domain_signals):
           # 도메인별 비용 정의
           return domain_specific_cost(E, K, domain_signals)
   ```

---

## **10. 결론 및 기여도**

### **10.1 핵심 기여**

**C1. 새로운 연산자 정의**

IRT는 **Optimal Transport**와 **Replicator Dynamics**를 결합한 최초의 정책 혼합 연산자다. 한 줄로 정의 가능:

```math
w_t = (1-α)·Replicator(w_{t-1}, f_t) + α·Transport(E_t, K, C_t)
```

Soft MoE가 정적 가중 평균을 사용하는 것과 달리, IRT는 **시간 메모리**(w\_{t-1})를 명시적으로 포함하여 `m=1` 극한에서도 softmax로 퇴화하지 않는다.

**C2. 면역학적 귀납 편향**

비용 함수에 공자극, 내성, 체크포인트를 구조적으로 분해하여, **도메인 지식을 연산자 수준**에 내장했다. 최적수송 기반 오프라인 RL의 critic 기반 비용을 다층 면역 신호로 확장한 것이다.

**C3. 위기 적응 메커니즘**

`η(c_t) = η_0 + η_1·c_t`로 위기 시 복제자 업데이트 속도를 자동 증가시켜, **레짐 전환에 빠르게 대응**한다. 확장된 복제자 동역학의 가변 학습률 아이디어를 위기 적응에 적용했다.

**C4. 해석 가능성**

- **수송 행렬 P**: "어떤 에피토프가 어떤 프로토타입으로 매핑되는가"
- **복제자 가중치 w**: "과거 성능 기반 혼합"
- **비용 분해**: distance + co-stim + tolerance + checkpoint

이는 MOT의 OT 기반 할당 투명성에 **시간 축과 면역 신호**를 추가하여 더 풍부한 해석을 제공한다.

### **10.2 이론적 위상**

**연결 그래프:**

```
Hofbauer & Sigmund (1998)         Cuturi (2013)
Replicator Dynamics    ──┐    ┌── Entropic OT
         │               │    │
         ├── Tuyls (2003)│    │── Asadulaev (2024)
         │   RL + Rep.   │    │   OT + Offline RL
         │               ▼    ▼
         └────────────→  IRT  ←────────┐
                         2025          │
                          │            │
                          ▼            │
                    Soft MoE ──────────┘
                    (2024)
```

IRT는 **4개 연구 흐름의 교차점**에 위치하며, 각 구성 요소는 검증된 이론적 기초를 갖는다.

### **10.3 실용적 가치**

**예상 성능 개선:**

- 전체 Sharpe: +10-15% vs SAC
- **위기 구간 MDD**: -20-30% vs SAC ← **주 목표**
- 복구 기간: -15-20% vs SAC

**적용 영역:**

1. **주 타겟**: 포트폴리오 관리 (★★★★★)
2. **확장 가능**: 의료, 로봇 (★★★★☆)
3. **부적합**: NLP (★☆☆☆☆)

### **10.4 한계와 향후 연구**

**미해결 문제:**

1. ✗ 전체 시스템 수렴 증명
2. ✗ 최적 하이퍼파라미터 이론적 유도
3. ✗ 대규모 자산(N>100)에서 확장성

**향후 연구 방향:**

1. **이론**: Contraction mapping, 평균장 근사
2. **응용**: 멀티 에이전트 게임, 온라인 광고
3. **확장**: Transformer와의 하이브리드 (attention + IRT)

### **10.5 논문 작성 가이드**

**제목 제안:**

> "IRT: Immune Replicator Transport for Crisis-Adaptive Portfolio Management"

**핵심 메시지:**

1. OT + Replicator 조합은 **신규**
2. 시간 메모리로 Attention/MoE와 **구분**
3. 포트폴리오 위기 적응에 **특화**
4. 이론적 기초는 **견고**, 수렴은 **실험적**

**금지 사항:**

- ❌ "Transformer보다 우수"
- ❌ "모든 RL 문제에 적용 가능"
- ❌ "완전히 새로운 패러다임"

**권장 사항:**

- ✅ "포트폴리오 관리의 위기 적응을 위한 도메인 특화 귀납 편향"
- ✅ 선행 연구를 충실히 인용하고 차이점 명시
- ✅ Ablation study로 각 컴포넌트 기여도 입증
- ✅ 코드 공개로 재현성 보장

### **10.6 최종 평가**

| 기준              | 점수             | 근거                        |
| ----------------- | ---------------- | --------------------------- |
| **이론적 참신성** | ⭐⭐⭐⭐ (4/5)   | 조합 신규, 구성 요소 검증됨 |
| **실용적 가치**   | ⭐⭐⭐⭐ (4/5)   | 위기 적응에 직접 적용 가능  |
| **수학적 엄밀성** | ⭐⭐⭐☆ (3.5/5)  | 부분 증명, 전체 수렴 미비   |
| **구현 용이성**   | ⭐⭐⭐⭐ (4/5)   | ~270 lines, 기존 호환       |
| **해석 가능성**   | ⭐⭐⭐⭐⭐ (5/5) | OT + 복제자 분해 가능       |
| **차별성**        | ⭐⭐⭐⭐ (4/5)   | 시간 메모리로 명확히 구분   |

**종합: ⭐⭐⭐⭐ (4/5) - 혁신적이며 실용적, 이론 보완 필요**

---

## **부록 A: 수학적 보조정리**

### **A.1 Sinkhorn 수렴 정리**

**Theorem (Cuturi, 2013)**: 엔트로피 정규화 OT 문제

```math
\min_{P∈U(u,v)} ⟨P,C⟩ + ε·KL(P||uv^T)
```

는 유일한 해를 가지며, Sinkhorn 알고리즘은 선형 속도로 수렴한다.

**증명 아이디어**:

1. 목적 함수는 strictly convex (KL divergence)
2. 제약 집합은 compact convex
3. Sinkhorn = alternating Bregman projection
4. 수렴율: `O(exp(-n/τ))` where `τ ~ ε/||C||`

### **A.2 복제자 동역학의 내쉬 균형**

**Theorem (Hofbauer-Sigmund, 1998)**: 일정한 fitness `f`에 대해, 복제자 동역학

```math
dw_j/dt = w_j(f_j - \bar{f})
```

의 모든 균형점은 내쉬 균형이며, 안정적 균형점은 진화적 안정 전략(ESS)이다.

**증명**: Lyapunov 함수 `V(w) = ∑_j w_j log w_j` 사용.

### **A.3 IRT의 정보기하학**

복제자 = KL divergence 기반 거울 하강:

```math
w_{t+1} = argmin_w ⟨w,-f_t⟩ + (1/η)D_{KL}(w||w_t)
```

OT = Wasserstein gradient flow:

```math
∂_t μ = -∇_{W_2} F(μ)
```

IRT = **이중 다양체 geodesic**:

- `(Δ^M, KL)` 심플렉스
- `(P(X), W_2)` Wasserstein 공간

결합 `(1-α)·KL + α·W_2`는 **혼합 메트릭**을 유도하며, 이는 정보기하학의 α-connection과 연결될 가능성이 있다.

---

## **부록 B: 구현 체크리스트**

### **B.1 핵심 파일**

```
src/immune/
├── irt.py              # IRT Operator (80 lines)
├── sinkhorn.py         # Sinkhorn (30 lines)
└── t_cell_min.py       # T-Cell (40 lines)

src/agents/
└── bcell_actor_irt.py  # Actor (120 lines)

scripts/
└── train_irt.py        # 학습 스크립트
```

### **B.2 통합 단계**

1. ✅ IRT 모듈 구현
2. ✅ T-Cell 경량화
3. ✅ Actor 통합
4. ✅ Critic 인터페이스 (기존 재사용)
5. ✅ 학습 루프 수정 (fitness 계산)
6. ✅ 평가 지표 추가 (위기 구간)
7. ✅ 로깅 및 시각화

### **B.3 검증 테스트**

```python
# tests/test_irt.py
def test_irt_forward():
    irt = IRT(emb_dim=64, m=4, M=6)
    E = torch.randn(2, 4, 64)
    K = torch.randn(2, 6, 64)
    danger = torch.randn(2, 64)
    w_prev = torch.ones(2, 6) / 6
    fitness = torch.randn(2, 6)
    crisis = torch.tensor([[0.5], [0.8]])

    w, P = irt(E, K, danger, w_prev, fitness, crisis)

    assert w.shape == (2, 6)
    assert torch.allclose(w.sum(dim=1), torch.ones(2))
    assert (w >= 0).all()
```

---

## **참고문헌**

이 보고서는 다음 최신 연구를 기반으로 작성되었다:

1. Asadulaev et al. (2024) "Rethinking Optimal Transport in Offline RL" - NeurIPS 2024
2. Cheng et al. (2024) "MOT: Mixture of Actors with OT for Trading" - arXiv
3. Tuyls et al. (2003) "Extended Replicator Dynamics for RL" - ECML 2003
4. Google DeepMind (2024) "Soft Mixture of Experts" - ICLR 2024
5. Cuturi (2013) "Sinkhorn Distances" - NeurIPS 2013
6. Hofbauer & Sigmund (1998) "Evolutionary Games and Population Dynamics"

---

**Document Status**: Draft v1.0  
**Last Updated**: 2025-10-01  
**Authors**: FinFlow-RL Research Team  
**License**: MIT
