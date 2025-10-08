# IRT 개선 로드맵

## 개요

이 문서는 IRT (Immune Replicator Transport) 아키텍처의 종합적인 개선 계획을 우선순위 Tier별로 정리한다. 모든 개선사항은 SAC baseline 구조를 유지하면서 최소 침습성과 학술적 엄밀성을 달성한다.

## 설계 원칙

1. **SAC Baseline 보존**: 기존 SAC 구조 위에 IRT overlay 유지
2. **최소 침습성**: 과도한 아키텍처 수정 회피
3. **학술적 정당성**: 2024-2025년 연구 문헌 기반
4. **실현 가능성**: 연구 논문 범위 내에서 완료 가능

---

## 우선순위 구조

### **Tier 0: Critical (1주차)**
학술적 정당성 확보 및 성능 개선에 필수적

### **Tier 1: High Priority (2주차)**
성능 향상 및 논문 기여도 강화

### **Tier 2: Medium Priority (3-4주차)**
용어 정정 및 ablation study

### **Tier 3: Future Work (논문 이후)**
높은 잠재력을 가진 새로운 기여도이나 현재 범위를 벗어남

---

## Tier 0: Critical 개선사항

### 0-1. Crisis Threshold Smooth Transition ⭐⭐⭐⭐⭐

**문제**: 현재 hard threshold (crisis > 0.5)는 불연속적이고 학술적으로 정당화하기 어려움

**현재 구현**:
```python
# finrl/agents/irt/irt_operator.py (Lines 277-283)
adaptive_alpha = torch.where(
    crisis_level > crisis_threshold,
    torch.zeros_like(crisis_level) * self.alpha * 0.2,  # Crisis: 0.06
    torch.ones_like(crisis_level) * self.alpha           # Normal: 0.30
)
```

**문제점**:
- Hard threshold: 불연속적
- Gradient-unfriendly: 미분 불가능
- 0.5 경계에서 급격한 전환
- **리뷰어 거부 위험 높음**

**해결책**: Gradient-friendly 전환을 위한 smooth cosine interpolation

**수학적 근거**:

2025년 LLM-RL 연구에서 cosine annealing의 효과성이 검증됨:

$$\alpha_{smooth}(c) = \alpha_{normal} + (\alpha_{crisis} - \alpha_{normal}) \cdot \frac{1 - \cos(\pi \cdot c)}{2}$$

여기서:
- $c \in [0, 1]$: T-Cell crisis level
- $\alpha_{normal} = 0.30$: 평시 OT 비중
- $\alpha_{crisis} = 0.06$: 위기시 OT 비중
- Cosine interpolation: 부드러운 gradient를 가진 smooth transition

**구현** (`finrl/agents/irt/irt_operator.py`, 15줄 수정):

```python
# finrl/agents/irt/irt_operator.py

class IRTOperator(nn.Module):
    def forward(self, E, K, w_prev, fitness, crisis_level):
        # ... 기존 OT, Replicator 계산 ...

        # ===== MODIFIED: Smooth Crisis-Adaptive Mixing =====
        # 이전: hard threshold (불연속적)
        # crisis_threshold = 0.5
        # adaptive_alpha = torch.where(crisis_level > crisis_threshold, 0.06, 0.30)

        # 현재: cosine interpolation (연속적, gradient-friendly)
        alpha_normal = 0.30
        alpha_crisis = 0.06
        pi = torch.tensor(3.14159265, device=crisis_level.device)

        # Smooth transition: α(c) = α_n + (α_c - α_n) * (1 - cos(πc)) / 2
        # c=0 → α=0.30 (평시), c=1 → α=0.06 (위기)
        smooth_alpha = alpha_normal + (alpha_crisis - alpha_normal) * \
                       (1 - torch.cos(pi * crisis_level)) / 2

        # Safety clipping
        smooth_alpha = torch.clamp(smooth_alpha, min=alpha_crisis, max=alpha_normal)

        # IRT mixing (Adaptive-Exploratory mixing)
        # 위기: Adaptive mechanism 우선 (Replicator 94%)
        # 평시: Exploratory mechanism 보조 (OT 30%)
        w = (1 - smooth_alpha) * tilde_w + smooth_alpha * p_mass

        # ... 나머지 코드 동일 ...

        debug_info['adaptive_alpha'] = smooth_alpha  # 호환성을 위해 이름 유지

        return w, P, debug_info
```

**예상 효과**:
- 학술적 정당성 확보
- Crisis detection 완전 활성화
- Gradient 안정성 개선

---

### 0-2. 학습 중 XAI 통합 ⭐⭐⭐⭐⭐

**문제**: 현재 XAI는 사후 분석에만 사용됨 (평가 중 시각화만)

**해결책**: 학습 중 auxiliary loss로 해석가능성 통합

**수학적 근거**:

IRT 분해: $w = (1-\alpha) \cdot w_{rep} + \alpha \cdot w_{ot}$

**정규화 목표**:
1. **Diversity**: 각 메커니즘이 다양한 프로토타입 활용 (붕괴 방지)
2. **Crisis specialization**: 위기시 Adaptive 메커니즘 집중

$$\mathcal{L}_{xai} = -\lambda_1 \mathbb{H}(w_{rep}) - \lambda_2 \mathbb{H}(w_{ot}) + \lambda_3 \| \text{HHI}(w_{rep}) - \text{target}(c) \|^2$$

여기서:
- $\mathbb{H}(w) = -\sum_j w_j \log w_j$ (다양성을 위한 엔트로피)
- $\text{HHI}(w) = \sum_j w_j^2$ (집중도를 위한 Herfindahl-Hirschman Index)
- $\text{target}(c) = 0.3 + 0.4 \cdot c$ (Crisis-adaptive 집중도 목표)

**구현** (`finrl/agents/irt/irt_policy.py`, 50줄 추가):

```python
# finrl/agents/irt/irt_policy.py

class IRTPolicy(SACPolicy):
    def __init__(
        self,
        *args,
        xai_reg_weight: float = 0.01,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.xai_reg_weight = xai_reg_weight
        self._xai_loss = 0.0

    def _compute_xai_regularization(self, irt_info: dict) -> torch.Tensor:
        """
        XAI 정규화 loss

        목표:
        1. w_rep, w_ot이 다양한 프로토타입 사용 (엔트로피 최대화)
        2. 위기시 Adaptive 메커니즘 집중 (HHI 조정)
        """
        w_rep = irt_info['w_rep']  # [B, M]
        w_ot = irt_info['w_ot']    # [B, M]
        crisis_level = irt_info['crisis_level']  # [B, 1]

        eps = 1e-8

        # 1. Diversity: 엔트로피 최대화 (붕괴 방지)
        entropy_rep = -(w_rep * torch.log(w_rep + eps)).sum(dim=-1).mean()
        entropy_ot = -(w_ot * torch.log(w_ot + eps)).sum(dim=-1).mean()

        # 2. Crisis-adaptive specialization
        hhi_rep = (w_rep ** 2).sum(dim=-1)  # [B]

        # Target HHI: 평시 0.3 (분산), 위기 0.7 (집중)
        target_hhi = 0.3 + 0.4 * crisis_level.squeeze(-1)  # [B]
        hhi_loss = ((hhi_rep - target_hhi) ** 2).mean()

        # 3. Total loss
        loss = (
            -0.01 * entropy_rep      # Adaptive diversity
            -0.01 * entropy_ot       # Exploratory diversity
            + 0.05 * hhi_loss        # Crisis-adaptive concentration
        )

        return loss
```

**예상 효과**:
- Prototype diversity 향상
- Crisis adaptation 완전 활성화
- XAI를 post-hoc에서 training으로 통합

---

### 0-3. Reward Fine-Tuning ⭐⭐⭐⭐

**문제**: Phase 2.1의 λ 값이 휴리스틱함 (수동 튜닝)

**현재 파라미터**:
```python
lambda_turnover = 0.005
lambda_diversity = 0.05
lambda_drawdown = 0.05
target_turnover = 0.06
turnover_band = 0.03
target_hhi = 0.20
```

**해결책**: Grid search 기반 체계적 최적화

**제안 파라미터**:

| 파라미터 | 현재 | 제안 | 근거 |
|----------|------|------|------|
| `lambda_turnover` | 0.005 | **0.003** | Turnover 제약 완화 |
| `lambda_diversity` | 0.05 | **0.03** | 일부 집중 허용 |
| `lambda_drawdown` | 0.05 | **0.07** | Drawdown 관리 강화 |
| `target_turnover` | 0.06 | **0.08** | 업계 표준 수준 |
| `turnover_band` | 0.03 | **0.04** | 유연성 증가 |
| `target_hhi` | 0.20 | **0.25** | 적당한 집중 |

**Grid Search Script** (`scripts/grid_search_reward.sh`):

```bash
#!/bin/bash
# Grid search (3x3x3 = 27 조합)

for lambda_t in 0.002 0.003 0.004; do
  for lambda_d in 0.02 0.03 0.04; do
    for lambda_dd in 0.05 0.07 0.09; do
      echo "Testing: λ_t=${lambda_t}, λ_d=${lambda_d}, λ_dd=${lambda_dd}"

      python scripts/train_irt.py \
        --episodes 50 \
        --lambda-turnover $lambda_t \
        --lambda-diversity $lambda_d \
        --lambda-drawdown $lambda_dd \
        --output "logs/grid_search/${lambda_t}_${lambda_d}_${lambda_dd}" \
        --no-plot
    done
  done
done
```

**예상 효과**:
- 체계적인 파라미터 최적화
- Turnover 정상화
- Max Drawdown 감소

---

## Tier 1: High Priority 개선사항

### 1-1. Prototype Diversity Regularization ⭐⭐⭐

**문제**: 학습 중 prototype collapse 위험

**해결책**: Batch-averaged 엔트로피 최대화

**수학적 근거**:

$$\mathcal{L}_{div} = -\mathbb{H}(\mathbb{E}[w]) + \lambda \cdot \text{Var}(w)$$

여기서:
- $\mathbb{E}[w]$: Batch 평균 가중치 (균일 분포 장려)
- $\text{Var}(w)$: 개별 state variance (다양성 장려)
- Target entropy: $\log(M)$ (M=8 → 2.08 nats)

**구현** (`finrl/agents/irt/bcell_actor.py`, 30줄 추가):

```python
def _compute_diversity_loss(self, w: torch.Tensor) -> torch.Tensor:
    """
    Prototype diversity 정규화

    목표:
    1. Batch 평균에서 모든 프로토타입 균일 사용
    2. 개별 state에서 적절한 프로토타입 선택
    """
    eps = 1e-8
    M = w.shape[1]
    target_entropy = np.log(M)  # 균일 분포 엔트로피

    # 1. Batch 평균 엔트로피 (균일 장려)
    w_mean = w.mean(dim=0)  # [M]
    entropy_mean = -(w_mean * torch.log(w_mean + eps)).sum()
    entropy_loss = (target_entropy - entropy_mean) ** 2

    # 2. 개별 variance (다양성 장려)
    w_var = w.var(dim=1).mean()
    var_penalty = -0.01 * w_var  # 최대화를 위해 음수

    return entropy_loss + var_penalty
```

---

### 1-2. 3-Way Comparison 실험 ⭐⭐⭐

**목표**: Architecture vs Reward design 기여도 분리

**실험 설계**:

| Configuration | Architecture | Reward | 목적 |
|--------------|--------------|--------|------|
| **Baseline SAC** | Standard SAC | Simple (ln V) | 기준선 |
| **SAC + Multi-Objective** | Standard SAC | Multi-objective | **Reward 기여도** |
| **IRT + Multi-Objective** | IRT | Multi-objective | **Architecture + Reward** |

**분석**:
```
Architecture 기여도 = (IRT+MO) - (SAC+MO)
Reward 기여도 = (SAC+MO) - (Baseline)
Total 기여도 = (IRT+MO) - (Baseline)
```

**구현** (`scripts/run_comparison.sh`):

```bash
#!/bin/bash
EPISODES=200

# 실험 1: Baseline SAC
python scripts/train.py --model sac --episodes $EPISODES --output logs/comparison/baseline_sac

# 실험 2: SAC + Multi-Objective
python scripts/train.py --model sac --episodes $EPISODES --use-multiobjective --output logs/comparison/sac_multiobjective

# 실험 3: IRT + Multi-Objective
python scripts/train_irt.py --episodes $EPISODES --output logs/comparison/irt_multiobjective

# 분석
python scripts/analyze_comparison.py --input logs/comparison
```

---

## Tier 2: Medium Priority 개선사항

### 2-1. Ablation Study ⭐⭐

**목표**: 각 구성요소의 필요성 검증

**연구 항목**:

| Study | Variable | Values | 목적 |
|-------|----------|--------|------|
| **Prototype 개수** | M | 4, 6, 8 | 최적 prototype 수 결정 |
| **Decoder 구조** | Architecture | Separate, Shared | 파라미터 효율성 vs 성능 |
| **T-Cell 유무** | Crisis detection | With, Without | T-Cell 필요성 검증 |

**구현** (`scripts/run_ablation.sh`):

```bash
#!/bin/bash
EPISODES=100

# Study 1: Prototype 개수
for M in 4 6 8; do
  python scripts/train_irt.py --episodes $EPISODES --M-proto $M --output logs/ablation/M_$M
done

# Study 2: Decoder 구조
python scripts/train_irt.py --episodes $EPISODES --shared-decoder --output logs/ablation/decoder_shared

# Study 3: T-Cell
python scripts/train_irt.py --episodes $EPISODES --no-tcell --output logs/ablation/without_tcell
```

---

### 2-2. 용어 정정 ⭐⭐

**문제**: 현재 문서가 Replicator/OT 메커니즘을 잘못 설명함

**수정 사항**:

| 이전 | 이후 | 근거 |
|------|------|------|
| Replicator: "과거 경험" | **"Adaptive (현재 fitness gradient)"** | 2024-2025 Replicator Dynamics 연구 |
| OT: "구조적 매칭" | **"Exploratory (fitness 무관)"** | 본질 명확화 |
| "보수적/진보적" | **"적응적/탐색적"** | 혼란 방지 |

**핵심 통찰**:
- **Adaptive Mechanism (Replicator)**: 현재 fitness gradient 추종, 빠른 적응
- **Exploratory Mechanism (OT)**: 구조적 유사성 기반, fitness 무관 탐색

---

## Tier 3: Future Work

### 3-1. Learnable Alpha Mixer ⭐⭐⭐⭐⭐ (Novel)

**개념**: End-to-end 학습 가능한 Adaptive-Exploratory mixing

**수학적 정의**:

$$\alpha(c, s) = \sigma(f_\theta([c, s]))$$

여기서:
- $c$: Crisis level
- $s$: Market state features (변동성, 스트레스, 상관관계)
- $f_\theta$: 2-layer MLP (64 hidden)
- $\sigma$: Sigmoid

**장점**:
- ⭐ **Novel**: 문헌에서 찾을 수 없음
- End-to-end 학습
- Crisis + market state 고려
- 더 정교한 mixing

**구현** (참고용):

```python
class LearnableAlphaMixer(nn.Module):
    """Meta-learned Adaptive-Exploratory mixing"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, crisis_level: torch.Tensor, market_features: torch.Tensor):
        x = torch.cat([crisis_level, market_features], dim=-1)
        alpha_normalized = self.net(x)  # [B, 1] in [0, 1]

        # [0.06, 0.30]으로 스케일
        alpha_min, alpha_max = 0.06, 0.30
        alpha = alpha_min + (alpha_max - alpha_min) * alpha_normalized

        return alpha
```

---

## 구현 일정

### **1주차: Tier 0 (Critical)**
- 1-2일차: Crisis threshold smooth transition
- 3-4일차: XAI 정규화
- 5일차: Reward fine-tuning 기본값
- 6-7일차: Grid search 실행

### **2주차: Tier 1 (High Priority)**
- 8-10일차: Prototype diversity 정규화
- 11-14일차: 3-way comparison 실험

### **3-4주차: Tier 2 (Medium Priority)**
- 3주차: Ablation studies
- 4주차: 용어 정정 및 문서화

### **Future (Tier 3)**
- Learnable alpha mixer
- 논문 Discussion/Future Work 섹션

---

## 논문 기여도

### **확정 기여도**
1. ✅ **OT + Replicator 하이브리드**: Novel (문헌 조사 필요)
2. ⭐ **Smooth crisis-adaptive mixing**: 2025년 연구 기반, gradient-friendly
3. ✅ **Multi-objective reward**: 3-way comparison으로 검증
4. ⭐ **XAI-guided learning**: 해석가능성을 auxiliary loss로 활용
5. ⭐ **Adaptive-Exploratory framework**: 학술적으로 정당화된 용어

### **추가 가능 기여도** (Tier 2 완료 시)
6. ✅ **Prototype diversity regularization**: 붕괴 방지
7. ✅ **Fine-tuned multi-objective**: Grid search 기반 최적화
8. ✅ **Ablation study**: M, Decoder, T-Cell 기여도 검증

### **Future Work** (Tier 3)
9. ⭐⭐⭐ **Meta-learned OT-Replicator mixing**: Novel + End-to-end

---

## 체크리스트

### **Tier 0 (필수)**
- [ ] Crisis threshold smooth transition 구현
- [ ] XAI 정규화 구현
- [ ] Reward fine-tuning 기본값 변경
- [ ] Grid search 실행 및 최적 선택

### **Tier 1 (강력 권장)**
- [ ] Prototype diversity 정규화 구현
- [ ] 3-way comparison 실험 실행

### **Tier 2 (논문 보강)**
- [ ] Ablation study (M, Decoder, T-Cell)
- [ ] 용어 정정 (docs + 주석)

### **Tier 3 (Future)**
- [ ] Learnable alpha mixer 구현 (선택사항)
- [ ] 논문 Discussion 섹션

---

**모든 개선사항은 SAC + IRT 구조를 유지하고, 최소 침습적이며, FinRL 환경과 호환되고, 학술적으로 정당화된다.**