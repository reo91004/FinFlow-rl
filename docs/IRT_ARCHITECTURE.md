# IRT Architecture Documentation

## Overview

IRT (Immune Replicator Transport) is a novel policy mixing operator that combines Optimal Transport with Replicator Dynamics for crisis-adaptive portfolio management.

## Mathematical Foundation

### Core Equation
```
w_t = (1-α)·Replicator(w_{t-1}, f_t) + α·Transport(E_t, K, C_t)
```

Where:
- `w_t`: Portfolio weights at time t
- `α`: Mixing coefficient (0=pure replicator, 1=pure OT)
- `E_t`: Epitope embeddings (current state features)
- `K`: Prototype keys (expert strategies)
- `C_t`: Cost matrix with immunological biases

### Cost Matrix Design
```
C_ij = d(e_i,k_j) - γ<e_i,d_t> + λ[tolerance] + ρ[checkpoint]
```

Components:
- **Distance**: Mahalanobis distance between epitopes and prototypes
- **Co-stimulation**: Alignment with danger signals (-γ term)
- **Tolerance**: Suppression of self-similar patterns (λ term)
- **Checkpoint**: Overconfidence penalty (ρ term)

## Architecture Components

### 1. IRT Operator (`src/immune/irt.py`)
- **Sinkhorn Algorithm**: Entropic optimal transport solver
- **Mahalanobis Metric**: Learnable distance metric
- **Replicator Dynamics**: Time memory via w_{t-1}
- **Debug Output**: Returns `(w, P, debug_info)` where debug_info contains:
  - `w_rep`: Replicator-only weights [B, M]
  - `w_ot`: OT-only weights [B, M]
  - `cost_matrix`: Immunological costs [B, m, M]
  - `eta`: Crisis-adaptive learning rate [B, 1]

### 2. T-Cell (`src/immune/t_cell.py`)
- **Crisis Detection**: Multi-dimensional crisis types
- **Danger Embedding**: Co-stimulation signals
- **Crisis Level**: Scalar crisis intensity

### 3. B-Cell Actor (`src/agents/bcell_irt.py`)
- **Epitope Encoder**: State → Multiple tokens
- **Prototype Decoders**: M independent Dirichlet policies
- **IRT Integration**: Weight mixing via OT+Replicator

### 4. REDQ Critics (`src/algorithms/critics/redq.py`)
- **Ensemble**: N Q-networks (default N=10)
- **Min-Q Target**: Conservative value estimation
- **High UTD Ratio**: Sample-efficient learning

## Training Pipeline

### Phase 1: Offline Pretraining (Optional)
- IQL (Implicit Q-Learning) with expectile regression
- Conservative value learning from historical data

### Phase 2: Online IRT Fine-tuning
- REDQ critics for stable Q-learning
- IRT actor for adaptive policy mixing
- Crisis-aware exploration/exploitation

## Key Innovations

### 1. Time Memory
Unlike attention/MoE, IRT maintains explicit time memory through w_{t-1}, preventing collapse to softmax in single-token limit.

### 2. Immunological Inductive Bias
Cost function incorporates domain knowledge:
- Co-stimulation for crisis signals
- Tolerance for failure avoidance
- Checkpoint for overconfidence suppression

### 3. Crisis Adaptation
Dynamic learning rate η(c) = η_0 + η_1·c automatically increases adaptation speed during crises.

## Performance Targets

| Metric | Baseline (SAC) | IRT Target | Improvement |
|--------|---------------|-----------|-------------|
| Sharpe Ratio | 1.2 | 1.4+ | +17% |
| Crisis MDD | -35% | -25% | -29% |
| Recovery Time | 45 days | 35 days | -22% |
| CVaR (5%) | -3.5% | -2.5% | -29% |

## Configuration

### Default Settings (`configs/default_irt.yaml`)
```yaml
irt:
  # Basic Structure
  emb_dim: 128       # Embedding dimension
  m_tokens: 6        # Number of epitope tokens
  M_proto: 8         # Number of prototypes
  alpha: 0.3         # OT-Replicator mixing ratio

  # Sinkhorn Algorithm
  eps: 0.10          # Sinkhorn entropy (updated: 0.05 → 0.10)
  max_iters: 10      # Maximum iterations
  tol: 0.001         # Convergence threshold

  # Cost Function Weights
  gamma: 0.5         # Co-stimulation weight
  lambda_tol: 2.0    # Tolerance weight
  rho: 0.3           # Checkpoint weight

  # Crisis Heating (Replicator)
  eta_0: 0.05        # Base learning rate
  eta_1: 0.15        # Crisis increase (updated: 0.10 → 0.15)

  # Self-Tolerance
  kappa: 1.0         # Tolerance gain
  eps_tol: 0.1       # Tolerance threshold
  n_self_sigs: 4     # Number of self signatures

  # EMA Memory
  ema_beta: 0.9      # Exponential moving average coefficient

objectives:
  lambda_turn: 0.01  # Turnover penalty (updated: 0.1 → 0.01)
  lambda_cvar: 1.0   # CVaR constraint weight
  lambda_dd: 0.0     # Drawdown penalty
```

### Ablation Studies
- `α=0`: Pure Replicator (temporal consistency)
- `α=0.3`: Balanced (default)
- `α=1`: Pure OT (structural matching)

### Hyperparameter Tuning Guide

#### Problem: No-Trade Loop (무거래 루프)

**증상**:
- Episode 전체에서 turnover ≈ 0
- 프로토타입 가중치가 균등 분포 유지 (1/M)
- 균등 배분(1/N) 정책 반복

**진단**:
1. **환경 레벨**: Turnover penalty가 거래 억제
   - `lambda_turn`이 일일 수익률(±1%)과 비슷한 스케일
2. **IRT 레벨**: Exploration 메커니즘 억제
   - Sinkhorn `eps` 너무 낮음 → deterministic OT
   - Dirichlet concentration 너무 높음 → deterministic policy

**해결 방법** (REFACTORING.md 철학: 기존 메커니즘 활용):

| 파라미터 | 변경 전 | 변경 후 | 근거 |
|---------|---------|---------|------|
| `lambda_turn` | 0.1 | 0.01 | 일일 수익률 스케일 정합 |
| `eps` | 0.05 | 0.10 | Cuturi (2013) 권장 범위 |
| `eta_1` | 0.10 | 0.15 | 빠른 위기 적응 (최대 0.20) |
| Dirichlet `min` | 1.0 | 0.5 | α<1 sparse, 높은 엔트로피 |
| Dirichlet `max` | 100 | 50 | Over-confidence 방지 |

**이론적 근거**:

1. **Sinkhorn Entropy** (REFACTORING.md:151):
   ```math
   min_{P∈U(u,v)} <P,C> + ε·KL(P||uv^T)
   ```
   - ε↑ → 수송 계획 P가 균등 분산 → exploration 증가

2. **Dirichlet Exploration**:
   - α_k < 1: Sparse 선호 (높은 엔트로피)
   - α_k → ∞: Deterministic (엔트로피 0)
   - min=0.5: 안전한 exploration 범위

3. **Turnover Penalty 스케일**:
   ```
   변경 전: 10% turnover → penalty 0.01 (수익률 ±1%와 동일 → 거래 억제)
   변경 후: 10% turnover → penalty 0.001 (합리적 수준)
   ```

**복잡도**: 0 (설정 파일 3줄 + 코드 1줄)

#### Recommended Ranges

| 파라미터 | 최소 | 기본 | 최대 | 용도 |
|---------|-----|------|------|------|
| `alpha` | 0.1 | 0.3 | 0.5 | OT-Replicator 균형 |
| `eps` | 0.01 | 0.10 | 0.2 | Exploration (높을수록 다양) |
| `eta_0` | 0.03 | 0.05 | 0.08 | 기본 적응 속도 |
| `eta_1` | 0.05 | 0.15 | 0.20 | 위기 가열 (높을수록 빠름, 불안정 주의) |
| `m_tokens` | 4 | 6 | 8 | 상태 정보 채널 수 |
| `M_proto` | 6 | 8 | 12 | 전략 다양성 (너무 많으면 과적합) |
| `lambda_turn` | 0.001 | 0.01 | 0.1 | 거래 비용 (낮을수록 자유로운 거래) |

**Warning**: `eta_1 > 0.20`은 불안정 가능성 (REFACTORING.md 경고)

## Usage

### Training
```bash
python scripts/train_irt.py --config configs/default_irt.yaml
```

### Evaluation
```bash
# Evaluation automatically generates 12 visualizations
python scripts/evaluate_irt.py --checkpoint logs/*/checkpoints/best_model.pth --config configs/default_irt.yaml

# Or via main.py
python main.py --mode evaluate --resume logs/*/checkpoints/best_model.pth
```

**Output Structure**:
```
logs/YYYYMMDD_HHMMSS/evaluation/
├── evaluation_results.json     # Raw data (metrics, returns, weights, IRT decomposition)
└── visualizations/             # 12 PNG files (auto-generated)
    ├── irt_decomposition.png   # [NEW] OT vs Replicator decomposition
    ├── tcell_analysis.png      # [NEW] Crisis types & regimes
    ├── cost_matrix.png         # [NEW] Immunological costs
    ├── stock_analysis.png      # [NEW] Stock-level attribution
    ├── attribution_analysis.png # [NEW] Contribution breakdown
    ├── performance_timeline.png # [NEW] Rolling metrics
    ├── benchmark_comparison.png # [NEW] vs Equal-weight
    ├── risk_dashboard.png      # [NEW] VaR/CVaR/Drawdown
    ├── portfolio_weights.png
    ├── returns.png
    ├── crisis_levels.png
    └── prototype_weights.png
```

### Legacy Visualization (Deprecated)
```bash
# Old standalone script (now integrated into evaluate_irt.py)
python scripts/visualize_irt.py --results logs/*/evaluation/evaluation_results.json
```

## 평가 및 시각화

### 자동 시각화 파이프라인

**v2.0.3 (2025-10-03)** 부터, `evaluate_irt.py`는 IRT 의사결정에 대한 완전한 설명 가능성을 제공하는 **12개의 종합 시각화**를 자동 생성한다.

#### 🔬 IRT 메커니즘 분석 (3개 플롯)

1. **IRT 분해** (`irt_decomposition.png`)
   - 대표 프로토타입에 대한 **w = (1-α)·w_rep + α·w_ot** 분해
   - 모든 프로토타입의 L2 norm 비교
   - 위기 레벨 대비 위기 적응형 학습률 **η(c) = η₀ + η₁·c**
   - **사용 사례**: IRT 혼합 검증, 위기 적응 확인

2. **T-Cell 분석** (`tcell_analysis.png`)
   - Crisis type 분포 (평균 + 상위 3개 시계열)
   - Crisis level vs returns 산점도
   - Crisis type 상관관계 히트맵
   - 위기 구간별 성과 (낮음/중간/높음)
   - **사용 사례**: 위기 감지 이해, T-Cell 동작 검증

3. **비용 행렬** (`cost_matrix.png`)
   - 평균 면역학적 비용 행렬 (γ, λ, ρ 효과)
   - 비용 분포 히스토그램
   - Early vs Late episode 비용 진화
   - **사용 사례**: 면역학적 편향 검사, OT 동작 디버그

#### 💼 포트폴리오 분석 (3개 플롯)

4. **종목 분석** (`stock_analysis.png`)
   - 평균 가중치 기준 상위 10개 보유 종목 (**실제 심볼**: AAPL, MSFT 등)
   - 가장 역동적인 상위 10개 종목 (가중치 변동성을 통한 위기 민감도)
   - **사용 사례**: 포트폴리오 구성, 위기 민감 자산 식별

5. **기여도 분석** (`attribution_analysis.png`)
   - 종목별 수익 기여도 분해 (상위 10개 누적, 상위 3개 시계열)
   - 프로토타입 활용도 (평균 가중치)
   - 프로토타입 성과 기여도 (높은 활성화 시 수익률)
   - **사용 사례**: 최고 성과 종목 식별, 전략 효과성

6. **포트폴리오 가중치** (`portfolio_weights.png`)
   - 시간에 따른 전체 자산 가중치 스택 차트
   - **사용 사례**: 전체 포트폴리오 진화

#### 📈 성과 & 리스크 (4개 플롯)

7. **성과 타임라인** (`performance_timeline.png`)
   - Rolling Sharpe Ratio (60일 윈도우, 목표=1.5)
   - Drawdown 타임라인 (목표=-25%)
   - 포트폴리오 회전율
   - **사용 사례**: 시간에 따른 성과 품질 추적

8. **벤치마크 비교** (`benchmark_comparison.png`)
   - IRT vs Equal-weight 누적 수익률
   - Outperformance/Underperformance 영역
   - **사용 사례**: 알파 생성 검증

9. **리스크 대시보드** (`risk_dashboard.png`) - 2×2 격자:
   - 수익률 분포 + VaR(5%), CVaR(5%)
   - Drawdown waterfall (Max DD 강조)
   - Risk-return 산점도 (rolling windows)
   - 위기 vs 비위기 수익률 (boxplot)
   - **사용 사례**: 종합 리스크 평가

10. **수익률** (`returns.png`)
    - 일일 수익률 시계열
    - 누적 수익률
    - **사용 사례**: 기본 성과 확인

#### 🧬 IRT 컴포넌트 (2개 플롯)

11. **위기 레벨** (`crisis_levels.png`)
    - 임계값(0.3, 0.7)을 포함한 T-Cell 위기 레벨 감지
    - **사용 사례**: 위기 감지 타임라인

12. **프로토타입 가중치** (`prototype_weights.png`)
    - 개별 프로토타입 가중치 (M=8)
    - 프로토타입 다양성 엔트로피
    - **사용 사례**: 전략 다양성 확인

### XAI 오버헤드

**학습 시**: 거의 0에 가까운 오버헤드 (<0.1%)
- Debug info는 이미 계산된 중간 값을 재사용
- 추가 forward pass 없음

**평가 시**: 12개 시각화에 약 5-10초
- 평가 완료 후 1회성 비용
- 평가 정확도에 영향 없음

### 고급 XAI (선택 사항)

더 깊은 분석을 원한다면 `src/evaluation/explainer.py` 사용 (수동 통합 필요):
- **SHAP values**: 특성 중요도
- **Integrated Gradients**: Attribution 분석
- **LIME**: 지역적 해석 가능성

## 해석 가능성 (Interpretability)

IRT는 다음을 통해 풍부한 해석 가능성을 제공한다:

1. **수송 행렬 P**: 에피토프-프로토타입 매핑 표시
2. **프로토타입 가중치 w**: 전략 혼합 공개
3. **IRT 분해**: OT vs Replicator 기여도 분리
4. **비용 행렬**: 면역학적 의사결정 요인 분해
5. **위기 분석**: 구간 분해를 포함한 다차원 위기 기여도
6. **종목 기여도**: 포트폴리오 레벨 기여도 분석

## 이론적 보장

### 증명된 속성
- Sinkhorn은 O(1/ε) 반복에서 수렴 (Cuturi, 2013)
- Replicator는 고정 적합도 하에서 ESS로 수렴 (Hofbauer & Sigmund, 1998)
- 엔트로피 정규화에서 OT 해는 유일함

### 미해결 문제
- OT+Replicator 결합 시스템의 완전한 수렴 증명
- 최적 α 선택 이론
- N>100 자산으로의 확장성

## 하위 호환성 (Backward Compatibility)

**⚠️ 호환성 깨짐 (BREAKING CHANGE) (v2.0.3, 2025-10-03)**

IRT Operator 시그니처가 다음과 같이 변경되었다:
```python
# 이전 (v2.0.2)
def forward(...) -> Tuple[torch.Tensor, torch.Tensor]:
    return w, P

# 신규 (v2.0.3)
def forward(...) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    return w, P, debug_info
```

**영향**:
- **이전 체크포인트** (v2.0.2 이하)는 `ValueError: not enough values to unpack` 오류 발생
- **해결 방법**: 새 하이퍼파라미터로 재학습 (권장) 또는 하위 호환 unpacking 구현

**권장 조치**: 무거래 루프 해결을 위해 업데이트된 하이퍼파라미터(`lambda_turn=0.01`, `eps=0.10`, `eta_1=0.15`)로 재학습.

**하위 호환 로딩** (재학습이 불가능한 경우):
```python
# src/agents/bcell_irt.py:193
irt_output = self.irt(...)
if isinstance(irt_output, tuple) and len(irt_output) == 3:
    w, P, irt_debug = irt_output
else:
    w, P = irt_output
    irt_debug = {'w_rep': None, 'w_ot': None, 'cost_matrix': None, 'eta': None}
```

## 참고 문헌

1. Cuturi (2013) "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
2. Hofbauer & Sigmund (1998) "Evolutionary Games and Population Dynamics"
3. Chen et al. (2021) "Randomized Ensembled Double Q-Learning"
4. Kostrikov et al. (2021) "Implicit Q-Learning"

## 문의

질문이나 기여는 저장소에 issue를 열어주세요.