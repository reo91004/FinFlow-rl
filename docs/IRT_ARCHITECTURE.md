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
python scripts/evaluate_irt.py --checkpoint logs/*/checkpoints/best_model.pth
```

### Visualization
```bash
python scripts/visualize_irt.py --results logs/*/evaluation/evaluation_results.json
```

## Interpretability

IRT provides rich interpretability through:

1. **Transport Matrix P**: Shows epitope-prototype mappings
2. **Prototype Weights w**: Reveals strategy mixing
3. **Cost Decomposition**: Breaks down decision factors
4. **Crisis Analysis**: Multi-dimensional crisis attribution

## Theoretical Guarantees

### Proven Properties
- Sinkhorn converges in O(1/ε) iterations (Cuturi, 2013)
- Replicator converges to ESS under fixed fitness (Hofbauer & Sigmund, 1998)
- OT solution is unique for entropic regularization

### Open Questions
- Full system convergence proof for combined OT+Replicator
- Optimal α selection theory
- Scalability to N>100 assets

## References

1. Cuturi (2013) "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
2. Hofbauer & Sigmund (1998) "Evolutionary Games and Population Dynamics"
3. Chen et al. (2021) "Randomized Ensembled Double Q-Learning"
4. Kostrikov et al. (2021) "Implicit Q-Learning"

## Contact

For questions or contributions, please open an issue on the repository.