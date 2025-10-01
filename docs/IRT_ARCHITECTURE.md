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
  emb_dim: 128       # Embedding dimension
  m_tokens: 6        # Number of epitope tokens
  M_proto: 8         # Number of prototypes
  alpha: 0.3         # OT-Replicator mixing ratio
  eps: 0.05          # Sinkhorn entropy
  gamma: 0.5         # Co-stimulation weight
  lambda_tol: 2.0    # Tolerance weight
  rho: 0.3           # Checkpoint weight
```

### Ablation Studies
- `α=0`: Pure Replicator (temporal consistency)
- `α=0.3`: Balanced (default)
- `α=1`: Pure OT (structural matching)

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