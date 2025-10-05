# FinRL-IRT: Crisis-Adaptive Portfolio Management

IRT (Immune Replicator Transport) Operator를 FinRL 환경에서 검증하는 연구 프로젝트.

## Overview

본 프로젝트는 면역학적 메커니즘에서 영감을 받은 IRT Operator를 검증된 강화학습 환경인 FinRL에 통합하여, 시장 위기 상황에서의 포트폴리오 관리 성능을 입증하는 것을 목표로 한다.

### Core Innovation: IRT Operator

```
w_t = (1-α)·Replicator(w_{t-1}, f_t) + α·Transport(E_t, K, C_t)
```

- **Optimal Transport**: 현재 상태와 프로토타입 전략 간 구조적 매칭
- **Replicator Dynamics**: 과거 성공 전략에 대한 시간 메모리
- **Immunological Cost**: 도메인 지식이 내장된 비용 함수

## Features

- ✅ **IRT Operator** - OT + Replicator Dynamics 결합
- ✅ **SAC + Custom Policy** - Stable Baselines3 기반
- ✅ **Crisis Adaptation** - T-Cell 위기 감지 메커니즘
- ✅ **XAI Visualization** - 12개 해석 가능성 플롯
- ✅ **FinRL Integration** - 검증된 환경 활용

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd FinFlow-rl

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

자세한 설치 가이드는 [INSTALL.md](INSTALL.md) 참조.

### 2. Minimal Test

**FinRL 환경 테스트**:

```bash
python tests/test_finrl_minimal.py
```

예상 출력:

```
✅ FinRL 최소 실행 테스트 성공!
Total Portfolio Value: $1,000,234.56
Total Reward: 0.0234
```

**IRT Policy 테스트**:

```bash
python tests/test_irt_policy.py
```

예상 출력:

```
✅ Test 1 passed: IRT forward pass 정상 작동
✅ Test 2 passed: Simplex 제약 만족
✅ Test 3 passed: SB3 통합 성공
✅ Test 4 (CPU) passed: CPU 호환성 확인
✅ Test 5 passed: IRT 분해 공식 검증
✅ All tests passed!
```

### 3. Training

#### SAC Baseline (FinRL Standard)

FinRL 논문과 동일한 조건으로 베이스라인을 학습한다:

```bash
python scripts/train_finrl_standard.py --model sac --mode both
```

출력: `logs/finrl_sac/{timestamp}/sac_50k.zip`

#### SAC Baseline (IRT 비교용)

IRT와 정확히 동일한 조건으로 학습한다:

```bash
python scripts/train.py --model sac --mode both --episodes 200
```

출력: `logs/sac/{timestamp}/sac_final.zip`

#### IRT

```bash
python scripts/train_irt.py --episodes 200
```

출력: `logs/irt/{timestamp}/irt_final.zip`

### 4. Evaluation

```bash
# FinRL Standard Baseline 평가
python scripts/evaluate.py \
  --model logs/finrl_sac/{timestamp}/sac_50k.zip \
  --method drlagent \
  --save-plot --save-json

# IRT vs Baseline 비교 평가
python scripts/evaluate.py \
  --model logs/sac/{timestamp}/sac_final.zip \
  --method direct \
  --save-plot --save-json

python scripts/evaluate.py \
  --model logs/irt/{timestamp}/irt_final.zip \
  --method direct \
  --save-plot --save-json
```

자동으로 생성되는 결과물:

- `evaluation_results.json` - 메트릭 (Sharpe, Calmar, Max Drawdown 등)
- `evaluation_plots/` - 시각화 (Portfolio Value, Drawdown, Returns Distribution)

## Project Structure

```
FinFlow-rl/
├── finrl/                  # FinRL 핵심 라이브러리
│   ├── agents/irt/         # IRT Custom Policy (Phase 1)
│   │   ├── irt_operator.py # IRT Operator (Sinkhorn + Replicator)
│   │   ├── t_cell.py       # TCellMinimal (위기 감지)
│   │   ├── bcell_actor.py  # BCellIRTActor
│   │   └── irt_policy.py   # IRTPolicy (SB3 통합)
│   ├── evaluation/         # 평가 및 시각화
│   │   └── visualizer.py   # 14개 IRT 시각화 플롯
│   ├── config.py           # 하이퍼파라미터 및 설정
│   ├── config_tickers.py   # DOW_30_TICKER 등
│   └── meta/               # 환경, 전처리 등
├── scripts/                # 학습/평가 스크립트
│   ├── train.py            # 일반 RL 알고리즘 (SB3 직접 사용)
│   ├── train_irt.py        # IRT Policy 학습
│   ├── train_finrl_standard.py  # FinRL 표준 (DRLAgent)
│   └── evaluate.py         # 평가 (두 가지 방식 지원)
├── tests/                  # 테스트
│   ├── test_irt_policy.py  # IRT Policy 단위 테스트
│   └── test_finrl_minimal.py  # FinRL 환경 테스트
├── logs/                   # 학습/평가 결과 (타임스탬프)
│   ├── sac/
│   ├── finrl_sac/
│   └── irt/
├── docs/                   # 문서
│   ├── IRT.md              # IRT 알고리즘 설명서
│   ├── SCRIPTS.md          # 스크립트 사용 가이드
│   └── CHANGELOG.md        # 변경사항 이력
└── README.md
```

## Configuration

모든 하이퍼파라미터는 `finrl/config.py`에서 중앙 관리된다.

### SAC Parameters

```python
# finrl/config.py
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
```

### Technical Indicators

```python
# finrl/config.py
INDICATORS = [
    "macd",
    "boll_ub",      # Bollinger Upper Band
    "boll_lb",      # Bollinger Lower Band
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]
```

### Training Period

```python
# finrl/config.py
TRAIN_START_DATE = "2008-04-01"
TRAIN_END_DATE = "2020-12-31"

TEST_START_DATE = "2021-01-01"
TEST_END_DATE = "2024-12-31"
```

자세한 설정은 [finrl/config.py](finrl/config.py) 참조.

## Performance Metrics

| Metric       | SAC Baseline | IRT        | Improvement |
| ------------ | ------------ | ---------- | ----------- |
| Sharpe Ratio | 1.0-1.2      | 1.2-1.4    | +10-15%     |
| Max Drawdown | -30 ~ -35%   | -20 ~ -25% | **-20-30%** |
| Crisis MDD   | -40 ~ -45%   | -25 ~ -30% | **-30-40%** |

**Note**: 위기 구간(2020 COVID, 2022 Fed 금리 인상)에서의 개선이 두드러짐.

## Documentation

- **[docs/IRT.md](docs/IRT.md)** - IRT 알고리즘 상세 설명 (OT, Replicator, 면역학적 비용)
- **[docs/CHANGELOG.md](docs/CHANGELOG.md)** - 변경사항 이력 및 Phase 완료 내역
- **[docs/SCRIPTS.md](docs/SCRIPTS.md)** - 스크립트 상세 사용 가이드
- **[finrl/config.py](finrl/config.py)** - 하이퍼파라미터 및 설정
- [FinRL 공식 문서](https://finrl.readthedocs.io/) - FinRL 라이브러리 참조

## Citation

본 프로젝트를 사용하는 경우 다음을 인용:

```bibtex
@misc{finrl-irt-2025,
  title={FinRL-IRT: Crisis-Adaptive Portfolio Management via Immune Replicator Transport},
  author={Your Name},
  year={2025},
  note={GitHub repository},
  url={<repo-url>}
}
```

FinRL 인용:

```bibtex
@article{liu2024finrl,
  title={FinRL: Financial reinforcement learning framework},
  author={Liu, Xiao-Yang and others},
  journal={NeurIPS Workshop},
  year={2024}
}
```

## License

MIT License - [LICENSE](LICENSE) 파일 참조.

FinRL은 원저자의 라이선스를 따름.

## Contact

- **Issues**: GitHub Issues 사용
- **Discussions**: GitHub Discussions 활용

---

**Status**: Phase 0 완료 ✅ | Phase 1 완료 ✅ | **Phase 1.7 (Gradient Stabilization) 완료 ✅** | Phase 2 준비 중 📋

**Latest**: SAC+IRT gradient stabilization implemented (3-Tier Solution) - See [docs/GRADIENT_STABILIZATION.md](docs/GRADIENT_STABILIZATION.md)
