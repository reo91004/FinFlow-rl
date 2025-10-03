# FinFlow-RL: IRT (Immune Replicator Transport) Portfolio Management

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

IRT (Immune Replicator Transport) Operator 기반 위기 적응형 포트폴리오 관리 시스템

## 📋 목차
- [개요](#개요)
- [주요 특징](#주요-특징)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [사용법](#사용법)
- [프로젝트 구조](#프로젝트-구조)
- [문서](#문서)
- [성능 목표](#성능-목표)
- [문제 해결](#문제-해결)

## 개요

FinFlow-RL IRT는 **Optimal Transport**와 **Replicator Dynamics**를 결합한 혁신적인 정책 혼합 연산자를 통해 위기 상황에 적응적으로 대응하는 포트폴리오 관리 시스템이다.

### IRT 핵심 수식
```
w_t = (1-α)·Replicator(w_{t-1}, f_t) + α·Transport(E_t, K, C_t)
```

### 차별점
- **시간 메모리**: w_{t-1}을 통한 과거 성공 전략 기억
- **구조적 매칭**: Optimal Transport로 현재 상태와 전문가 전략 최적 결합
- **면역학적 비용**: 공자극, 내성, 체크포인트를 통한 도메인 지식 내장

### 최근 업데이트 (v2.1.0-IRT, 2025-10-04)
- ✅ **BC Warm-start**: IQL 완전 대체, AWR/Expectile bias 제거
- ✅ **Progressive Exploration**: 3-stage 적응형 탐색 스케줄 추가
- ✅ **Config 기반 설정**: 모든 하드코딩 제거 (Dirichlet, Progressive)
- ✅ **레거시 정리**: IQL 삭제, 코드 간소화

### 이전 업데이트 (v2.0-IRT)
- 🆕 **IRT Operator**: OT + Replicator 기반 새로운 정책 혼합
- 🆕 **경량 T-Cell**: 단일 신경망으로 위기 감지 간소화
- ✅ **코드 간소화**: 파일 수 33% 감소, 코드 라인 31% 감소
- ✅ **해석 가능성 강화**: 수송 행렬, 복제자 가중치 시각화
- 🔧 **실전 작동 보장**: End-to-end 학습 가능

## 주요 특징

- 🧬 **IRT Operator**: Optimal Transport + Replicator Dynamics 결합
- 🎯 **위기 적응**: 위기 시 자동으로 방어적 전략으로 전환
- 📊 **REDQ Critics**: 10개 Q-network 앙상블로 안정적 학습
- 🔍 **해석 가능성**: 수송 행렬 P, 프로토타입 가중치 w 시각화
- ⚡ **경량화**: 기존 대비 코드 31% 감소, 실행 속도 향상
- 💰 **실전 검증**: 다우존스 30종목 백테스팅

## 설치

### 요구사항
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (GPU 사용 시)

### 설치 방법

```bash
# 저장소 클론
git clone https://github.com/yourusername/FinFlow-rl.git
cd FinFlow-rl

# 의존성 설치
pip install -r requirements.txt

# (선택) 개발 도구 설치
pip install -e ".[dev]"
```

## 빠른 시작

### 🚀 3분 데모 (최소 설정)

```bash
# 1. 빠른 IRT 테스트 (1 에피소드, 5종목)
python main.py --mode demo --config configs/default_irt.yaml

# 2. 결과 확인
ls logs/*/results/
cat logs/*/finflow_training.log | tail -5

# 3. 학습된 모델로 평가
python main.py --mode evaluate \
    --resume logs/*/checkpoints/best_model.pth
```

### 📊 전체 IRT 파이프라인 실행

```bash
# 1. IRT 학습 (BC warm-start → IRT 미세조정)
python scripts/train_irt.py --config configs/default_irt.yaml

# 2. 평가 및 시각화 (12 plots 자동 생성)
python scripts/evaluate_irt.py \
    --checkpoint logs/*/checkpoints/best_model.pth

# 3. Ablation studies (BC 기여도 검증)
python scripts/train_irt.py --config configs/experiments/ablation_bc_a1.yaml  # Random init
python scripts/train_irt.py --config configs/experiments/ablation_bc_a2.yaml  # BC only
python scripts/train_irt.py --config configs/experiments/ablation_bc_a3.yaml  # BC + Diversity
```

## 사용법

### 1. 메인 엔트리포인트 (main.py)

#### IRT 학습 모드
```bash
# 1. 기본 IRT 학습
python main.py --config configs/default_irt.yaml

# 2. 위기 구간 집중 학습
python main.py --config configs/experiments/crisis_focus.yaml

# 3. Ablation study (α 파라미터 비교)
python main.py --config configs/experiments/ablation_irt.yaml

# 4. 빠른 데모 (3개 종목, 10 에피소드)
python main.py --mode demo
```

#### 평가 모드
```bash
# 1. 기본 평가
python main.py --mode evaluate \
    --resume logs/20250122_120000/models/checkpoint_best.pt

# 2. 특정 설정으로 평가 (yaml 사용)
python main.py --mode evaluate \
    --config configs/experiments/test_iql_redq.yaml \
    --resume logs/*/checkpoints/best.pt

# 3. 빠른 테스트 후 바로 평가
python main.py --mode train \
    --config configs/experiments/quick_test.yaml
python main.py --mode evaluate \
    --config configs/experiments/quick_test.yaml \
    --resume logs/latest/checkpoints/best.pt

# 4. 다른 데이터로 평가 (일반화 성능 테스트)
python main.py --mode evaluate \
    --config configs/experiments/test_td3bc_tqc.yaml \
    --resume logs/*/checkpoints/best.pt \
    --tickers NVDA AMD TSM  # 다른 종목으로 평가
```

#### 주요 옵션
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode` | train, evaluate, demo | train |
| `--config` | 설정 파일 경로 | configs/default.yaml |
| `--resume` | 체크포인트 경로 | None |
| `--tickers` | 주식 심볼 리스트 | config 파일 참조 |
| `--no-cache` | 데이터 재다운로드 | False |
| `--bc-epochs` | BC warm-start 에포크 | config 파일 |
| `--sac-episodes` | SAC 미세조정 에피소드 | config 파일 |
| `--batch-size` | 배치 크기 | config 파일 |
| `--device` | auto, cuda, mps, cpu | auto |
| `--verbose` | 상세 출력 | False |

> 📖 전체 옵션은 [docs/CONFIGURATION.md](docs/CONFIGURATION.md) 참조

### 2. 개별 스크립트 실행

#### 통합 학습 (권장)
```bash
# BC + IRT 전체 파이프라인
python scripts/train_irt.py --config configs/default_irt.yaml
```

#### BC Warm-start만
```bash
python scripts/validate_offline_data.py --data data/offline_data.npz  # 데이터 검증
# BC는 trainer_irt.py의 pretrain_with_bc() 메소드에서 자동 실행
```

#### 평가 + 백테스팅
```bash
# 현실적 백테스트 포함
python scripts/evaluate.py \
    --checkpoint logs/*/models/checkpoint_best.pt \
    --with-backtest
```

> 📖 학습 상세 가이드: [docs/TRAINING.md](docs/TRAINING.md)
> 📖 평가 상세 가이드: [docs/EVALUATION.md](docs/EVALUATION.md)

### 3. 고급 사용법

#### GPU/MPS 가속
```bash
# CUDA GPU
python main.py --mode train --device cuda

# Apple Silicon
python main.py --mode train --device mps
```

#### 체크포인트 재개
```bash
python main.py --mode train \
    --resume logs/20250122_120000/models/checkpoint_latest.pt
```

#### 하이퍼파라미터 튜닝
```bash
python src/core/tuning.py \
    --config configs/default.yaml \
    --n-trials 100
```

## 프로젝트 구조

```
FinFlow-rl/
├── main.py                     # 메인 엔트리포인트
├── configs/
│   ├── default_irt.yaml        # IRT 기본 설정
│   └── experiments/
│       ├── ablation_irt.yaml   # Ablation study
│       └── crisis_focus.yaml   # 위기 구간 집중
│
├── src/
│   ├── immune/                 # [NEW] IRT 면역 모듈
│   │   ├── __init__.py
│   │   ├── irt.py              # IRT Operator
│   │   └── t_cell.py           # 경량 T-Cell
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   └── bcell_irt.py        # IRT 기반 Actor
│   │
│   ├── algorithms/
│   │   ├── offline/
│   │   │   ├── __init__.py
│   │   │   └── bc_agent.py     # BC Warm-start (v2.1.0+)
│   │   └── critics/
│   │       ├── __init__.py
│   │       └── redq.py         # REDQ 앙상블
│   │
│   ├── environments/           # 변경 없음
│   │   ├── portfolio_env.py
│   │   └── reward_functions.py
│   │
│   ├── data/                   # 변경 없음
│   │   ├── market_loader.py
│   │   ├── feature_extractor.py
│   │   ├── offline_dataset.py
│   │   └── replay_buffer.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── visualizer.py       # IRT 시각화 추가
│   │   └── explainer.py        # IRT 해석 추가
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer_irt.py      # IRT 전용 트레이너
│   │
│   └── utils/                  # 변경 없음
│       ├── logger.py
│       ├── monitoring.py
│       └── training_utils.py
│
├── scripts/
│   ├── train_irt.py            # IRT 학습
│   ├── evaluate_irt.py         # IRT 평가
│   └── visualize_irt.py        # IRT 시각화
│
├── tests/
│   ├── test_irt.py             # IRT 단위 테스트
│   └── test_integration_irt.py # 통합 테스트
│
├── docs/
│   ├── IRT_ARCHITECTURE.md     # IRT 아키텍처
│   ├── HANDOVER.md             # 리팩토링 가이드
│   └── REFACTORING.md          # IRT 이론적 기초
│
└── logs/                        # 실행 로그
```

## 문서

### 📚 상세 문서
- [학습 가이드](docs/TRAINING.md) - 오프라인/온라인 학습, 알고리즘 비교
- [설정 가이드](docs/CONFIGURATION.md) - 파라미터 튜닝 및 문제 해결
- [평가 가이드](docs/EVALUATION.md) - 백테스팅과 메트릭
- [아키텍처](docs/ARCHITECTURE.md) - 시스템 구조, 알고리즘 조합
- [API 레퍼런스](docs/API.md) - 주요 클래스와 함수
- [XAI 문서](docs/XAI.md) - 설명 가능한 AI 기능
- [변경 이력](docs/CHANGELOG.md) - 버전별 업데이트

### 📊 학습 결과

학습 완료 후 생성되는 파일:

```
logs/YYYYMMDD_HHMMSS/
├── models/                    # 체크포인트
│   ├── checkpoint_best.pt     # 최고 성능
│   └── checkpoint_latest.pt   # 최신
├── reports/                   # 평가 결과
│   ├── metrics.json          # 성능 지표
│   ├── equity_curve.png      # 수익률 곡선
│   ├── drawdown.png         # 낙폭 분석
│   └── weights.png          # 포트폴리오 구성
├── metrics.jsonl             # 학습 메트릭
└── console.log               # 실행 로그
```

## 성능 목표

### IRT vs Baseline 개선 목표
| 메트릭 | SAC Baseline | IRT 목표 | 개선율 |
|--------|-------------|----------|--------|
| **전체 Sharpe** | 1.2 | 1.4+ | +17% |
| **위기 MDD** | -35% | -25% | **-29%** |
| **복구 기간** | 45일 | 35일 | -22% |
| **CVaR (5%)** | -3.5% | -2.5% | -29% |

### 절대 성능 목표
| 메트릭 | 목표값 | 설명 |
|--------|--------|------|
| Sharpe Ratio | ≥ 1.5 | 리스크 조정 수익률 |
| 최대 낙폭 | ≤ 25% | 최대 손실 제한 |
| 연간 수익률 | ≥ 15% | 목표 수익률 |
| 회전율 | ≤ 50% | 일일 거래 빈도 |

## 문제 해결

### CUDA 메모리 부족
```bash
# 배치 크기 감소
python main.py --mode train --batch-size 64

# CPU 사용
python main.py --mode train --device cpu
```

### 데이터 다운로드 실패
```bash
# 캐시 삭제 후 재시도
rm -rf data/cache/
python main.py --mode train --no-cache
```

### ImportError 해결
```bash
# 패키지 재설치
pip install -r requirements.txt --upgrade
```

> 더 많은 해결책: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## 테스트

```bash
# 전체 테스트
pytest tests/

# 특정 테스트
pytest tests/test_full_pipeline.py -v

# 커버리지
pytest tests/ --cov=src
```

## 기여

기여를 환영합니다! [CONTRIBUTING.md](CONTRIBUTING.md) 참조

## 라이센스

MIT License - [LICENSE](LICENSE) 파일 참조

## 인용

```bibtex
@software{finflow_irt_2025,
  title = {FinFlow-RL: IRT (Immune Replicator Transport) for Crisis-Adaptive Portfolio Management},
  author = {FinFlow Team},
  year = {2025},
  version = {2.0-IRT},
  url = {https://github.com/yourusername/FinFlow-rl}
}
```

## 문의

- Issue: [GitHub Issues](https://github.com/yourusername/FinFlow-rl/issues)
- Email: contact@finflow.ai

---

*Last Updated: 2025-10-02 | Version: 2.0-IRT*