# FinFlow-RL: Biologically-Inspired Portfolio Defense 2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

생물학적 면역 시스템에서 영감을 받은 설명 가능한 포트폴리오 관리 시스템

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

FinFlow-RL (BIPD 2.0)은 IQL(Implicit Q-Learning)에서 Distributional SAC(Soft Actor-Critic)로 이어지는 파이프라인을 통해 안정적이고 설명 가능한 포트폴리오 최적화를 수행하는 강화학습 시스템이다.

### 핵심 파이프라인
1. **오프라인 사전학습**: IQL을 통한 안정적인 가치 함수 학습
2. **온라인 미세조정**: B-Cell (Distributional SAC + CQL 정규화)
3. **목적 함수**: Differential Sharpe 최대화 + CVaR 제약

### 최근 업데이트 (v2.0.0)
- ✅ `sac.py` 제거 및 B-Cell에 통합 완료
- ✅ 백테스트 기능 `evaluate.py`에 통합 (`--with-backtest`)
- ✅ 오프라인 데이터 재사용 기능 수정
- ✅ SafeTensors 통합으로 안전한 모델 저장
- 📄 [전체 변경사항](docs/CHANGELOG.md)

## 주요 특징

- 🧬 **생물학적 메타포**: T-Cell(위기 감지), B-Cell(전략 실행), Memory Cell(경험 재활용)
- 📊 **분포적 강화학습**: Quantile 기반 리스크 인지 의사결정
- 🔍 **XAI 통합**: SHAP 기반 의사결정 설명 + 반사실적 분석
- ⚡ **실시간 모니터링**: 성능 추적 및 안정성 모니터링
- 🎯 **다중 목적 최적화**: Sharpe, CVaR, 회전율 동시 고려
- 💰 **현실적 백테스팅**: 거래 비용, 슬리피지, 세금 모델링

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
# 데모 모드: 빠른 테스트용 축소 설정
python main.py --mode demo
```

### 📊 전체 파이프라인 실행

```bash
# 1. 학습 (IQL 사전학습 → B-Cell 미세조정)
python main.py --mode train \
    --tickers AAPL MSFT GOOGL AMZN NVDA \
    --iql-epochs 50 \
    --sac-episodes 500

# 2. 평가 및 시각화
python main.py --mode evaluate \
    --resume logs/*/models/checkpoint_best.pt

# 3. 결과 확인
# logs/YYYYMMDD_HHMMSS/reports/ 에서 시각화 확인
```

## 사용법

### 1. 메인 엔트리포인트 (main.py)

#### 기본 학습
```bash
python main.py --mode train \
    --config configs/default.yaml \
    --tickers AAPL MSFT GOOGL \
    --iql-epochs 100 \
    --sac-episodes 1000
```

#### 평가 모드
```bash
python main.py --mode evaluate \
    --resume logs/20250122_120000/models/checkpoint_best.pt
```

#### 주요 옵션
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode` | train, evaluate, demo | train |
| `--config` | 설정 파일 경로 | configs/default.yaml |
| `--tickers` | 주식 심볼 리스트 | config 파일 참조 |
| `--iql-epochs` | IQL 사전학습 에포크 | 100 |
| `--sac-episodes` | SAC 미세조정 에피소드 | 1000 |
| `--device` | auto, cuda, mps, cpu | auto |

> 📖 전체 옵션은 [docs/CONFIGURATION.md](docs/CONFIGURATION.md) 참조

### 2. 개별 스크립트 실행

#### 통합 학습 (권장)
```bash
# IQL + B-Cell 전체 파이프라인
python scripts/train.py --config configs/default.yaml
```

#### IQL 사전학습만
```bash
python scripts/pretrain_iql.py \
    --config configs/default.yaml \
    --collect-episodes 500 \
    --train-steps 10000
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
├── main.py                 # 메인 엔트리포인트
├── configs/
│   └── default.yaml        # 기본 설정
├── docs/                   # 📚 상세 문서
│   ├── API.md             # API 레퍼런스
│   ├── TRAINING.md        # 학습 가이드
│   ├── EVALUATION.md      # 평가 가이드
│   ├── XAI.md             # XAI 설명
│   ├── CONFIGURATION.md   # 설정 가이드
│   └── CHANGELOG.md       # 변경 이력
├── scripts/
│   ├── train.py           # 통합 학습 스크립트
│   ├── evaluate.py        # 평가 + 백테스트
│   └── pretrain_iql.py    # IQL 사전학습
├── src/
│   ├── agents/            # 강화학습 에이전트
│   │   ├── b_cell.py      # IQL + Distributional SAC
│   │   ├── t_cell.py      # 위기 감지
│   │   ├── memory.py      # 경험 재활용
│   │   └── meta.py        # 메타 학습
│   ├── core/              # 핵심 모듈
│   │   ├── env.py         # 거래 환경
│   │   ├── iql.py         # IQL 구현
│   │   ├── trainer.py     # 학습 파이프라인
│   │   └── networks.py    # 신경망
│   ├── analysis/          # 분석 도구
│   │   ├── xai.py         # SHAP 설명
│   │   ├── backtest.py    # 백테스팅
│   │   └── monitor.py     # 모니터링
│   ├── data/              # 데이터 처리
│   │   ├── loader.py      # yfinance 로더
│   │   └── features.py    # 피처 엔지니어링
│   └── utils/             # 유틸리티
│       └── logger.py      # 로깅 시스템
├── tests/                 # 테스트
├── logs/                  # 실행 로그
└── ARCHITECTURE.md        # 전체 아키텍처 문서
```

## 문서

### 📚 상세 문서
- [API 레퍼런스](docs/API.md) - 주요 클래스와 함수
- [학습 가이드](docs/TRAINING.md) - IQL과 B-Cell 학습 상세
- [평가 가이드](docs/EVALUATION.md) - 백테스팅과 메트릭
- [XAI 문서](docs/XAI.md) - 설명 가능한 AI 기능
- [설정 가이드](docs/CONFIGURATION.md) - 파라미터 튜닝
- [변경 이력](docs/CHANGELOG.md) - 버전별 업데이트
- [아키텍처](ARCHITECTURE.md) - 전체 시스템 구조

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

| 메트릭 | 목표값 | 설명 |
|--------|--------|------|
| Sharpe Ratio | ≥ 1.5 | 리스크 조정 수익률 |
| CVaR (5%) | ≥ -0.02 | 하방 리스크 제약 |
| 최대 낙폭 | ≤ 25% | 최대 손실 제한 |
| 연간 수익률 | ≥ 15% | 목표 수익률 |
| 회전율 | ≤ 200% | 연간 거래 빈도 |

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
@software{finflow2025,
  title = {FinFlow-RL: Biologically-Inspired Portfolio Defense 2.0},
  author = {FinFlow Team},
  year = {2025},
  url = {https://github.com/yourusername/FinFlow-rl}
}
```

## 문의

- Issue: [GitHub Issues](https://github.com/yourusername/FinFlow-rl/issues)
- Email: contact@finflow.ai

---

*Last Updated: 2025-01-22 | Version: 2.0.0 (BIPD)*