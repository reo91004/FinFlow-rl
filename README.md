# FinFlow-RL: Biologically-Inspired Portfolio Defense 2.0

생물학적 면역 시스템에서 영감을 받은 설명 가능한 포트폴리오 관리 시스템

## 📋 목차
- [개요](#개요)
- [주요 특징](#주요-특징)
- [설치](#설치)
- [사용법](#사용법)
- [명령줄 옵션](#명령줄-옵션)
- [아키텍처](#아키텍처)
- [프로젝트 구조](#프로젝트-구조)
- [성능 목표](#성능-목표)
- [문제 해결](#문제-해결)

## 개요

FinFlow-RL은 IQL(Implicit Q-Learning)에서 Distributional SAC(Soft Actor-Critic)로 이어지는 파이프라인을 통해 안정적이고 설명 가능한 포트폴리오 최적화를 수행하는 강화학습 시스템입니다.

### 핵심 파이프라인
1. **오프라인 사전학습**: IQL을 통한 안정적인 가치 함수 학습
2. **온라인 미세조정**: Distributional SAC + CQL 정규화
3. **목적 함수**: Differential Sharpe 최대화 + CVaR 제약

## 주요 특징

- 🧬 **생물학적 메타포**: T-Cell(위기 감지), B-Cell(전략 실행), Memory Cell(경험 재활용)
- 📊 **분포적 강화학습**: Quantile 기반 리스크 인지 의사결정
- 🔍 **XAI 통합**: SHAP 기반 의사결정 설명 + 반사실적 분석
- ⚡ **실시간 모니터링**: 성능 추적 및 안정성 모니터링
- 🎯 **다중 목적 최적화**: Sharpe, CVaR, 회전율 동시 고려

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

# (선택) SHAP 설치 - XAI 기능 활성화
pip install shap
```

## 사용법

### 🚀 빠른 시작 (학습부터 시각화까지)

```bash
# 1단계: 학습 실행 (IQL 사전학습 → SAC 미세조정)
python main.py --mode train \
    --tickers AAPL GOOGL MSFT AMZN NVDA \
    --iql-epochs 100 \
    --sac-episodes 1000 \
    --target-sharpe 1.5

# 학습 완료 후 생성되는 디렉토리 확인
# logs/YYYYMMDD_HHMMSS/models/checkpoint_best.pt

# 2단계: 평가 및 시각화 생성
python main.py --mode evaluate \
    --resume logs/*/models/checkpoint_best.pt

# 결과 확인
# logs/YYYYMMDD_HHMMSS/reports/ 디렉토리에서 시각화 및 리포트 확인
```

### 1. 기본 학습 (전체 파이프라인)

```bash
# IQL 사전학습 → SAC 미세조정
python main.py --mode train \
    --tickers AAPL GOOGL MSFT AMZN NVDA \
    --iql-epochs 100 \
    --sac-episodes 1000 \
    --target-sharpe 1.5
```

**학습 과정에서 생성되는 파일:**
- `logs/{timestamp}/models/`: 체크포인트 파일
- `logs/{timestamp}/metrics.jsonl`: 실시간 학습 메트릭
- `logs/{timestamp}/console.log`: 콘솔 출력 로그
- `logs/{timestamp}/alerts/`: 모니터링 알람 시 자동 생성되는 그래프

### 2. 평가 및 시각화

```bash
# 학습된 모델 평가 + XAI 분석 + 시각화
python main.py --mode evaluate \
    --resume logs/YYYYMMDD_HHMMSS/models/checkpoint_best.pt
```

**평가 모드에서 생성되는 파일:**
- `logs/{timestamp}/reports/metrics.json`: Sharpe, CVaR, MDD 등 성능 지표
- `logs/{timestamp}/reports/decision_card_*.json`: XAI 의사결정 설명
- `logs/{timestamp}/reports/equity_curve.png`: 누적 수익률 곡선
- `logs/{timestamp}/reports/drawdown.png`: 낙폭 분석 그래프
- `logs/{timestamp}/reports/weights.png`: 포트폴리오 가중치 분포

### 3. 데모 모드 (빠른 테스트)

```bash
# 축소된 설정으로 빠른 테스트
python main.py --mode demo
```

### 4. 개별 스크립트 실행

#### IQL 사전학습만
```bash
python scripts/train.py --use-trainer --mode iql \
    --config configs/default.yaml
```

#### SAC 미세조정만
```bash
python scripts/train.py --use-trainer --mode sac \
    --config configs/default.yaml
```

#### 전체 파이프라인 (IQL + SAC)
```bash
python scripts/train.py --use-trainer --mode full \
    --config configs/default.yaml
```

#### IQL 오프라인 데이터 수집 및 학습
```bash
python scripts/pretrain_iql.py \
    --config configs/default.yaml \
    --collect-episodes 1000 \
    --train-steps 50000 \
    --eval-interval 1000
```

#### 평가 스크립트
```bash
python scripts/evaluate.py \
    --checkpoint models/best_model.pt \
    --data data/test_prices.csv \
    --device cuda \
    --seed 42
```

### 5. 고급 사용법

#### 특정 주식으로 학습
```bash
python main.py --mode train \
    --tickers SPY QQQ IWM TLT GLD \
    --data-period 5y \
    --initial-balance 1000000 \
    --transaction-cost 0.001
```

#### GPU/MPS 사용
```bash
# CUDA GPU
python main.py --mode train --device cuda

# Apple Silicon MPS
python main.py --mode train --device mps

# 자동 감지 (기본값)
python main.py --mode train --device auto
```

#### 체크포인트에서 재개
```bash
python main.py --mode train \
    --resume logs/20250910_120000/checkpoint_best.pt
```

#### 데이터 캐시 관리
```bash
# 캐시 강제 재다운로드
python main.py --mode train --no-cache

# 기본: 캐시 사용 (빠른 로딩)
python main.py --mode train
```

## 명령줄 옵션

### main.py 주요 옵션

| 옵션 | 설명 | 기본값 | 예시 |
|------|------|--------|------|
| `--mode` | 실행 모드 | train | train, evaluate, demo |
| `--config` | 설정 파일 경로 | configs/default.yaml | configs/custom.yaml |
| `--tickers` | 주식 심볼 리스트 | config 파일 참조 | AAPL GOOGL MSFT |
| `--data-period` | 데이터 기간 | 2y | 1y, 2y, 5y, 10y, max |
| `--iql-epochs` | IQL 사전학습 에포크 | 100 | 50, 100, 200 |
| `--sac-episodes` | SAC 미세조정 에피소드 | 1000 | 500, 1000, 2000 |
| `--batch-size` | 배치 크기 | 256 | 64, 128, 256, 512 |
| `--lr` | 학습률 | 3e-4 | 1e-4, 3e-4, 1e-3 |
| `--hidden-dim` | 네트워크 은닉층 차원 | 256 | 128, 256, 512 |
| `--num-quantiles` | 분포적 RL 분위수 개수 | 32 | 8, 16, 32, 64 |
| `--memory-capacity` | 메모리 셀 용량 | 50000 | 10000, 50000, 100000 |
| `--initial-balance` | 초기 포트폴리오 잔고 | 1000000 | 100000, 1000000 |
| `--transaction-cost` | 거래 수수료율 | 0.001 | 0.0005, 0.001, 0.002 |
| `--max-weight` | 자산당 최대 가중치 | 0.2 | 0.1, 0.2, 0.3 |
| `--target-sharpe` | 목표 샤프 비율 | 1.5 | 1.0, 1.5, 2.0 |
| `--target-cvar` | 목표 CVaR (5%) | -0.02 | -0.05, -0.02, -0.01 |
| `--device` | 연산 장치 | auto | auto, cuda, mps, cpu |
| `--seed` | 랜덤 시드 | 42 | 임의의 정수 |
| `--verbose` | 상세 출력 모드 | False | --verbose (플래그) |
| `--no-cache` | 데이터 캐시 사용 안함 | False | --no-cache (플래그) |
| `--resume` | 체크포인트 경로 | None | path/to/checkpoint.pt |

### scripts/train.py 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--config` | 설정 파일 경로 | configs/default.yaml |
| `--resume` | 체크포인트 경로 | None |
| `--debug` | 디버그 모드 | False |
| `--mode` | 학습 모드 (full/iql/sac) | full |
| `--use-trainer` | 통합 트레이너 사용 | True |

### scripts/pretrain_iql.py 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--config` | 설정 파일 경로 | configs/default.yaml |
| `--collect-episodes` | 오프라인 데이터 수집 에피소드 | 100 |
| `--train-steps` | IQL 학습 스텝 | 10000 |
| `--eval-interval` | 평가 간격 | 1000 |

### scripts/evaluate.py 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--checkpoint` | 체크포인트 경로 | 필수 |
| `--data` | 테스트 데이터 경로 | 필수 |
| `--config` | 설정 파일 경로 | None |
| `--device` | 연산 장치 | cpu |
| `--seed` | 랜덤 시드 | 42 |

## 아키텍처

### 상태 공간 (43차원)
- **시장 특성**: 12차원
  - 수익률 통계 (3): 최근 수익률, 평균 수익률, 변동성
  - 기술적 지표 (4): RSI, MACD, Bollinger Bands, 거래량
  - 시장 구조 (3): 상관관계, 베타, 최대 낙폭
  - 모멘텀 (2): 단기(5일), 장기(20일)
- **포트폴리오 가중치**: 30차원 (현재 자산별 할당)
- **위기 수준**: 1차원 (T-Cell 출력)

### 행동 공간
- Simplex 위의 포트폴리오 가중치 (합 = 1.0)
- Dirichlet 정책을 통한 유효한 할당 보장

### 보상 함수
```python
reward = portfolio_return  # 기본 수익률
       + sharpe_bonus      # Differential Sharpe 보너스
       - cvar_penalty      # CVaR 위반 페널티
       - turnover_penalty  # 회전율 페널티
       - drawdown_penalty  # 낙폭 페널티
```

### T+1 결제 모델
- t 시점의 행동은 종가에 실행
- t+1의 수익률이 포트폴리오에 적용
- 거래 비용 및 슬리피지 모델링

## 프로젝트 구조

```
FinFlow-rl/
├── src/
│   ├── agents/          # 생물학적 메타포 에이전트
│   │   ├── t_cell.py    # 위기 감지 (Isolation Forest + SHAP)
│   │   ├── b_cell.py    # 전략 실행 (5가지 전문화)
│   │   ├── memory.py    # 경험 재활용 (k-NN)
│   │   └── gating.py    # 전략 선택 (MoE)
│   ├── core/            # 핵심 RL 모듈
│   │   ├── env.py       # 포트폴리오 환경
│   │   ├── iql.py       # Implicit Q-Learning
│   │   ├── sac.py       # Distributional SAC
│   │   ├── networks.py  # 신경망 아키텍처
│   │   ├── replay.py    # 리플레이 버퍼
│   │   ├── objectives.py # 목적 함수
│   │   ├── distributional.py # 분포적 RL
│   │   └── trainer.py   # 통합 학습기
│   ├── data/            # 데이터 처리
│   │   └── loader.py    # 데이터 로더 + 특성 추출기
│   ├── analysis/        # 분석 도구
│   │   ├── metrics.py   # 메트릭 계산 (MetricsCalculator)
│   │   ├── visualization.py  # 시각화
│   │   ├── backtest.py  # 백테스팅 엔진
│   │   ├── monitor.py   # 실시간 모니터링
│   │   ├── explainer.py # XAI 설명 (SHAP)
│   │   └── tuning.py    # 하이퍼파라미터 튜닝 (Optuna)
│   └── utils/           # 유틸리티
│       ├── logger.py    # 세션 기반 로깅
│       ├── seed.py      # 시드 및 디바이스 관리
│       └── optimizer_utils.py  # 최적화 유틸
├── configs/             # 설정 파일
│   └── default.yaml     # 기본 설정
├── scripts/             # 실행 스크립트
│   ├── train.py         # 학습 스크립트
│   ├── evaluate.py      # 평가 스크립트
│   └── pretrain_iql.py  # IQL 사전학습
├── tests/               # 테스트
│   ├── test_env.py      # 환경 테스트
│   └── test_integration.py # 통합 테스트
├── logs/                # 로그 및 체크포인트
│   └── YYYYMMDD_HHMMSS/ # 세션별 디렉토리
│       ├── console.log  # 콘솔 출력
│       ├── debug.log    # 디버그 정보
│       ├── metrics.jsonl # 메트릭 기록
│       ├── models/      # 체크포인트
│       │   ├── checkpoint_best.pt
│       │   └── checkpoint_final.pt
│       ├── reports/     # 평가 결과 (evaluate 모드)
│       │   ├── metrics.json           # 성능 지표
│       │   ├── decision_card_*.json   # XAI 설명
│       │   ├── equity_curve.png       # 누적 수익률
│       │   ├── drawdown.png          # 낙폭 분석
│       │   └── weights.png           # 포트폴리오 가중치
│       └── alerts/      # 모니터링 알람 시각화
│           ├── equity_*.png
│           ├── dd_*.png
│           └── weights_*.png
├── data/                # 데이터 디렉토리
│   └── cache/           # 다운로드 캐시
├── main.py              # 메인 엔트리
├── requirements.txt     # 의존성
└── README.md            # 문서
```

## 성능 목표

| 메트릭 | 목표값 | 설명 |
|--------|--------|------|
| Sharpe Ratio | ≥ 1.5 | 리스크 조정 수익률 |
| CVaR (5%) | ≥ -0.02 | 하방 리스크 제약 |
| 최대 낙폭 | ≤ 25% | 최대 손실 제한 |
| 연간 수익률 | ≥ 15% | 목표 수익률 |
| 승률 | ≥ 55% | 수익 거래 비율 |
| 회전율 | ≤ 50% | 거래 빈도 제한 |

## 학습 결과 및 시각화

### 학습 중 실시간 모니터링
학습 과정에서 자동으로 생성되는 파일:
- **메트릭 추적**: `metrics.jsonl`에 실시간 기록
- **체크포인트**: 최고 성능 모델 자동 저장
- **알람 시각화**: 이상 감지 시 `alerts/` 디렉토리에 그래프 자동 생성

### 평가 후 생성되는 시각화
`python main.py --mode evaluate` 실행 시:

#### 1. 성능 지표 (`reports/metrics.json`)
```json
{
  "sharpe": 1.82,
  "cvar_95": -0.018,
  "max_drawdown": -0.142,
  "total_return": 0.42,
  "annual_return": 0.21,
  "volatility": 0.115
}
```

#### 2. XAI 의사결정 설명 (`reports/decision_card_*.json`)
- **local_attribution**: 각 특징의 기여도 분석
- **counterfactual**: "만약 ~했다면" 시나리오 분석
- **regime_report**: 시장 레짐별 전략 설명

#### 3. 시각화 그래프
- **누적 수익률** (`equity_curve.png`): 포트폴리오 가치 변화
- **낙폭 분석** (`drawdown.png`): 최고점 대비 하락률
- **포트폴리오 구성** (`weights.png`): 자산별 할당 비중

### 출력 디렉토리 구조
```
logs/YYYYMMDD_HHMMSS/
├── models/                    # 학습 중 저장
│   ├── checkpoint_best.pt     # 최고 성능 모델
│   └── checkpoint_final.pt    # 최종 모델
├── reports/                   # 평가 모드에서 생성
│   ├── metrics.json          # 전체 성능 지표
│   ├── decision_card_*.json  # XAI 설명 (각 결정마다)
│   ├── equity_curve.png      # 누적 수익 곡선
│   ├── drawdown.png         # 낙폭 그래프
│   └── weights.png          # 포트폴리오 가중치
└── alerts/                    # 학습 중 알람 발생 시
    ├── equity_{step}.png     # 해당 시점 수익률
    ├── dd_{step}.png        # 해당 시점 낙폭
    └── weights_{step}.png   # 해당 시점 가중치
```

## 테스트

```bash
# 통합 테스트 - 전체 시스템 검증
python test_integration.py

# 유닛 테스트
pytest tests/

# 특정 테스트
pytest tests/test_env.py -v

# 커버리지 측정
pytest tests/ --cov=src --cov-report=html
```

## 로그 및 모니터링

### 실시간 모니터링
학습 중 다음 메트릭을 실시간으로 추적:
- **안정성 모니터링**: Q-value 폭발, 엔트로피 급락, 보상 이상치 감지
- **성능 추적**: Sharpe Ratio, CVaR, 낙폭 실시간 계산
- **자동 개입**: 이상 감지 시 학습률 조정, 체크포인트 저장

### 로그 파일 구조
```
logs/YYYYMMDD_HHMMSS/
├── console.log         # INFO 레벨 콘솔 출력
├── debug.log           # DEBUG 레벨 상세 로그
├── metrics.jsonl       # 에포크별 메트릭 (JSON Lines)
└── session_info.json   # 세션 메타데이터
```

### 메트릭 파일 형식 (`metrics.jsonl`)
```json
{"timestamp": "2025-09-10T12:00:00", "episode": 100, "sharpe": 1.2, "cvar": -0.02, "total_return": 0.15}
{"timestamp": "2025-09-10T12:05:00", "episode": 200, "sharpe": 1.5, "cvar": -0.018, "total_return": 0.22}
```

## 문제 해결

### ImportError: cannot import name 'MetricsCalculator'
```bash
# 이미 해결됨 - MetricsCalculator 클래스가 추가되었습니다
python main.py --mode train
```

### SHAP 경고 메시지
```bash
# SHAP 라이브러리 설치로 XAI 기능 활성화
pip install shap
```

### CUDA 메모리 부족
```bash
# 배치 크기 줄이기
python main.py --mode train --batch-size 64

# 또는 CPU 사용
python main.py --mode train --device cpu
```

### 데이터 다운로드 실패
```bash
# 캐시 삭제 후 재다운로드
rm -rf data/cache/
python main.py --mode train --no-cache
```

### 학습 속도 느림
```bash
# GPU 사용 확인
python main.py --mode train --device cuda --verbose

# 배치 크기 증가
python main.py --mode train --batch-size 512

# 학습률 조정
python main.py --mode train --lr 1e-3
```

## 설정 파일 예시

`configs/default.yaml`:
```yaml
data:
  tickers: ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]
  period: "2y"
  interval: "1d"

training:
  iql_epochs: 100
  sac_episodes: 1000
  batch_size: 256
  lr: 3e-4

model:
  hidden_dim: 256
  num_quantiles: 32

env:
  initial_balance: 1000000
  transaction_cost: 0.001
  max_weight: 0.2

targets:
  sharpe_ratio: 1.5
  cvar: -0.02
  max_drawdown: -0.25
```

## 기여 방법

1. Fork 저장소
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경 사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치 푸시 (`git push origin feature/AmazingFeature`)
5. Pull Request 오픈

## 라이센스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

## 인용

```bibtex
@software{finflow2024,
  title = {FinFlow-RL: Biologically-Inspired Portfolio Defense},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/FinFlow-rl}
}
```

## 문의

- Issue: [GitHub Issues](https://github.com/yourusername/FinFlow-rl/issues)
- Email: your.email@example.com