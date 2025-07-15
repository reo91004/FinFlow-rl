# 강화학습 기반 주식 포트폴리오 리스크 관리 프로젝트

(FinFlow: 2025)

---

## 📋 프로젝트 개요

이 프로젝트는 **강화학습(Reinforcement Learning)** 을 이용하여  
주식 포트폴리오의 **수익률을 극대화**하고, 동시에 **리스크를 안정적으로 제어**하는 것을 목표로 한다.

---

## 🛠 프로젝트 폴더 구조

### 전체 구조
```
finflow-rl/
├── Empirical/                  # Empirical Approach 기반 강화학습
│   ├── main.py
│   ├── src/
│   ├── data/
│   ├── requirements.txt
│   └── ...
├── BIPD/                       # Bio-Inspired Portfolio Defense (생체모방 면역 포트폴리오)
│   ├── main.py
│   ├── data/
│   ├── models/
│   ├── results/
│   └── ...
├── QSPM/                       # Quantum-Inspired Superposition Portfolio Management
│   ├── main.py
│   ├── src/
│   ├── data/
│   ├── models/
│   ├── results/
│   └── ...
└── README.md
```

### Empirical Approach (Empirical/)

| 경로                | 설명                                                  |
| :------------------ | :---------------------------------------------------- |
| `/main.py`          | 전체 학습–평가–XAI를 관리하는 메인 스크립트           |
| `/data/`            | 주가 데이터 (Yahoo Finance) 다운로드 및 전처리 캐시   |
| `/models/`          | 시드별로 저장된 학습된 모델 가중치                    |
| `/results/`         | 학습 로그, 평가 결과 그래프, XAI 시각화 저장 디렉토리 |
| `/tmp/`             | 임시 파일 저장 디렉토리                               |
| `/pdf/`             | 관련 논문 및 문서 저장 디렉토리                       |
| `/requirements.txt` | 필요한 Python 패키지 목록                             |

### BIPD - Bio-Inspired Portfolio Defense (BIPD/)

생체모방 면역 시스템을 활용한 포트폴리오 방어 전략

| 경로        | 설명                                    |
| :---------- | :-------------------------------------- |
| `/main.py`  | 면역 시스템 기반 포트폴리오 관리 스크립트 |
| `/data/`    | 시장 데이터 저장소                      |
| `/models/`  | 면역 시스템 모델 가중치                 |
| `/results/` | 백테스트 결과 및 성과 분석              |

**주요 특징:**
- T-세포: 시장 위험 탐지
- B-세포: 위험 유형별 대응 전략
- 기억 세포: 과거 위기 패턴 학습 및 대응
- 적응적 면역 반응: 시장 변화에 따른 동적 포트폴리오 조정

### QSPM - Quantum-Inspired Superposition Portfolio Management (QSPM/)

양자역학에서 영감을 받은 중첩 상태 포트폴리오 관리

| 경로        | 설명                                          |
| :---------- | :-------------------------------------------- |
| `/main.py`  | 양자 중첩 기반 포트폴리오 관리 스크립트       |
| `/src/`     | 양자 알고리즘 핵심 모듈                       |
| `/data/`    | 시장 데이터 저장소                            |
| `/models/`  | 양자 상태 및 전략 가중치                      |
| `/results/` | 백테스트 결과 및 양자 측정 분석               |

**주요 특징:**
- 양자 중첩: 다중 전략 동시 유지 (conservative, balanced, aggressive)
- 양자 측정: 시장 변동성에 따른 전략 선택
- 양자 얽힘: 자산 간 상관관계 기반 동적 조정
- 진폭 학습: 성과 기반 전략 확률 업데이트

---

## ⚙️ 사용 기술 스택

- Python 3.11+
- PyTorch 2.1
- Gymnasium 0.29 (환경 구성)
- scikit-learn (선형 모델, 스케일링)
- yfinance (주가 데이터 수집)
- tqdm, matplotlib, numpy, pandas

> GPU (CUDA) 사용을 기본으로 최적화되어 있다.

---

## 🧠 강화학습 로직 설명

### 1. 환경 (StockPortfolioEnv)

- **관측 공간**:  
  각 종목별 (가격, 지표) 특성 행렬 (Open, High, Low, Close, Volume, MACD, RSI, MA14, MA21, MA100)
- **행동 공간**:
  - 각 종목 + 현금 슬롯 포함
  - Dirichlet 분포에서 샘플링한 비율로 자산 할당
- **보상 함수**:
  - K=5일 누적 선형 수익률
  - `tanh` 클리핑으로 -1~1 범위로 안정화
  - 가중치 변화 L1 패널티 추가 → 과도 매매 억제
- **거래 비용**: 0.05% 고정 수수료 반영
- **현금 관리**:
  - 주식 외에도 포트폴리오에 현금을 보유할 수 있도록 설계
  - 급락장에서 강제 풀투자 방지

---

### 2. PPO 에이전트 (PPO)

- **정책 네트워크 구조**:
  - 종목별 시계열 피처를 개별 LSTM으로 통과
  - 모든 hidden state를 concat 후 MLP에 입력
  - 액터(Actor) → Dirichlet 분포 파라미터 출력
  - 크리틱(Critic) → 상태 가치(State Value) 예측
- **행동 샘플링**:
  - Dirichlet 분포에서 샘플링
  - 행동에 탐색성을 주기 위해 Softplus 활성화 후 스케일링
- **학습 전략**:
  - GAE (Generalized Advantage Estimation) 사용
  - PPO Clip objective 사용 (epsilon_clip=0.2)
  - EMA (Exponential Moving Average) 타깃 모델 적용
- **다중 시드 학습**:
  - 서로 다른 시드로 3개 모델 독립 학습
  - 평가 시 앙상블 평균 정책 사용

---

### 3. 학습 과정

- 에피소드당 최대 길이: 504 step (약 2년)
- 학습 총 에피소드 수: 500회
- PPO 업데이트 주기: 4000 step마다 10 epoch 최적화
- Gradient Clipping: max norm 0.5
- Learning Rate: 3e-5 (Cosine Annealing은 미사용)

---

### 4. 평가 과정

- 개별 모델 평가: 각각 수익률, 샤프비율, 변동성 계산
- 앙상블 모델 평가: 모든 시드 모델들의 행동을 평균 내어 결정
- 성과 지표 계산:
  - Annualized Return
  - Annualized Volatility
  - Sharpe Ratio
  - Max Drawdown
  - Calmar Ratio
- 성과 비교:
  - Buy-and-Hold 전략과 직접 비교
  - 개별 모델 대비 앙상블 수익률·변동성 비교

---

### 5. XAI (설명 가능한 강화학습)

- **Integrated Gradients**:
  - 액터가 아니라 Critic Value Function에 대해 계산
  - baseline은 0 벡터 사용
  - 통합 그래디언트를 통해 각 피처별 기여도 분석
- **선형 회귀 참조 모델**:
  - test set 수익률을 종속변수로 사용
  - 입력 피처(Flattened Asset-Feature)를 통해 회귀
  - 모델 계수로 참조 피처 중요도 생성
- **시각화**:
  - DRL 기반 특성 중요도 vs. 선형 모델 특성 중요도 비교
  - 통합 그래디언트 결과 단독 시각화 (Raw / Normalized 버전)

---

## 사용법

다음 명령어로 실행합니다:

```
python main.py [모드] [옵션]
```

### 모드

- `all`: 훈련, 백테스팅, XAI 분석을 모두 수행
- `train`: 단일 모델 훈련
- `ensemble_train`: 앙상블 모델 훈련
- `backtest`: 단일 모델 백테스팅
- `ensemble_backtest`: 앙상블 모델 백테스팅
- `xai`: 설명 가능한 AI 분석

### XAI 모드에서 모델 경로 지정

XAI 모드에서는 선택적으로 모델 경로를 직접 지정할 수 있습니다:

```
python main.py xai [모델_경로]
```

모델 경로를 지정하지 않으면 기본 경로를 사용합니다.

### 예시

```
# 기본 XAI 분석
python main.py xai

# 지정된 모델로 XAI 분석
python main.py xai ./results/finflow_train_20230415_123045/models

# 특정 모델 파일로 XAI 분석
python main.py xai ./results/finflow_train_20230415_123045/models/best_model.pth
```

---

## 🏁 실행 방법

### Empirical Approach 실행

```bash
# Clone repository
git clone https://github.com/your-id/finflow-rl.git
cd finflow-rl/Empirical

# Install dependencies
pip install -r requirements.txt

# Run training, evaluation, and XAI analysis
python main.py
```

### BIPD (Bio-Inspired Portfolio Defense) 실행

```bash
cd finflow-rl/BIPD

# Install dependencies (if needed)
pip install numpy pandas yfinance scikit-learn

# Run immune portfolio system
python main.py
```

### QSPM (Quantum-Inspired Superposition Portfolio Management) 실행

```bash
cd finflow-rl/QSPM

# Install dependencies (if needed)
pip install numpy pandas yfinance scikit-learn

# Run quantum portfolio system
python main.py
```

### 결과물

#### Empirical Approach
실행 후 `/results/` 폴더에 다음 결과물이 저장된다:
- training.log : 학습 및 평가 로그
- ensemble_performance.png : 앙상블 평가 결과 그래프
- feature_importance_comparison.png : DRL vs 선형모델 특성 중요도 비교
- integrated_gradients.png : 통합 그래디언트 결과 그래프
- models/ : 시드별 최적 모델 체크포인트

#### BIPD
실행 후 콘솔에 백테스트 결과가 출력되며, `/results/` 폴더에 결과가 저장된다:
- 총 수익률, 변동성, 최대 낙폭, 샤프 비율, 칼마 비율
- 면역 시스템 활성화 로그 및 대응 전략 분석

#### QSPM
실행 후 콘솔에 백테스트 결과가 출력되며, `/results/` 폴더에 결과가 저장된다:
- 양자 측정 기반 성과 분석
- 전략별 확률 변화 추적
- 양자 얽힘 효과 분석

---

## 🔍 차별화 포인트

### Empirical Approach
- 단순 수익률이 아니라 변동성 제어까지 직접 고려한 보상 설계
- 실제 투자환경을 반영한 현금 비중 관리, 거래 비용 모델링
- LSTM 기반 정책 네트워크로 종목별 시계열 패턴 포착
- EMA Target Network를 통한 학습 안정화
- 다중 시드 앙상블로 수익률 편차 최소화
- Explainable AI (XAI) 적용해 모델 해석 가능성 확보

### BIPD (Bio-Inspired Portfolio Defense)
- **생체모방 면역 시스템**: 인간 면역 체계를 모방한 위험 탐지 및 대응
- **적응적 학습**: 과거 위기 패턴 학습을 통한 미래 대응 능력 향상
- **다층 방어**: T-세포, B-세포, 기억세포를 통한 체계적 리스크 관리
- **실시간 대응**: 시장 이상 상황 실시간 탐지 및 즉각적 포트폴리오 조정

### QSPM (Quantum-Inspired Superposition Portfolio Management)
- **양자 중첩 상태**: 다중 투자 전략을 동시에 유지하여 유연성 극대화
- **확률적 의사결정**: 시장 변동성에 따른 동적 전략 선택
- **양자 얽힘 효과**: 자산 간 상관관계를 활용한 최적 포트폴리오 구성
- **진폭 적응**: 성과 기반 학습을 통한 전략 확률 지속적 최적화
