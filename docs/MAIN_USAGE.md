# main.py 사용법

## 개요
FinFlow-RL의 메인 엔트리포인트로 학습(train), 평가(evaluate), 데모(demo) 모드를 지원한다.

## 기본 사용법

### 1. 학습 모드 (기본)
```bash
# 기본 설정으로 학습
python main.py

# 특정 설정 파일로 학습
python main.py --config configs/default.yaml

# 1 에피소드 테스트
python main.py --config configs/test_1episode_td3bc.yaml

# GPU 사용
python main.py --device cuda

# CPU 명시
python main.py --device cpu
```

### 2. 평가 모드
```bash
# 체크포인트에서 평가
python main.py --mode evaluate --resume checkpoints/episode_100.pt

# 특정 설정으로 평가
python main.py --mode evaluate --config configs/test_1episode_td3bc.yaml --resume checkpoints/episode_100.pt
```

### 3. 데모 모드
```bash
# 인터랙티브 데모
python main.py --mode demo --resume checkpoints/episode_100.pt
```

## 주요 옵션

### 실행 모드
- `--mode {train,evaluate,demo}`: 실행 모드 선택 (기본: train)

### 설정 관련
- `--config CONFIG`: 설정 파일 경로 (기본: configs/default.yaml)
- `--checkpoint-dir DIR`: 체크포인트 저장 디렉토리
- `--resume CHECKPOINT`: 체크포인트에서 재개

### 데이터 파라미터
- `--tickers TICKER1 TICKER2 ...`: 사용할 주식 심볼 (설정 파일 오버라이드)
- `--data-period PERIOD`: 데이터 기간 (1y, 2y, 5y 등)
- `--no-cache`: 데이터 강제 재다운로드

### 학습 파라미터
- `--iql-epochs N`: IQL 사전학습 에포크 수 (설정 오버라이드)
- `--sac-episodes N`: SAC 미세조정 에피소드 수 (설정 오버라이드)
- `--batch-size N`: 배치 크기 (설정 오버라이드)
- `--lr RATE`: 학습률 (설정 오버라이드)

### 모델 파라미터
- `--hidden-dim N`: 네트워크 은닉층 차원 (설정 오버라이드)
- `--num-quantiles N`: Distributional RL 분위수 개수 (설정 오버라이드)
- `--memory-capacity N`: Memory Cell 용량 (설정 오버라이드)

### 환경 파라미터
- `--initial-balance AMOUNT`: 초기 포트폴리오 잔액 (설정 오버라이드)
- `--transaction-cost RATE`: 거래 비용률 (설정 오버라이드)
- `--max-weight WEIGHT`: 자산당 최대 비중 (설정 오버라이드)

### 목표 메트릭
- `--target-sharpe RATIO`: 목표 샤프 비율 (설정 오버라이드)
- `--target-cvar VALUE`: 목표 CVaR (5% 수준) (설정 오버라이드)

### 시스템 파라미터
- `--device {auto,cuda,cpu}`: 디바이스 선택 (기본: auto)
- `--seed N`: 랜덤 시드 (기본: 42)
- `--verbose`: 상세 로깅 활성화

## 실행 예시

### 예시 1: 빠른 테스트
```bash
# TD3+BC로 1 에피소드 테스트
python main.py --config configs/test_1episode_td3bc.yaml

# IQL로 1 에피소드 테스트
python main.py --config configs/test_1episode_iql.yaml
```

### 예시 2: 커스텀 종목으로 학습
```bash
# FAANG 종목으로 학습
python main.py --tickers AAPL MSFT GOOGL AMZN META --data-period 2y

# 다우존스 30 종목으로 학습
python main.py --config configs/dow30.yaml
```

### 예시 3: 하이퍼파라미터 조정
```bash
# 학습률과 배치 크기 조정
python main.py --lr 0.0001 --batch-size 512

# IQL 에포크와 SAC 에피소드 조정
python main.py --iql-epochs 100 --sac-episodes 500
```

### 예시 4: 평가 및 백테스팅
```bash
# 학습된 모델 평가
python main.py --mode evaluate --resume logs/20250923_170000/checkpoints/episode_100.pt

# 다른 데이터로 평가
python main.py --mode evaluate --resume checkpoints/best_model.pt --tickers SPY QQQ DIA
```

## 출력 구조

모든 출력은 타임스탬프가 있는 세션 디렉토리에 저장된다:

```
logs/
└── YYYYMMDD_HHMMSS/           # 세션 디렉토리
    ├── finflow_training.log    # 상세 로그
    ├── metrics.jsonl           # 학습 메트릭
    ├── checkpoints/            # 모델 체크포인트
    │   ├── episode_10.pt
    │   ├── t_cell_10.pkl
    │   └── memory_10.pkl
    ├── results/                # 최종 결과
    │   └── final_results.json
    ├── tensorboard/            # TensorBoard 로그
    └── visualizations/         # 시각화 결과
        ├── equity_curve.png
        ├── portfolio_weights.png
        └── drawdown.png
```

## 설정 파일 우선순위

1. 명령줄 인자 (최우선)
2. 설정 파일 (--config)
3. 기본값

예:
```bash
# configs/default.yaml에 batch_size: 256이 있어도
# 명령줄 인자가 우선한다
python main.py --config configs/default.yaml --batch-size 512
```

## 주의사항

1. **디바이스 설정**: `--device auto`는 CUDA가 있으면 GPU, 없으면 CPU 사용
2. **캐싱**: 데이터는 기본적으로 `data/cache/`에 캐시됨
3. **로깅**: 모든 로그는 세션별 디렉토리에 저장됨
4. **평가 모드**: 반드시 `--resume` 체크포인트 필요
5. **메모리**: 대규모 데이터셋 사용 시 충분한 RAM 필요

## 문제 해결

### 메모리 부족
```bash
# 배치 크기 줄이기
python main.py --batch-size 128

# CPU 사용
python main.py --device cpu
```

### 데이터 다운로드 실패
```bash
# 캐시 무시하고 재다운로드
python main.py --no-cache

# 다른 기간 시도
python main.py --data-period 1y
```

### 학습 불안정
```bash
# 학습률 낮추기
python main.py --lr 0.00001

# 시드 변경
python main.py --seed 123
```