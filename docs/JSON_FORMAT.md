# JSON 데이터 포맷 가이드

**v2.0.4** (2025-10-04)

평가 (`evaluate_irt.py`) 실행 시 생성되는 JSON 파일들의 구조와 활용 방법을 설명한다.

---

## 생성되는 파일

평가 완료 시 `logs/YYYYMMDD_HHMMSS/evaluation/` 디렉토리에 다음 파일들이 생성된다:

```
evaluation/
├── evaluation_results.json      # Raw 데이터 (시계열, 메트릭, IRT 중간값)
├── evaluation_insights.json     # 해석된 인사이트 (요약, 분석)
└── visualizations/              # 12개 시각화 PNG
    ├── irt_decomposition.png
    ├── returns.png
    ├── ...
    └── cost_matrix.png
```

---

## 1. `evaluation_results.json`

### 목적
시각화와 분석에 필요한 **모든 raw 데이터**를 저장한다. 외부 도구(Jupyter, pandas, 웹 대시보드)에서 읽어서 재분석 가능하다.

### 구조

```json
{
  "metrics": {
    "sharpe_ratio": 1.52,
    "sortino_ratio": 2.15,
    "calmar_ratio": 0.85,
    "max_drawdown": -0.18,
    "cvar_5": -0.032,
    "var_5": -0.025,
    "avg_turnover": 0.12,
    "total_return": 0.234,
    "avg_crisis_level": 0.42,
    "downside_deviation": 0.011
  },

  "returns": [0.002, -0.001, 0.003, ...],  // 일일 수익률 (length: n_steps)

  "weights": [                              // 포트폴리오 가중치 (shape: [n_steps, n_assets])
    [0.03, 0.05, 0.02, ...],
    [0.04, 0.06, 0.01, ...],
    ...
  ],

  "crisis_levels": [0.3, 0.5, 0.8, ...],   // T-Cell 위기 레벨 (length: n_steps)

  "crisis_types": [                         // 위기 유형 분포 (shape: [n_steps, K])
    [0.1, 0.2, 0.05, ...],                  // K = T-Cell crisis embedding dim
    [0.15, 0.25, 0.03, ...],
    ...
  ],

  "prototype_weights": [                    // IRT 프로토타입 가중치 w (shape: [n_steps, M])
    [0.12, 0.08, 0.15, ...],                // M = 프로토타입 수 (default: 8)
    [0.10, 0.09, 0.14, ...],
    ...
  ],

  "w_rep": [                                // Replicator 출력 (shape: [n_steps, M])
    [0.10, 0.09, 0.12, ...],
    [0.11, 0.08, 0.13, ...],
    ...
  ],

  "w_ot": [                                 // OT 출력 (shape: [n_steps, M])
    [0.14, 0.07, 0.18, ...],
    [0.09, 0.10, 0.15, ...],
    ...
  ],

  "cost_matrices": [                        // Immunological cost (shape: [n_samples, m, M])
    [                                       // 10 step마다 샘플링
      [0.5, 0.3, ...],                      // m = epitope 토큰 수 (default: 6)
      [0.4, 0.6, ...],
      ...
    ],
    ...
  ],

  "eta": [0.12, 0.15, 0.18, ...],          // Crisis learning rate (length: n_steps)

  "symbols": ["AAPL", "MSFT", "JPM", ...], // 주식 심볼 (length: n_assets)

  "price_data": [                           // 가격 데이터 (벤치마크용, shape: [n_steps, n_assets])
    [150.2, 280.5, 120.8, ...],
    [151.0, 282.1, 121.2, ...],
    ...
  ],

  "dates": ["2023-01-03", "2023-01-04", ...] // 날짜 (length: n_steps)
}
```

### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `metrics` | dict | 성능 지표 (Sharpe, Sortino, Calmar, CVaR 등) |
| `returns` | list[float] | 일일 포트폴리오 수익률 |
| `weights` | list[list[float]] | 포트폴리오 가중치 시계열 [n_steps, n_assets] |
| `crisis_levels` | list[float] | T-Cell 위기 감지 레벨 (0~1) |
| `crisis_types` | list[list[float]] | 위기 유형 분포 [n_steps, K] |
| `prototype_weights` | list[list[float]] | IRT 프로토타입 가중치 w [n_steps, M] |
| `w_rep` | list[list[float]] | Replicator 출력 (w = (1-α)·w_rep + α·w_ot) |
| `w_ot` | list[list[float]] | OT 출력 |
| `cost_matrices` | list[list[list[float]]] | Immunological cost [n_samples, m, M] |
| `eta` | list[float] | Crisis-adaptive learning rate η(c) |
| `symbols` | list[str] | 주식 심볼 |
| `price_data` | list[list[float]] | 가격 데이터 (벤치마크 계산용) |
| `dates` | list[str] | 날짜 문자열 (YYYY-MM-DD) |

---

## 2. `evaluation_insights.json`

### 목적
그림 없이도 JSON으로 IRT 의사결정을 **해석**할 수 있도록 핵심 인사이트를 추출한다.

### 구조

```json
{
  "summary": {
    "total_return": 0.234,
    "sharpe_ratio": 1.52,
    "sortino_ratio": 2.15,
    "calmar_ratio": 0.85,
    "max_drawdown": -0.18,
    "avg_crisis_level": 0.42,
    "total_steps": 250
  },

  "top_holdings": [
    {
      "symbol": "AAPL",
      "avg_weight": 0.152,
      "contribution": 0.034
    },
    {
      "symbol": "MSFT",
      "avg_weight": 0.128,
      "contribution": 0.028
    },
    ...  // Top 10
  ],

  "crisis_vs_normal": {
    "crisis": {
      "sharpe": 1.2,
      "avg_return": 0.0012,
      "volatility": 0.015,
      "steps": 75
    },
    "normal": {
      "sharpe": 1.8,
      "avg_return": 0.0015,
      "volatility": 0.008,
      "steps": 175
    }
  },

  "irt_decomposition": {
    "avg_w_rep_contribution": 0.68,
    "avg_w_ot_contribution": 0.32,
    "correlation_w_rep_w_ot": -0.12,
    "avg_eta": 0.142,
    "max_eta": 0.180,
    "min_eta": 0.100
  },

  "prototype_analysis": {
    "most_used_prototypes": [0, 3, 7],
    "prototype_avg_weights": [0.12, 0.08, 0.09, ...],
    "avg_entropy": 1.85,
    "max_entropy": 2.07,
    "min_entropy": 1.23
  },

  "risk_metrics": {
    "VaR_5": -0.025,
    "CVaR_5": -0.032,
    "downside_deviation": 0.011,
    "avg_turnover": 0.12
  },

  "tcell_insights": {
    "crisis_regime_pct": 0.30,
    "top_crisis_types": [2, 5, 1],
    "avg_crisis_type_distribution": [0.1, 0.08, 0.15, ...],
    "avg_danger_level": 0.42
  }
}
```

### 필드 설명

| 섹션 | 필드 | 설명 |
|------|------|------|
| `summary` | - | 핵심 성능 지표 요약 |
| `top_holdings` | `symbol`, `avg_weight`, `contribution` | Top 10 보유 자산 |
| `crisis_vs_normal` | `crisis`, `normal` | 위기/정상 국면별 성능 비교 |
| `irt_decomposition` | `avg_w_rep_contribution`, `avg_w_ot_contribution` | IRT 분해: Replicator vs OT 기여도 |
| `prototype_analysis` | `most_used_prototypes`, `avg_entropy` | 프로토타입 활용 패턴 |
| `risk_metrics` | `VaR_5`, `CVaR_5` | 리스크 지표 |
| `tcell_insights` | `crisis_regime_pct`, `top_crisis_types` | T-Cell 위기 감지 분석 |

---

## 3. 활용 예시

### 3.1 Jupyter Notebook에서 분석

```python
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# JSON 로드
with open('logs/20251003_123456/evaluation/evaluation_results.json', 'r') as f:
    results = json.load(f)

# DataFrame 변환
df = pd.DataFrame({
    'date': results['dates'],
    'return': results['returns'],
    'crisis_level': results['crisis_levels']
})

# 위기 국면 필터링
crisis_mask = df['crisis_level'] > 0.5
crisis_returns = df[crisis_mask]['return']
normal_returns = df[~crisis_mask]['return']

print(f"Crisis Sharpe: {crisis_returns.mean() / crisis_returns.std() * np.sqrt(252):.3f}")
print(f"Normal Sharpe: {normal_returns.mean() / normal_returns.std() * np.sqrt(252):.3f}")

# 시각화
plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['crisis_level'], label='Crisis Level')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.show()
```

### 3.2 웹 대시보드 연동

```python
from flask import Flask, jsonify
import json

app = Flask(__name__)

@app.route('/api/evaluation/<session_id>')
def get_evaluation(session_id):
    # JSON 로드
    path = f'logs/{session_id}/evaluation/evaluation_insights.json'
    with open(path, 'r') as f:
        insights = json.load(f)

    return jsonify(insights)

@app.route('/api/timeseries/<session_id>')
def get_timeseries(session_id):
    path = f'logs/{session_id}/evaluation/evaluation_results.json'
    with open(path, 'r') as f:
        results = json.load(f)

    return jsonify({
        'dates': results['dates'],
        'returns': results['returns'],
        'crisis_levels': results['crisis_levels']
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### 3.3 시각화 재생성

```bash
# evaluation_results.json으로부터 12개 시각화 재생성
python scripts/visualize_from_json.py --results logs/20251003_123456/evaluation/evaluation_results.json

# 커스텀 출력 디렉토리 지정
python scripts/visualize_from_json.py --results logs/.../evaluation_results.json --output custom_viz/
```

### 3.4 인사이트 자동 추출

```python
import json

# Insights JSON 로드
with open('logs/20251003_123456/evaluation/evaluation_insights.json', 'r') as f:
    insights = json.load(f)

# 자동 리포트 생성
print(f"### IRT Portfolio Evaluation Report ###")
print(f"\n[Summary]")
print(f"  Total Return: {insights['summary']['total_return']:.2%}")
print(f"  Sharpe Ratio: {insights['summary']['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {insights['summary']['max_drawdown']:.2%}")

print(f"\n[Top Holdings]")
for holding in insights['top_holdings'][:5]:
    print(f"  {holding['symbol']}: {holding['avg_weight']:.2%} (contrib: {holding['contribution']:.4f})")

print(f"\n[Crisis Adaptation]")
crisis = insights['crisis_vs_normal']['crisis']
normal = insights['crisis_vs_normal']['normal']
print(f"  Crisis Sharpe: {crisis['sharpe']:.2f} ({crisis['steps']} steps)")
print(f"  Normal Sharpe: {normal['sharpe']:.2f} ({normal['steps']} steps)")

print(f"\n[IRT Decomposition]")
print(f"  Replicator: {insights['irt_decomposition']['avg_w_rep_contribution']:.2%}")
print(f"  OT:         {insights['irt_decomposition']['avg_w_ot_contribution']:.2%}")
print(f"  Avg η(c):   {insights['irt_decomposition']['avg_eta']:.3f}")
```

**출력 예시:**
```
### IRT Portfolio Evaluation Report ###

[Summary]
  Total Return: 23.40%
  Sharpe Ratio: 1.52
  Max Drawdown: -18.00%

[Top Holdings]
  AAPL: 15.20% (contrib: 0.0340)
  MSFT: 12.80% (contrib: 0.0280)
  JPM: 10.50% (contrib: 0.0220)
  ...

[Crisis Adaptation]
  Crisis Sharpe: 1.20 (75 steps)
  Normal Sharpe: 1.80 (175 steps)

[IRT Decomposition]
  Replicator: 68.00%
  OT:         32.00%
  Avg η(c):   0.142
```

---

## 4. FAQ

### Q1. `evaluation_results.json`과 `evaluation_insights.json`의 차이는?

- **`evaluation_results.json`**: 모든 raw 데이터 (시계열, 배열, 메트릭)
- **`evaluation_insights.json`**: 해석된 인사이트 (요약, Top N, 통계)

**비유**: results는 로그, insights는 리포트.

### Q2. JSON을 읽어서 시각화를 재생성할 수 있나?

네, `visualize_from_json.py` 스크립트를 사용한다:

```bash
python scripts/visualize_from_json.py --results logs/.../evaluation_results.json
```

### Q3. 외부 도구(예: Tableau, Grafana)에 연동 가능한가?

네, JSON 파일을 읽어서 데이터베이스나 API로 제공하면 된다 (예시 3.2 참조).

### Q4. `cost_matrices`는 왜 크기가 다른가?

Cost matrix는 계산 비용 때문에 **10 step마다만 샘플링**된다. 따라서 `len(cost_matrices) ≈ n_steps / 10`.

### Q5. `crisis_types`의 K 차원은 무엇인가?

T-Cell의 crisis embedding 차원이다. 각 차원은 서로 다른 위기 유형(예: 변동성, 극단 손실, 상관관계 급증)을 나타낸다.

---

## 5. 버전 히스토리

| 버전 | 날짜 | 변경사항 |
|------|------|----------|
| v2.0.4 | 2025-10-04 | `evaluation_insights.json` 추가, `visualize_from_json.py` 추가 |
| v2.0.3 | 2025-10-03 | IRT decomposition 필드 추가 (`w_rep`, `w_ot`, `cost_matrices`, `eta`) |
| v2.0.0 | 2025-10-01 | 초기 버전 |

---

## 참고 문서

- [IRT_ARCHITECTURE.md](./IRT_ARCHITECTURE.md) - IRT 아키텍처 상세 설명
- [CLAUDE.md](../CLAUDE.md) - 프로젝트 개요 및 사용법
- [CHANGELOG.md](./CHANGELOG.md) - 버전별 변경사항

---

**Last Updated**: 2025-10-04
**Version**: 2.0.4
