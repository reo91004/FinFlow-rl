# Evaluation Guide

FinFlow-RL 평가 파이프라인 및 백테스팅 가이드

## 목차
- [개요](#개요)
- [평가 메트릭](#평가-메트릭)
- [평가 실행](#평가-실행)
- [백테스팅](#백테스팅)
- [벤치마크 비교](#벤치마크-비교)
- [시각화](#시각화)
- [성과 분석](#성과-분석)
- [리포트 생성](#리포트-생성)

---

## 개요

FinFlow-RL 평가 시스템은 다음을 제공한다:

1. **표준 메트릭**: Sharpe, CVaR, MDD 등
2. **현실적 백테스팅**: 거래 비용, 슬리피지, 세금
3. **벤치마크 비교**: 균등가중, Buy&Hold, 60/40
4. **시각화**: 수익률 곡선, 낙폭, 가중치 분포

## 평가 메트릭

### 수익률 메트릭

| 메트릭 | 설명 | 계산식 | 목표값 |
|-------|------|--------|-------|
| **Total Return** | 총 수익률 | `(final - initial) / initial` | > 0.15 |
| **Annual Return** | 연환산 수익률 | `(1 + total_return)^(252/days) - 1` | > 0.15 |
| **Daily Return** | 일일 평균 수익률 | `mean(daily_returns)` | > 0.0006 |

### 위험 조정 메트릭

| 메트릭 | 설명 | 계산식 | 목표값 |
|-------|------|--------|-------|
| **Sharpe Ratio** | 위험 조정 수익률 | `(return - rf) / volatility` | ≥ 1.5 |
| **Sortino Ratio** | 하방 위험 조정 | `(return - rf) / downside_vol` | ≥ 2.0 |
| **Calmar Ratio** | 낙폭 대비 수익률 | `annual_return / max_drawdown` | ≥ 1.0 |

### 위험 메트릭

| 메트릭 | 설명 | 계산식 | 목표값 |
|-------|------|--------|-------|
| **Volatility** | 변동성 | `std(daily_returns) * sqrt(252)` | ≤ 0.15 |
| **Max Drawdown** | 최대 낙폭 | `max((peak - trough) / peak)` | ≤ 0.25 |
| **CVaR (5%)** | 조건부 VaR | `mean(returns < VaR_5%)` | ≥ -0.02 |
| **Downside Deviation** | 하방 변동성 | `std(negative_returns)` | ≤ 0.10 |

### 거래 메트릭

| 메트릭 | 설명 | 계산식 | 목표값 |
|-------|------|--------|-------|
| **Turnover** | 회전율 | `sum(abs(weight_changes))` | ≤ 2.0 |
| **Win Rate** | 승률 | `count(positive) / total` | ≥ 0.55 |
| **Profit Factor** | 손익비 | `sum(gains) / sum(losses)` | ≥ 1.5 |
| **Trade Count** | 거래 횟수 | `count(trades)` | - |

---

## 평가 실행

### 기본 평가

```bash
# main.py 사용
python main.py --mode evaluate \
    --resume logs/*/models/checkpoint_best.pt

# evaluate.py 직접 사용
python scripts/evaluate.py \
    --checkpoint logs/*/models/checkpoint_best.pt \
    --config configs/default.yaml
```

### 현실적 백테스트 포함

```bash
# 거래 비용 모델링 포함
python scripts/evaluate.py \
    --checkpoint logs/*/models/checkpoint_best.pt \
    --with-backtest
```

### 평가 코드 예시

```python
from scripts.evaluate import FinFlowEvaluator

# 평가기 초기화
evaluator = FinFlowEvaluator(
    checkpoint_path="checkpoint.pt",
    config=config
)

# 기본 평가
results = evaluator.evaluate()

# 백테스트 포함 평가
backtest_results = evaluator.evaluate_with_backtest()

print(f"Sharpe Ratio: {results['sharpe']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Total Return: {results['total_return']:.2%}")
```

---

## 백테스팅

### 거래 비용 모델

#### 1. 슬리피지
```python
def calculate_slippage(trade_size, daily_volume, model='square_root'):
    base_slippage = 0.0005  # 0.05%
    volume_impact = trade_size / daily_volume

    if model == 'linear':
        slippage = base_slippage * (1 + volume_impact)
    elif model == 'square_root':
        slippage = base_slippage * (1 + sqrt(volume_impact))
    elif model == 'exponential':
        slippage = base_slippage * exp(volume_impact)

    return slippage
```

#### 2. 시장 충격
```python
def market_impact(trade_value, market_cap):
    temporary_impact = 0.1 * (trade_value / market_cap) ** 0.5
    permanent_impact = 0.05 * (trade_value / market_cap)
    return temporary_impact + permanent_impact
```

#### 3. 거래 수수료
```python
def transaction_costs(trade_value):
    fixed_cost = 5.0  # $5 고정
    proportional_cost = 0.001 * trade_value  # 0.1%
    return max(fixed_cost, proportional_cost)
```

#### 4. 세금 모델
```python
def calculate_tax(realized_gains, holding_period):
    if holding_period > 365:
        # 장기 자본이득세
        tax_rate = 0.15
    else:
        # 단기 자본이득세
        tax_rate = 0.35
    return realized_gains * tax_rate
```

### 백테스트 설정

```yaml
# configs/default.yaml
backtest:
  cost_model:
    fixed_cost: 5.0
    proportional_cost: 0.001

  slippage_model:
    base_slippage: 0.0005
    model_type: "square_root"

  market_impact_model:
    temporary_impact: 0.1
    permanent_impact: 0.05

  constraints:
    max_position_size: 0.3
    max_leverage: 2.0
```

### 백테스트 실행

```python
from src.analysis.backtest import RealisticBacktester

backtester = RealisticBacktester(config['backtest'])

# 전략 함수 정의
def finflow_strategy(data, positions, timestamp):
    state = extract_features(data, timestamp)
    action = model.act(state)
    return action  # 새로운 포트폴리오 가중치

# 백테스트 실행
results = backtester.backtest(
    strategy=finflow_strategy,
    data=price_data,
    initial_capital=1000000,
    verbose=True
)

# 결과 분석
print(f"Net Return: {results['net_return']:.2%}")
print(f"Total Costs: ${results['total_costs']:,.2f}")
print(f"Sharpe (after costs): {results['sharpe_after_costs']:.2f}")
```

---

## 벤치마크 비교

### 지원 벤치마크

1. **Equal Weight**: 균등 가중 포트폴리오
2. **Buy & Hold**: 초기 가중치 유지
3. **60/40**: 주식 60%, 채권 40%
4. **Market Cap**: 시가총액 가중
5. **Risk Parity**: 위험 균등 배분

### 비교 코드

```python
from src.analysis.benchmarks import run_benchmarks

# 벤치마크 실행
benchmarks = run_benchmarks(
    data=price_data,
    benchmarks=['equal_weight', 'buy_hold', '60_40']
)

# 비교 테이블
comparison = pd.DataFrame({
    'FinFlow': results,
    'Equal Weight': benchmarks['equal_weight'],
    'Buy & Hold': benchmarks['buy_hold'],
    '60/40': benchmarks['60_40']
})

print(comparison[['sharpe', 'total_return', 'max_drawdown']])
```

### 상대 성과 메트릭

```python
# 정보 비율 (Information Ratio)
tracking_error = std(portfolio_returns - benchmark_returns)
information_ratio = (portfolio_return - benchmark_return) / tracking_error

# 알파와 베타
alpha, beta = calculate_alpha_beta(portfolio_returns, market_returns)

# 승률 (Win Rate)
outperformance_days = (portfolio_returns > benchmark_returns).mean()
```

---

## 시각화

### 자동 생성 그래프

평가 시 자동으로 생성되는 시각화:

#### 1. 누적 수익률 곡선
```python
# logs/*/reports/equity_curve.png
plt.figure(figsize=(12, 6))
plt.plot(dates, portfolio_values, label='FinFlow', linewidth=2)
plt.plot(dates, benchmark_values, label='Benchmark', alpha=0.7)
plt.fill_between(dates, portfolio_values, benchmark_values,
                  where=(portfolio_values >= benchmark_values),
                  color='green', alpha=0.3)
plt.legend()
plt.title('Portfolio Performance')
```

#### 2. 낙폭 분석
```python
# logs/*/reports/drawdown.png
drawdowns = calculate_drawdowns(portfolio_values)
plt.fill_between(dates, 0, drawdowns, color='red', alpha=0.3)
plt.axhline(y=max_drawdown, color='darkred', linestyle='--')
plt.title(f'Drawdown Analysis (Max: {max_drawdown:.2%})')
```

#### 3. 포트폴리오 구성
```python
# logs/*/reports/weights.png
plt.stackplot(dates, *weights.T, labels=asset_names, alpha=0.8)
plt.legend(loc='upper right', ncol=3)
plt.title('Portfolio Allocation Over Time')
```

#### 4. 위험-수익 산점도
```python
# logs/*/reports/risk_return.png
plt.scatter(volatilities, returns, s=100)
plt.scatter(our_vol, our_return, color='red', s=200, marker='*')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Risk-Return Profile')
```

### 커스텀 시각화

```python
from src.analysis.visualization import VisualizationToolkit

viz = VisualizationToolkit()

# 히트맵
viz.correlation_heatmap(returns)

# 롤링 메트릭
viz.rolling_metrics(returns, window=60)

# 분포 분석
viz.return_distribution(returns)

# 시장 레짐
viz.regime_analysis(returns, volatilities)
```

---

## 성과 분석

### 기간별 분석

```python
def period_analysis(returns, dates):
    # 월별 성과
    monthly = returns.resample('M').apply(lambda x: (1+x).prod()-1)

    # 연도별 성과
    yearly = returns.resample('Y').apply(lambda x: (1+x).prod()-1)

    # 시장 상황별
    bull_market = returns[volatility < median_vol]
    bear_market = returns[volatility >= median_vol]

    return {
        'monthly_avg': monthly.mean(),
        'yearly_avg': yearly.mean(),
        'bull_sharpe': calculate_sharpe(bull_market),
        'bear_sharpe': calculate_sharpe(bear_market)
    }
```

### 스트레스 테스트

```python
# 역사적 위기 시뮬레이션
crisis_periods = {
    '2008 Financial Crisis': ('2008-09-01', '2009-03-31'),
    '2020 COVID Crash': ('2020-02-20', '2020-03-23'),
    '2022 Rate Hike': ('2022-01-01', '2022-06-30')
}

for crisis_name, (start, end) in crisis_periods.items():
    crisis_returns = returns[start:end]
    crisis_performance = evaluate_period(crisis_returns)
    print(f"{crisis_name}: {crisis_performance['total_return']:.2%}")
```

### 팩터 분석

```python
from src.analysis.factor_analysis import FactorAnalyzer

analyzer = FactorAnalyzer()

# Fama-French 3-Factor
factors = analyzer.fama_french_attribution(returns)
print(f"Market Beta: {factors['market_beta']:.2f}")
print(f"Size Factor: {factors['smb']:.2f}")
print(f"Value Factor: {factors['hml']:.2f}")

# 리스크 팩터 분해
risk_decomposition = analyzer.risk_decomposition(returns)
```

---

## 리포트 생성

### 자동 리포트

```python
from src.analysis.reporting import ReportGenerator

generator = ReportGenerator()

# HTML 리포트
generator.create_html_report(
    results=evaluation_results,
    output_path="reports/evaluation_report.html"
)

# PDF 리포트
generator.create_pdf_report(
    results=evaluation_results,
    output_path="reports/evaluation_report.pdf"
)

# JSON 메트릭
generator.save_metrics_json(
    results=evaluation_results,
    output_path="reports/metrics.json"
)
```

### 리포트 구성

생성되는 리포트 구조:

```
📊 Executive Summary
   - Key Metrics Dashboard
   - Performance vs Benchmark
   - Risk Profile

📈 Performance Analysis
   - Cumulative Returns
   - Period Analysis
   - Drawdown Analysis

💰 Transaction Analysis
   - Turnover Statistics
   - Cost Breakdown
   - Trade Distribution

🎯 Risk Analysis
   - VaR and CVaR
   - Stress Test Results
   - Factor Exposures

🔍 Portfolio Composition
   - Weight Evolution
   - Concentration Analysis
   - Correlation Matrix
```

### 커스텀 리포트

```python
# 맞춤형 리포트 템플릿
template = """
# Portfolio Evaluation Report
Generated: {date}

## Performance Summary
- Sharpe Ratio: {sharpe:.2f}
- Total Return: {total_return:.2%}
- Max Drawdown: {max_drawdown:.2%}

## Risk Metrics
- CVaR (5%): {cvar:.2%}
- Volatility: {volatility:.2%}

## Trading Statistics
- Turnover: {turnover:.2f}
- Win Rate: {win_rate:.2%}
"""

report = template.format(**results)
with open("custom_report.md", "w") as f:
    f.write(report)
```

---

## 실전 팁

### 1. 평가 체크리스트
- [ ] 충분한 테스트 기간 (최소 1년)
- [ ] 다양한 시장 상황 포함
- [ ] 거래 비용 반영
- [ ] 여러 벤치마크와 비교
- [ ] 스트레스 테스트 수행

### 2. 주의사항
- 과적합 검증: In-sample vs Out-of-sample
- 생존 편향: Delisted 종목 포함
- 전방 편향: 미래 정보 사용 금지

### 3. 성능 개선
```python
# 여러 시드로 평가
results = []
for seed in range(10):
    set_seed(seed)
    result = evaluate_model(model, data)
    results.append(result)

# 평균과 표준편차
mean_sharpe = np.mean([r['sharpe'] for r in results])
std_sharpe = np.std([r['sharpe'] for r in results])
print(f"Sharpe: {mean_sharpe:.2f} ± {std_sharpe:.2f}")
```

---

## 다음 단계

평가 완료 후:
1. [XAI.md](XAI.md) - 의사결정 설명 분석
2. [CONFIGURATION.md](CONFIGURATION.md) - 파라미터 최적화
3. [API.md](API.md) - 프로그래밍 인터페이스

---

*Last Updated: 2025-01-22*