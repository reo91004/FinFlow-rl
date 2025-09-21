# XAI (Explainable AI) Documentation

FinFlow-RL의 설명 가능한 AI 기능 상세 가이드

## 목차
- [개요](#개요)
- [SHAP 분석](#shap-분석)
- [T-Cell 위기 설명](#t-cell-위기-설명)
- [의사결정 카드](#의사결정-카드)
- [포트폴리오 귀속](#포트폴리오-귀속)
- [반사실적 분석](#반사실적-분석)
- [시장 레짐 분석](#시장-레짐-분석)
- [해석 가능성 대시보드](#해석-가능성-대시보드)

---

## 개요

FinFlow-RL의 XAI 시스템은 다음을 제공한다:

1. **SHAP 기반 설명**: 피처 중요도와 기여도 분석
2. **T-Cell 위기 해석**: 이상치 탐지 원인 설명
3. **의사결정 투명성**: 각 거래 결정의 근거
4. **포트폴리오 귀속**: 수익 원천 분해

### XAI의 중요성

- **신뢰성**: 모델의 결정을 이해하고 신뢰
- **규제 준수**: 금융 규제 요구사항 충족
- **디버깅**: 모델 개선점 발견
- **리스크 관리**: 예상치 못한 행동 감지

---

## SHAP 분석

### SHAP 값 계산

```python
from src.analysis.xai import XAIAnalyzer
import shap

analyzer = XAIAnalyzer(model, config)

# SHAP Explainer 초기화
explainer = shap.DeepExplainer(model, background_data)

# SHAP 값 계산
shap_values = explainer.shap_values(state)
```

### 글로벌 피처 중요도

전체 데이터셋에서 각 피처의 평균적 중요도:

```python
# 글로벌 중요도 계산
global_importance = analyzer.compute_global_importance(test_data)

# 상위 10개 피처
top_features = global_importance.nlargest(10)
print("가장 중요한 피처:")
for feature, importance in top_features.items():
    print(f"  {feature}: {importance:.3f}")
```

**주요 피처 카테고리:**
| 카테고리 | 피처 | 중요도 범위 |
|---------|------|------------|
| 시장 동향 | Returns, Momentum | 0.15-0.25 |
| 기술 지표 | RSI, MACD, Bollinger | 0.10-0.20 |
| 위험 지표 | Volatility, Correlation | 0.10-0.15 |
| 위기 신호 | T-Cell Crisis Level | 0.20-0.30 |
| 포트폴리오 | Current Weights | 0.05-0.10 |

### 로컬 설명

개별 의사결정에 대한 상세 설명:

```python
# 특정 시점의 결정 분석
decision_explanation = analyzer.explain_decision(
    state=current_state,
    action=selected_action,
    timestamp=t
)

# 시각화
shap.waterfall_plot(
    shap.Explanation(
        values=decision_explanation['shap_values'],
        base_values=decision_explanation['base_value'],
        feature_names=feature_names
    )
)
```

### SHAP 시각화

#### 1. Summary Plot
```python
# 전체 피처의 영향력 요약
shap.summary_plot(shap_values, features, feature_names)
```

#### 2. Dependence Plot
```python
# 특정 피처와 다른 피처 간 상호작용
shap.dependence_plot("RSI", shap_values, features)
```

#### 3. Force Plot
```python
# 개별 예측의 힘 다이어그램
shap.force_plot(explainer.expected_value, shap_values[0], features[0])
```

---

## T-Cell 위기 설명

### 위기 감지 메커니즘

T-Cell은 Isolation Forest를 사용하여 비정상 시장 상황을 감지:

```python
from src.agents.t_cell import TCell

tcell = TCell(config)

# 위기 감지 및 설명
crisis_level, explanation = tcell.detect_and_explain(market_features)

print(f"위기 수준: {crisis_level:.2f}")
print("위기 요인:")
for factor, contribution in explanation['factors'].items():
    print(f"  {factor}: {contribution:.3f}")
```

### 위기 수준 해석

| 위기 수준 | 의미 | 권장 조치 |
|----------|------|----------|
| 0.0-0.3 | 정상 | 일반 전략 유지 |
| 0.3-0.5 | 주의 | 리스크 모니터링 강화 |
| 0.5-0.7 | 경고 | 방어적 포지셔닝 |
| 0.7-1.0 | 위기 | 리스크 최소화 |

### 이상치 원인 분석

```python
# 이상치 스코어 분해
anomaly_breakdown = tcell.explain_anomaly(features)

# 주요 이상 패턴
patterns = {
    'volatility_spike': features['volatility'] > threshold,
    'correlation_breakdown': features['correlation'] < -0.5,
    'volume_anomaly': features['volume'] > 3 * avg_volume,
    'price_gap': abs(features['return']) > 0.05
}

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for (pattern, detected), ax in zip(patterns.items(), axes.flat):
    ax.bar(pattern, detected, color='red' if detected else 'green')
    ax.set_title(f"{pattern}: {'Detected' if detected else 'Normal'}")
```

---

## 의사결정 카드

### 의사결정 카드 생성

각 거래 결정에 대한 상세 설명 카드:

```python
# 의사결정 카드 생성
decision_card = analyzer.create_decision_card(
    state=state,
    action=action,
    reward=reward,
    timestamp=timestamp
)

# 카드 내용
{
    "timestamp": "2025-01-22 10:30:00",
    "decision": {
        "action": "Reduce AAPL from 15% to 10%",
        "confidence": 0.85
    },
    "rationale": {
        "primary_factors": [
            "High RSI (75) indicating overbought",
            "Increased volatility (25% annualized)",
            "T-Cell crisis level elevated (0.45)"
        ],
        "risk_assessment": "Medium-High",
        "expected_impact": "+0.2% Sharpe improvement"
    },
    "alternatives": [
        {"action": "Hold", "expected_value": 0.012},
        {"action": "Increase", "expected_value": -0.005}
    ]
}
```

### 의사결정 템플릿

```python
template = """
📊 Decision Card #{card_id}
📅 Date: {date}
⏰ Time: {time}

🎯 Action Taken:
{action_description}

📈 Key Factors:
1. {factor1} (Impact: {impact1:.2%})
2. {factor2} (Impact: {impact2:.2%})
3. {factor3} (Impact: {impact3:.2%})

⚖️ Risk-Reward:
- Expected Return: {expected_return:.2%}
- Risk Level: {risk_level}
- Confidence: {confidence:.0%}

🔄 Alternative Actions:
{alternatives}

📝 Notes:
{additional_notes}
"""
```

---

## 포트폴리오 귀속

### 수익 원천 분해

```python
from src.analysis.attribution import PortfolioAttributor

attributor = PortfolioAttributor()

# 수익 귀속 분석
attribution = attributor.analyze(
    returns=portfolio_returns,
    weights=portfolio_weights,
    benchmark=benchmark_returns
)

# 결과
{
    "total_return": 0.15,
    "attribution": {
        "asset_selection": 0.08,    # 종목 선택
        "timing": 0.04,              # 타이밍
        "interaction": 0.03          # 상호작용
    },
    "by_asset": {
        "AAPL": 0.05,
        "MSFT": 0.03,
        "GOOGL": 0.04,
        ...
    }
}
```

### Brinson 귀속 모델

```python
# Brinson-Fachler 귀속
def brinson_attribution(portfolio, benchmark):
    # 선택 효과 (Selection Effect)
    selection = sum(
        benchmark_weight * (portfolio_return - benchmark_return)
        for each asset
    )

    # 배분 효과 (Allocation Effect)
    allocation = sum(
        (portfolio_weight - benchmark_weight) * benchmark_return
        for each asset
    )

    # 상호작용 효과 (Interaction Effect)
    interaction = sum(
        (portfolio_weight - benchmark_weight) *
        (portfolio_return - benchmark_return)
        for each asset
    )

    return {
        'selection': selection,
        'allocation': allocation,
        'interaction': interaction,
        'total': selection + allocation + interaction
    }
```

### 리스크 기여도

```python
# 각 자산의 리스크 기여도
risk_contribution = attributor.risk_attribution(
    weights=current_weights,
    covariance_matrix=cov_matrix
)

# 시각화
plt.pie(risk_contribution.values(),
        labels=risk_contribution.keys(),
        autopct='%1.1f%%')
plt.title('Portfolio Risk Contribution')
```

---

## 반사실적 분석

### "What-If" 시나리오

```python
# 반사실적 시나리오 생성
counterfactuals = analyzer.generate_counterfactuals(
    original_state=state,
    n_scenarios=5
)

for i, cf in enumerate(counterfactuals):
    print(f"\n시나리오 {i+1}:")
    print(f"변경사항: {cf['changes']}")
    print(f"예상 행동: {cf['predicted_action']}")
    print(f"예상 수익: {cf['expected_return']:.2%}")
```

### 민감도 분석

```python
# 피처 민감도 분석
sensitivity = analyzer.sensitivity_analysis(
    state=current_state,
    features_to_vary=['volatility', 'rsi', 'momentum']
)

# 3D 시각화
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sensitivity['volatility'],
           sensitivity['rsi'],
           sensitivity['portfolio_return'])
ax.set_xlabel('Volatility')
ax.set_ylabel('RSI')
ax.set_zlabel('Expected Return')
```

---

## 시장 레짐 분석

### 레짐 식별

```python
from src.analysis.regime import RegimeAnalyzer

regime_analyzer = RegimeAnalyzer()

# 현재 시장 레짐 식별
current_regime = regime_analyzer.identify_regime(market_data)

print(f"현재 시장 레짐: {current_regime['name']}")
print(f"특징: {current_regime['characteristics']}")
print(f"권장 전략: {current_regime['recommended_strategy']}")
```

### 레짐별 성과 분석

| 시장 레짐 | 특징 | FinFlow 성과 | 권장 전략 |
|----------|------|-------------|-----------|
| Bull Market | 상승 추세, 낮은 변동성 | Sharpe 1.8 | Growth Expert 활용 |
| Bear Market | 하락 추세, 높은 변동성 | Sharpe 1.2 | Defensive Expert 활용 |
| Sideways | 횡보, 중간 변동성 | Sharpe 1.5 | Correlation Expert 활용 |
| Crisis | 극단적 변동성 | Sharpe 0.9 | T-Cell 주도 방어 |

### 전문가 활성화 패턴

```python
# 각 전문가의 활성화 패턴 분석
expert_activation = analyzer.analyze_expert_activation(
    gating_history=model.gating_history
)

# 시각화
plt.stackplot(dates,
              expert_activation['volatility'],
              expert_activation['correlation'],
              expert_activation['momentum'],
              expert_activation['defensive'],
              expert_activation['growth'],
              labels=['Vol', 'Corr', 'Mom', 'Def', 'Growth'],
              alpha=0.8)
plt.legend(loc='upper right')
plt.title('Expert Activation Over Time')
```

---

## 해석 가능성 대시보드

### 실시간 대시보드

```python
from src.analysis.dashboard import XAIDashboard

dashboard = XAIDashboard(model, config)

# 대시보드 실행
dashboard.run(port=8050)  # http://localhost:8050
```

### 대시보드 구성요소

#### 1. 실시간 모니터링
- 현재 포트폴리오 구성
- 피처 중요도 (실시간 SHAP)
- T-Cell 위기 수준
- 전문가 활성화 상태

#### 2. 의사결정 추적
- 최근 10개 결정 이력
- 각 결정의 설명
- 성과 vs 예측

#### 3. 리스크 대시보드
- VaR/CVaR 실시간
- 리스크 기여도
- 스트레스 테스트 결과

#### 4. 성과 분석
- 누적 수익률
- 롤링 Sharpe
- 벤치마크 대비

### 대시보드 커스터마이징

```python
# 커스텀 위젯 추가
dashboard.add_widget(
    name="Custom Metric",
    update_function=lambda: calculate_custom_metric(),
    visualization_type="gauge",
    update_interval=5000  # 5초마다 업데이트
)

# 알림 설정
dashboard.set_alert(
    condition=lambda: tcell.crisis_level > 0.7,
    message="High crisis level detected!",
    action=send_notification
)
```

---

## 리포트 생성

### XAI 리포트

```python
# 종합 XAI 리포트 생성
report = analyzer.generate_comprehensive_report(
    start_date="2025-01-01",
    end_date="2025-01-22"
)

# HTML 리포트
report.save_html("xai_report.html")

# PDF 리포트
report.save_pdf("xai_report.pdf")
```

### 리포트 내용

1. **Executive Summary**
   - 주요 의사결정 요약
   - 성과 기여 요인

2. **Feature Analysis**
   - 글로벌 피처 중요도
   - 시간별 피처 변화

3. **Decision Log**
   - 모든 주요 결정
   - 각 결정의 근거

4. **Risk Analysis**
   - 위기 감지 이력
   - 리스크 요인 분석

5. **Performance Attribution**
   - 수익 원천
   - 전략별 기여도

---

## 실전 활용

### 1. 디버깅
```python
# 이상한 결정 디버깅
if unexpected_action:
    debug_info = analyzer.debug_decision(state, action)
    print(f"Unexpected factors: {debug_info['anomalies']}")
    print(f"Suggested investigation: {debug_info['suggestions']}")
```

### 2. 모델 개선
```python
# 피처 엔지니어링 힌트
underutilized_features = analyzer.find_underutilized_features()
correlated_features = analyzer.find_correlated_features()

print(f"제거 고려: {underutilized_features}")
print(f"통합 고려: {correlated_features}")
```

### 3. 규제 보고
```python
# 규제 준수 리포트
compliance_report = analyzer.generate_compliance_report(
    regulations="MiFID_II",  # 또는 "SEC", "FSA" 등
    period="quarterly"
)
```

---

## 베스트 프랙티스

1. **정기적 분석**: 주간/월간 XAI 리포트 생성
2. **이상 감지**: 설명할 수 없는 행동 모니터링
3. **피처 관리**: 중요도 기반 피처 선택
4. **투명성**: 주요 이해관계자와 결과 공유
5. **지속적 개선**: XAI 인사이트로 모델 개선

---

## 다음 단계

XAI 분석 후:
1. [CONFIGURATION.md](CONFIGURATION.md) - 인사이트 기반 설정 조정
2. [TRAINING.md](TRAINING.md) - 모델 재학습
3. [API.md](API.md) - 프로그래밍 통합

---

*Last Updated: 2025-01-22*