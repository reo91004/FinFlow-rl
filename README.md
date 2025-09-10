# FinFlow-BIPD v0.2

T-Cell/B-Cell 메타포를 유지하면서 **분포형 SAC + CQL**과 **Sharpe/CVaR 라그랑주**를 결합한 실행 가능한 연구 코드.

## 설치
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 학습/평가
```bash
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
```

## 핵심 아이디어
- T-Cell: IsolationForest로 레짐/위기 점수 z∈[0,1] + SHAP 상위 기여 피처
- B-Cell: Actor(softmax→단체), 두 개의 Quantile Critic, CQL 정규화, α 자동 튜닝
- 목적: Sharpe 최대화 + CVaR(α) >= target 제약(라그랑주 듀얼) + Turnover 벌점
- Gating: z가 높을수록(위기) 더 보수적으로(λ_cvar↑, entropy target↓)

## 산출물
- runs/.../checkpoints/final.pt
- runs/.../figures/equity.png / evaluate_equity.png
- runs/.../summary.json
