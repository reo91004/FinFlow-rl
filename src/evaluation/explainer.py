# src/analysis/explainer.py

"""
XAI 설명기: 포트폴리오 의사결정 설명 가능 AI

목적: B-Cell 정책의 투자 결정 해석 및 설명 생성
의존: SHAP, logger.py
사용처: FinFlowTrainer (선택적 XAI 활성화 시)
역할: 투자 의사결정 투명성 제공

구현 내용:
- 3가지 전략: SHAP/Integrated Gradients/LIME
- 특성 중요도 분석 (어떤 시장 지표가 결정에 영향?)
- 반사실 분석 (어떤 조건이면 다른 결정?)
- 위기별 의사결정 패턴 분석
- 포트폴리오 가중치 변화 설명
- 리스크 평가 근거 제시
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime
from src.utils.logger import FinFlowLogger

@dataclass
class DecisionReport:
    """의사결정 리포트"""
    timestamp: str
    state: Dict[str, float]
    action: np.ndarray
    expected_return: float
    risk_assessment: Dict[str, float]
    feature_importance: Dict[str, float]
    counterfactuals: List[Dict]
    confidence: float
    reasoning: str

class XAIExplainer:
    """
    신뢰성 있는 XAI 모듈 - 3가지 전략
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None, memory_cell: Optional[Any] = None):
        self.model = model
        self.memory_cell = memory_cell
        self.logger = FinFlowLogger("XAIExplainer")

        if feature_names is None:
            self.feature_names = self._generate_feature_names()
        else:
            self.feature_names = feature_names

        self.explainer = None
        self.background_data = None
        self.proxy_model = None  # 근사 모델
        self.baseline = None  # Integrated Gradients용
        self._explainer_type = None  # 전략 타입

        # 반사실 분석용 파라미터
        self.counterfactual_n_samples = 10
        self.counterfactual_step_size = 0.1

        self.logger.info("XAI Explainer 초기화 완료")
    
    def _generate_feature_names(self) -> List[str]:
        """기본 특성 이름 생성"""
        names = []
        
        # Market features (dynamic dimensions from config)
        names.extend([
            'recent_return', 'avg_return', 'volatility',
            'rsi', 'macd', 'bb_position', 'volume_ratio',
            'correlation', 'market_beta', 'max_drawdown',
            'short_momentum', 'long_momentum'
        ])
        
        # Portfolio weights (30)
        names.extend([f'weight_{i}' for i in range(30)])
        
        # Crisis level (1)
        names.append('crisis_level')
        
        return names
    
    def initialize_shap(self, background_data: np.ndarray):
        """
        SHAP explainer 초기화 - 신뢰성 있는 방법 선택
        """
        self.background_data = background_data

        # 자동으로 최적 전략 선택
        strategy = self._select_best_strategy()

        if strategy == "gradient":
            self._init_gradient_explainer(background_data)
        elif strategy == "proxy":
            self._init_proxy_explainer(background_data)
        else:  # integrated
            self._init_integrated_gradients(background_data)

    def _select_best_strategy(self) -> str:
        """모델 구조를 분석하여 최적 전략 선택"""
        # 모델이 BatchNorm, Dropout 등을 포함하는지 확인
        has_problematic_layers = self._check_problematic_layers()

        if has_problematic_layers:
            self.logger.info("문제가 될 수 있는 레이어 감지, Integrated Gradients 사용")
            return "integrated"
        else:
            self.logger.info("GradientExplainer 사용 (가장 정확)")
            return "gradient"

    def _check_problematic_layers(self) -> bool:
        """SHAP와 충돌할 수 있는 레이어 확인"""
        if hasattr(self.model, 'actor'):
            model_to_check = self.model.actor
        else:
            model_to_check = self.model

        problematic_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout,
                           nn.LayerNorm, nn.GroupNorm)

        for module in model_to_check.modules():
            if isinstance(module, problematic_types):
                return True
        return False

    def _init_gradient_explainer(self, background_data: np.ndarray):
        """GradientExplainer - DeepExplainer보다 안정적"""
        # 배경 데이터 샘플링 (계산 효율성)
        n_background = min(100, len(background_data))
        indices = np.random.choice(len(background_data), n_background, replace=False)
        background_subset = background_data[indices]

        if hasattr(self.model, 'actor'):
            target_model = self.model.actor
        else:
            target_model = self.model

        # GradientExplainer는 더 안정적
        self.explainer = shap.GradientExplainer(
            target_model,
            torch.FloatTensor(background_subset)
        )

        self.logger.info(f"GradientExplainer 초기화 완료 (samples: {n_background})")
        self._explainer_type = "gradient"

    def _init_integrated_gradients(self, background_data: np.ndarray):
        """Integrated Gradients - 이론적으로 가장 견고"""
        self.logger.info("Integrated Gradients 방식 사용")

        # 기준점 설정 (보통 0 또는 평균)
        self.baseline = np.mean(background_data, axis=0)
        self._explainer_type = "integrated_gradients"
        self.explainer = None  # IG는 직접 구현

    def _init_proxy_explainer(self, background_data: np.ndarray):
        """복잡한 모델을 단순한 모델로 근사 (사용하지 않음 - 연구용 코드)"""
        # 연구용 코드에서는 proxy 모델 사용하지 않음
        # 대신 Integrated Gradients 사용
        self.logger.info("Proxy 모델 대신 Integrated Gradients 사용")
        self._init_integrated_gradients(background_data)
    
    def _model_predict(self, X: np.ndarray) -> np.ndarray:
        """모델 예측 래퍼"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            if hasattr(self.model, 'actor'):
                # SAC 모델
                actions, _ = self.model.actor(X_tensor, deterministic=True)
                return actions.numpy()
            else:
                return self.model(X_tensor).numpy()
    
    def explain_decision(self, 
                         state: np.ndarray,
                         action: np.ndarray,
                         model_output: Optional[Dict] = None) -> DecisionReport:
        """
        의사결정 설명 생성
        
        Args:
            state: 현재 상태
            action: 선택된 액션
            model_output: 모델 출력 (Q값, 정책 등)
            
        Returns:
            DecisionReport
        """
        timestamp = datetime.now().isoformat()
        
        # SHAP 값 계산
        feature_importance = self._compute_feature_importance(state)
        
        # 반사실적 분석
        counterfactuals = self._generate_counterfactuals(state, action)
        
        # 리스크 평가
        risk_assessment = self._assess_risk(state, action, model_output)
        
        # 기대 수익
        expected_return = self._estimate_expected_return(state, action, model_output)
        
        # 신뢰도 계산
        confidence = self._compute_confidence(state, action, model_output)
        
        # 추론 설명 생성
        reasoning = self._generate_reasoning(
            state, action, feature_importance, risk_assessment
        )
        
        # 상태 딕셔너리 생성
        state_dict = {name: float(state[i]) 
                     for i, name in enumerate(self.feature_names[:len(state)])}
        
        return DecisionReport(
            timestamp=timestamp,
            state=state_dict,
            action=action,
            expected_return=expected_return,
            risk_assessment=risk_assessment,
            feature_importance=feature_importance,
            counterfactuals=counterfactuals,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _compute_feature_importance(self, state: np.ndarray) -> Dict[str, float]:
        """특성 중요도 계산"""
        if self.explainer is None:
            # Fallback: 간단한 그래디언트 기반 중요도
            importance = {}
            for i, name in enumerate(self.feature_names[:len(state)]):
                importance[name] = abs(state[i]) / (np.sum(np.abs(state)) + 1e-8)
            return importance
        
        # SHAP 값 계산
        shap_values = self.explainer.shap_values(state.reshape(1, -1))
        
        if isinstance(shap_values, list):
            # 다중 출력의 경우 첫 번째 출력 사용
            shap_values = shap_values[0]
        
        shap_values = shap_values.squeeze()
        
        # 특성별 중요도
        importance = {}
        for i, name in enumerate(self.feature_names[:len(shap_values)]):
            importance[name] = float(abs(shap_values[i]))
        
        # 정규화
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        # 상위 10개만 반환
        sorted_importance = dict(sorted(
            importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])
        
        return sorted_importance
    
    def _generate_counterfactuals(self, 
                                  state: np.ndarray, 
                                  action: np.ndarray,
                                  n_samples: int = 5) -> List[Dict]:
        """반사실적 시나리오 생성"""
        counterfactuals = []
        
        # 주요 특성 변경 시나리오
        important_features = ['volatility', 'crisis_level', 'recent_return']
        
        for feature in important_features:
            if feature in self.feature_names:
                idx = self.feature_names.index(feature)
                
                # 특성 값 변경
                cf_state = state.copy()
                original_value = state[idx]
                
                # 20% 증가/감소 시나리오
                for change in [-0.2, 0.2]:
                    cf_state[idx] = original_value * (1 + change)
                    
                    # 변경된 상태에서 액션 예측
                    with torch.no_grad():
                        cf_action = self._model_predict(cf_state.reshape(1, -1)).squeeze()
                    
                    # 액션 변화 계산
                    action_change = np.linalg.norm(cf_action - action)
                    
                    counterfactuals.append({
                        'feature': feature,
                        'original_value': float(original_value),
                        'new_value': float(cf_state[idx]),
                        'change_percent': float(change * 100),
                        'action_change': float(action_change),
                        'new_action': cf_action.tolist()
                    })
                
                # 원래 값으로 복원
                cf_state[idx] = original_value
        
        return counterfactuals[:n_samples]
    
    def _assess_risk(self, 
                    state: np.ndarray, 
                    action: np.ndarray,
                    model_output: Optional[Dict]) -> Dict[str, float]:
        """리스크 평가"""
        risk_metrics = {}
        
        # 포트폴리오 집중도 (HHI)
        hhi = np.sum(action ** 2)
        risk_metrics['concentration_risk'] = float(hhi)
        
        # 변동성 리스크
        volatility_idx = self.feature_names.index('volatility') if 'volatility' in self.feature_names else 2
        risk_metrics['volatility_risk'] = float(state[volatility_idx])
        
        # 위기 수준
        crisis_idx = self.feature_names.index('crisis_level') if 'crisis_level' in self.feature_names else -1
        risk_metrics['crisis_level'] = float(state[crisis_idx])
        
        # 최대 낙폭 리스크
        dd_idx = self.feature_names.index('max_drawdown') if 'max_drawdown' in self.feature_names else 9
        risk_metrics['drawdown_risk'] = abs(float(state[dd_idx]))
        
        # 전체 리스크 점수 (0-1)
        risk_score = np.mean([
            hhi,
            min(1.0, risk_metrics['volatility_risk'] / 0.3),
            risk_metrics['crisis_level'],
            min(1.0, risk_metrics['drawdown_risk'] / 0.3)
        ])
        risk_metrics['overall_risk'] = float(risk_score)
        
        return risk_metrics
    
    def _estimate_expected_return(self, 
                                  state: np.ndarray,
                                  action: np.ndarray,
                                  model_output: Optional[Dict]) -> float:
        """기대 수익 추정"""
        # 모델 출력에서 Q값 사용
        if model_output and 'q_value' in model_output:
            return float(model_output['q_value'])
        
        # 간단한 휴리스틱: 최근 수익률과 포트폴리오 가중치 기반
        recent_return_idx = 0
        recent_return = state[recent_return_idx]
        
        # 액션(포트폴리오 가중치)과 최근 수익률 결합
        expected_return = float(recent_return * np.sum(action))
        
        return expected_return
    
    def _compute_confidence(self,
                           state: np.ndarray,
                           action: np.ndarray,
                           model_output: Optional[Dict]) -> float:
        """신뢰도 계산"""
        # 정책 엔트로피 기반 신뢰도
        if model_output and 'entropy' in model_output:
            # 낮은 엔트로피 = 높은 신뢰도
            entropy = model_output['entropy']
            confidence = float(np.exp(-entropy))
        else:
            # 포트폴리오 집중도 기반
            # 높은 집중도 = 높은 신뢰도
            max_weight = np.max(action)
            confidence = float(max_weight)
        
        return min(1.0, max(0.0, confidence))
    
    def _generate_reasoning(self,
                           state: np.ndarray,
                           action: np.ndarray,
                           feature_importance: Dict[str, float],
                           risk_assessment: Dict[str, float]) -> str:
        """추론 설명 생성"""
        reasoning_parts = []
        
        # 주요 특성 설명
        top_features = list(feature_importance.keys())[:3]
        if top_features:
            reasoning_parts.append(
                f"주요 고려 요인: {', '.join(top_features)}"
            )
        
        # 리스크 수준
        overall_risk = risk_assessment.get('overall_risk', 0)
        if overall_risk > 0.7:
            reasoning_parts.append("높은 리스크 감지 - 보수적 전략 채택")
        elif overall_risk < 0.3:
            reasoning_parts.append("낮은 리스크 환경 - 적극적 전략 가능")
        else:
            reasoning_parts.append("중간 리스크 수준 - 균형잡힌 접근")
        
        # 포트폴리오 특징
        max_weight_idx = np.argmax(action)
        max_weight = action[max_weight_idx]
        if max_weight > 0.5:
            reasoning_parts.append(f"자산 {max_weight_idx}에 집중 투자 ({max_weight:.1%})")
        else:
            n_assets = np.sum(action > 0.05)
            reasoning_parts.append(f"{n_assets}개 자산에 분산 투자")
        
        return " | ".join(reasoning_parts)
    
    def generate_report(self, 
                       decisions: List[DecisionReport],
                       format: str = 'json') -> str:
        """
        리포트 생성
        
        Args:
            decisions: 의사결정 리포트 리스트
            format: 출력 형식 ('json', 'html')
            
        Returns:
            리포트 문자열
        """
        if format == 'json':
            return self._generate_json_report(decisions)
        elif format == 'html':
            return self._generate_html_report(decisions)
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
    
    def _generate_json_report(self, decisions: List[DecisionReport]) -> str:
        """JSON 리포트 생성"""
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'num_decisions': len(decisions),
            'decisions': [asdict(d) for d in decisions[-10:]]  # 최근 10개
        }
        
        # 요약 통계
        if decisions:
            confidences = [d.confidence for d in decisions]
            returns = [d.expected_return for d in decisions]
            risks = [d.risk_assessment['overall_risk'] for d in decisions]
            
            report_data['summary'] = {
                'avg_confidence': float(np.mean(confidences)),
                'avg_expected_return': float(np.mean(returns)),
                'avg_risk': float(np.mean(risks)),
                'total_decisions': len(decisions)
            }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_html_report(self, decisions: List[DecisionReport]) -> str:
        """HTML 리포트 생성"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>XAI Decision Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .decision { border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
                .metric { display: inline-block; margin: 5px 10px; }
                .high-risk { color: red; }
                .low-risk { color: green; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            </style>
        </head>
        <body>
            <h1>XAI Decision Report</h1>
            <p>Generated: {timestamp}</p>
            <h2>Recent Decisions</h2>
        """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        for decision in decisions[-5:]:  # 최근 5개
            risk_class = "high-risk" if decision.risk_assessment['overall_risk'] > 0.7 else "low-risk"
            
            html += f"""
            <div class="decision">
                <h3>Decision at {decision.timestamp}</h3>
                <div class="metric">Expected Return: {decision.expected_return:.2%}</div>
                <div class="metric {risk_class}">Risk: {decision.risk_assessment['overall_risk']:.2%}</div>
                <div class="metric">Confidence: {decision.confidence:.2%}</div>
                <p><strong>Reasoning:</strong> {decision.reasoning}</p>
                
                <h4>Top Features</h4>
                <table>
                    <tr><th>Feature</th><th>Importance</th></tr>
            """
            
            for feature, importance in list(decision.feature_importance.items())[:5]:
                html += f"<tr><td>{feature}</td><td>{importance:.2%}</td></tr>"
            
            html += """
                </table>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def local_attribution(self, state: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        """
        신뢰성 있는 특성 기여도 분석
        """
        if state.ndim > 1:
            state = state.squeeze()

        assert state.shape[-1] == len(self.feature_names), f"State dimension mismatch: {state.shape[-1]} vs {len(self.feature_names)}"

        if self._explainer_type == "integrated_gradients":
            shap_values = self._integrated_gradients_attribution(state)
        elif self._explainer_type == "gradient":
            shap_values = self._gradient_explainer_attribution(state)
        else:
            # Fallback
            shap_values = self._simple_gradient_attribution(state)

        # 텐서/배열 처리
        if isinstance(shap_values, list):
            # 다중 출력의 경우 첫 번째 사용
            shap_values = shap_values[0]

        if hasattr(shap_values, 'cpu'):
            shap_values = shap_values.cpu().numpy()

        shap_values = shap_values.squeeze()

        # NaN 체크
        assert not np.any(np.isnan(shap_values)), "NaN in SHAP values"

        # 특성별 기여도 매핑
        attribution = {}
        for i, feature_name in enumerate(self.feature_names):
            attribution[feature_name] = float(shap_values[i])

        # 정규화 (합이 1이 되도록)
        total_abs = np.sum(np.abs(list(attribution.values())))
        if total_abs > 0:
            for key in attribution:
                attribution[key] /= total_abs

        # 중요도 순으로 정렬
        sorted_attribution = dict(sorted(
            attribution.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))

        self.logger.debug(f"Top 5 attributions: {list(sorted_attribution.items())[:5]}")

        return sorted_attribution

    def _integrated_gradients_attribution(self, state: np.ndarray) -> np.ndarray:
        """Integrated Gradients를 사용한 attribution 계산"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        baseline_tensor = torch.FloatTensor(self.baseline).unsqueeze(0)

        # Device 확인
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = torch.device('cpu')

        state_tensor = state_tensor.to(device)
        baseline_tensor = baseline_tensor.to(device)

        # 적분 스텝 수
        n_steps = 50
        alphas = np.linspace(0, 1, n_steps)

        # 그래디언트 누적
        integrated_grads = torch.zeros_like(state_tensor)

        for alpha in alphas:
            # 보간된 입력
            interpolated = baseline_tensor + alpha * (state_tensor - baseline_tensor)
            interpolated.requires_grad = True

            # Forward pass
            if hasattr(self.model, 'actor'):
                output = self.model.actor(interpolated)
                if isinstance(output, tuple):
                    output = output[0]  # mean action
            else:
                output = self.model(interpolated)

            # 액션 선택 (최대값 또는 평균)
            action_value = output.mean()

            # Backward pass
            self.model.zero_grad()
            action_value.backward()

            # 그래디언트 누적
            integrated_grads += interpolated.grad / n_steps

        # (입력 - 베이스라인) * 통합 그래디언트
        attributions = (state_tensor - baseline_tensor) * integrated_grads

        return attributions.detach().cpu().numpy().squeeze()

    def _gradient_explainer_attribution(self, state: np.ndarray) -> np.ndarray:
        """SHAP GradientExplainer를 사용한 attribution"""
        if self.explainer is None:
            self.logger.error("GradientExplainer가 초기화되지 않음")
            return np.zeros_like(state)

        # SHAP 값 계산
        shap_values = self.explainer.shap_values(np.array([state]))

        # 다중 출력인 경우 평균
        if isinstance(shap_values, list):
            shap_values = np.mean(shap_values, axis=0)

        return shap_values.squeeze()

    def _simple_gradient_attribution(self, state: np.ndarray) -> np.ndarray:
        """단순 그래디언트 기반 attribution (최종 폴백)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Device 확인
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = torch.device('cpu')

        state_tensor = state_tensor.to(device)
        state_tensor.requires_grad = True

        # Forward pass
        if hasattr(self.model, 'actor'):
            output = self.model.actor(state_tensor)
            if isinstance(output, tuple):
                output = output[0]  # mean action
        else:
            output = self.model(state_tensor)

        # 출력의 평균 또는 합계에 대한 그래디언트
        target = output.mean()

        # Backward pass
        self.model.zero_grad()
        target.backward()

        # 그래디언트가 attribution
        gradients = state_tensor.grad.detach().cpu().numpy().squeeze()

        # 입력값과 곱하기 (Gradient * Input)
        attributions = gradients * state

        return attributions
    
    def counterfactual(self, state: np.ndarray, action: np.ndarray, 
                      deltas: Dict[str, float]) -> Dict[str, Any]:
        """
        반사실 분석: "만약 X가 달랐다면?" 시나리오
        
        Args:
            state: 현재 상태
            action: 현재 액션
            deltas: 변경할 특성과 변경량 (예: {'volatility': -0.1, 'crisis_level': 0.2})
            
        Returns:
            반사실 분석 결과
        """
        assert state.shape[-1] == len(self.feature_names), "State dimension mismatch"
        
        original_state = state.copy()
        cf_state = state.copy()
        
        # 델타 적용
        applied_deltas = {}
        for feature_name, delta in deltas.items():
            if feature_name in self.feature_names:
                idx = self.feature_names.index(feature_name)
                original_value = cf_state[idx]
                new_value = original_value + delta
                
                # 물리적 제약 확인
                if feature_name == 'crisis_level':
                    new_value = np.clip(new_value, 0, 1)
                elif 'weight' in feature_name:
                    new_value = np.clip(new_value, 0, 1)
                elif feature_name == 'volatility':
                    new_value = max(0, new_value)
                
                cf_state[idx] = new_value
                applied_deltas[feature_name] = {
                    'original': float(original_value),
                    'new': float(new_value),
                    'delta': float(new_value - original_value)
                }
        
        # 포트폴리오 가중치 정규화 (필요시)
        # 동적 feature 차원 처리
        weight_start_idx = len([f for f in self.feature_names if 'weight' not in f.lower()])
        weight_end_idx = 42    # 30개 가중치
        if weight_start_idx < len(cf_state):
            weights = cf_state[weight_start_idx:weight_end_idx]
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                cf_state[weight_start_idx:weight_end_idx] = weights / weight_sum
        
        # 반사실 액션 예측
        with torch.no_grad():
            if hasattr(self.model, 'actor'):
                cf_state_tensor = torch.FloatTensor(cf_state.reshape(1, -1))
                cf_action, _ = self.model.actor(cf_state_tensor)
                cf_action = cf_action.cpu().numpy().squeeze()
            else:
                cf_action = self._model_predict(cf_state.reshape(1, -1)).squeeze()
        
        # 액션 변화 분석
        action_diff = cf_action - action
        action_change_norm = float(np.linalg.norm(action_diff))
        
        # 주요 변화 자산
        top_changes = []
        for i in np.argsort(np.abs(action_diff))[-5:]:
            top_changes.append({
                'asset': i,
                'original_weight': float(action[i]),
                'new_weight': float(cf_action[i]),
                'change': float(action_diff[i])
            })
        
        # 예상 성과 변화 (간단한 휴리스틱)
        original_expected_return = self._estimate_expected_return(original_state, action, None)
        cf_expected_return = self._estimate_expected_return(cf_state, cf_action, None)
        
        # 리스크 변화
        original_risk = self._assess_risk(original_state, action, None)
        cf_risk = self._assess_risk(cf_state, cf_action, None)
        
        result = {
            'scenario': deltas,
            'applied_changes': applied_deltas,
            'original_action': action.tolist(),
            'counterfactual_action': cf_action.tolist(),
            'action_change_norm': action_change_norm,
            'top_portfolio_changes': top_changes,
            'performance_impact': {
                'original_return': original_expected_return,
                'cf_return': cf_expected_return,
                'return_delta': float(cf_expected_return - original_expected_return)
            },
            'risk_impact': {
                'original_risk': original_risk['overall_risk'],
                'cf_risk': cf_risk['overall_risk'],
                'risk_delta': float(cf_risk['overall_risk'] - original_risk['overall_risk'])
            }
        }
        
        self.logger.debug(f"Counterfactual analysis: return_delta={result['performance_impact']['return_delta']:.4f}, "
                         f"risk_delta={result['risk_impact']['risk_delta']:.4f}")
        
        return result
    
    def regime_report(self, crisis_info: Dict[str, float], 
                     shap_topk: List[Tuple[str, float]],
                     similar_cases: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        레짐 분석 리포트 생성
        
        Args:
            crisis_info: T-Cell의 위기 정보
            shap_topk: SHAP 상위 k개 특성
            similar_cases: 메모리셀에서 검색한 유사 사례
            
        Returns:
            레짐 분석 리포트
        """
        # 레짐 분류
        overall_crisis = crisis_info.get('overall_crisis', 0)
        
        if overall_crisis < 0.3:
            regime = 'normal'
            regime_desc = '정상 시장'
        elif overall_crisis < 0.5:
            regime = 'caution'
            regime_desc = '주의 시장'
        elif overall_crisis < 0.7:
            regime = 'stress'
            regime_desc = '스트레스 시장'
        else:
            regime = 'crisis'
            regime_desc = '위기 시장'
        
        # 위기 요인 분석
        crisis_factors = []
        for key, value in crisis_info.items():
            if 'crisis' in key and value > 0.5:
                factor_name = key.replace('_crisis', '').replace('_', ' ').title()
                crisis_factors.append({
                    'factor': factor_name,
                    'severity': float(value),
                    'status': 'high' if value > 0.7 else 'moderate'
                })
        
        # SHAP 기반 주요 드라이버
        drivers = []
        for feature, importance in shap_topk:
            drivers.append({
                'feature': feature,
                'importance': float(importance),
                'direction': 'positive' if importance > 0 else 'negative'
            })
        
        # 유사 사례 분석
        similar_summary = None
        if similar_cases and len(similar_cases) > 0:
            # 유사 사례들의 성과 통계
            returns = [case.get('return', 0) for case in similar_cases]
            strategies = [case.get('strategy', 'unknown') for case in similar_cases]
            
            # 가장 빈번한 전략
            from collections import Counter
            strategy_counts = Counter(strategies)
            most_common_strategy = strategy_counts.most_common(1)[0] if strategy_counts else ('unknown', 0)
            
            similar_summary = {
                'num_cases': len(similar_cases),
                'avg_return': float(np.mean(returns)) if returns else 0,
                'std_return': float(np.std(returns)) if returns else 0,
                'best_return': float(max(returns)) if returns else 0,
                'worst_return': float(min(returns)) if returns else 0,
                'common_strategy': most_common_strategy[0],
                'strategy_frequency': float(most_common_strategy[1] / len(similar_cases)) if similar_cases else 0
            }
            
            # 상위 3개 사례
            best_cases = sorted(similar_cases, key=lambda x: x.get('return', 0), reverse=True)[:3]
            similar_summary['best_cases'] = [
                {
                    'date': case.get('date', 'unknown'),
                    'return': float(case.get('return', 0)),
                    'strategy': case.get('strategy', 'unknown'),
                    'similarity': float(case.get('similarity', 0))
                }
                for case in best_cases
            ]
        
        # 권장 전략 (레짐 기반)
        recommendations = []
        if regime == 'crisis':
            recommendations = [
                '방어적 자산 비중 확대',
                '현금 비중 증가',
                '변동성 헤지 전략 고려',
                '손절 기준 엄격 적용'
            ]
        elif regime == 'stress':
            recommendations = [
                '포트폴리오 분산 강화',
                '리스크 모니터링 강화',
                '유동성 확보',
                '단기 포지션 선호'
            ]
        elif regime == 'caution':
            recommendations = [
                '선별적 투자',
                '모멘텀 전략 제한적 사용',
                '섹터 로테이션 고려'
            ]
        else:  # normal
            recommendations = [
                '성장주 비중 확대 가능',
                '레버리지 적절히 활용',
                '장기 포지션 구축'
            ]
        
        report = {
            'regime': regime,
            'regime_description': regime_desc,
            'crisis_level': float(overall_crisis),
            'crisis_factors': crisis_factors,
            'key_drivers': drivers[:5],  # 상위 5개
            'similar_regime_analysis': similar_summary,
            'recommendations': recommendations,
            'confidence': float(1.0 - min(0.5, np.std([cf['severity'] for cf in crisis_factors]) if crisis_factors else 0)),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.debug(f"Regime report generated: {regime} (crisis={overall_crisis:.2f})")
        
        return report