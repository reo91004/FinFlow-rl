"""
설명 가능한 AI(XAI) 모듈

강화학습 모델의 의사결정을 설명하는 기능을 제공합니다.
특성 중요도 계산, 민감도 분석, 의사결정 요인 시각화 등의 함수를 포함합니다.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shap
from lime import lime_tabular
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
import traceback

from src.constants import FEATURE_NAMES, STOCK_TICKERS, RESULTS_BASE_PATH, DEVICE, INTEGRATED_GRADIENTS_STEPS, XAI_SAMPLE_COUNT
from src.models.running_mean_std import RunningMeanStd
from src.environment.portfolio_env import StockPortfolioEnv
from src.visualization.visualization import plot_feature_importance, plot_feature_correlation

def compute_feature_importance(
    agent, 
    test_data, 
    test_dates, 
    method="shap", 
    n_samples=100, 
    save_dir=None, 
    logger=None,
    use_ema=True
):
    """
    특성 중요도를 계산하고 시각화합니다.
    
    Args:
        agent: 학습된 PPO 에이전트
        test_data: 테스트 데이터 배열 (시간 x 자산 x 피처)
        test_dates: 테스트 날짜 인덱스
        method: 특성 중요도 계산 방법 ("shap" 또는 "permutation")
        n_samples: 분석할 샘플 수
        save_dir: 결과 저장 디렉토리
        logger: 로깅 객체
        use_ema: EMA 모델 사용 여부
        
    Returns:
        dict: 특성 중요도 결과 딕셔너리
    """
    import logging
    if logger is None:
        logger = logging.getLogger("PortfolioRL")
    
    # 저장 디렉토리 생성
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(RESULTS_BASE_PATH, f"{method}_importance_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Feature importance analysis started: method={method}, samples={n_samples}")
    
    # 테스트 환경 생성 및 샘플 수집
    env = StockPortfolioEnv(test_data)
    n_assets = env.n_assets
    n_features = env.n_features
    
    # 샘플 날짜 랜덤 선택
    if len(test_dates) > n_samples:
        sample_indices = np.random.choice(len(test_dates) - 1, size=n_samples, replace=False)
    else:
        sample_indices = np.arange(len(test_dates) - 1)
        n_samples = len(sample_indices)
    
    # 상태, 액션, 날짜 샘플 수집
    states = []
    actions = []
    dates_sampled = []
    
    for i in tqdm(sample_indices, desc="Collecting samples"):
        state, _ = env.reset(start_index=i)
        
        if hasattr(agent, 'select_action'):
            action = agent.select_action(state, use_ema=use_ema)
        else:
            action, _, _ = agent.policy_old.act(state)
        
        states.append(state)
        actions.append(action)
        dates_sampled.append(test_dates[i])
    
    states = np.array(states)
    actions = np.array(actions)
    
    # SHAP 메소드 사용
    if method == "shap":
        try:
            logger.info("SHAP 분석을 위한 샘플 데이터 준비 중...")
            # 상태 샘플을 2D 형태로 변형 (샘플 × 피처)
            states_2d = states.reshape(n_samples, n_assets * n_features)
            
            # 상태가 비어있는지 확인
            if states_2d.size == 0:
                logger.error("SHAP 분석을 위한 상태 데이터가 비어있습니다.")
                return None
                
            logger.info(f"SHAP 분석 입력 형태: {states_2d.shape}")
            
            # 에러 핸들링을 위한 래퍼 함수 정의
            def safe_shapley_model_fn(x):
                try:
                    # 입력 형태 재구성
                    if x.ndim == 1:
                        x = x.reshape(1, -1)
                    
                    # 에이전트의 입력 크기에 맞게 조정
                    reshaped_x = x.reshape(x.shape[0], n_assets, n_features)
                    
                    return shapley_model_fn(agent, reshaped_x, use_ema)
                except Exception as e:
                    logger.error(f"SHAP 예측 중 오류: {e}")
                    # 오류 발생 시 0으로 채운 결과 반환
                    return np.zeros((x.shape[0], n_assets))
            
            # SHAP 분석
            import shap
            
            # KernelExplainer를 사용하여 SHAP 값 계산
            # 백그라운드 데이터 수 줄이기
            n_background = min(10, len(states_2d))
            background_data = states_2d[:n_background]
            
            logger.info(f"SHAP 백그라운드 데이터 크기: {background_data.shape}")
            explainer = shap.KernelExplainer(
                safe_shapley_model_fn, 
                background_data,
                link="identity"
            )
            
            # 랜덤 샘플에 대한 SHAP 값 계산 (작은 샘플로 테스트)
            n_test_samples = min(5, len(states_2d) - n_background)
            test_samples = states_2d[-n_test_samples:]
            
            logger.info(f"SHAP 테스트 샘플 크기: {test_samples.shape}")
            shap_values = explainer.shap_values(test_samples)
            
            # 특성 중요도 계산
            feature_importance = np.zeros(n_features)
            feature_importance_by_asset = np.zeros((n_assets, n_features))
            feature_importance_by_feature = np.zeros(n_features)
            
            # SHAP 값이 리스트인 경우 처리 (각 출력 클래스에 대한 값)
            if isinstance(shap_values, list):
                # 모든 출력에 대한 평균 SHAP 값
                abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                abs_shap = np.abs(shap_values).mean(axis=0)
                
            logger.info(f"SHAP 평균 절댓값 형태: {abs_shap.shape}")
            
            # 출력 차원에 대한 평균 계산 (예상 형태: n_features,) 
            if abs_shap.ndim == 2:
                 abs_shap_avg_outputs = abs_shap.mean(axis=1)
                 logger.info(f"SHAP 출력 평균 후 형태: {abs_shap_avg_outputs.shape}")
            else:
                 abs_shap_avg_outputs = abs_shap
            
            # 특성 중요도 계산 (자산별, 피처별)
            for asset_idx in range(n_assets):
                asset_start = asset_idx * n_features
                asset_end = (asset_idx + 1) * n_features
                
                # 평균화된 SHAP 값 사용
                if asset_end <= len(abs_shap_avg_outputs):  
                    asset_shap = abs_shap_avg_outputs[asset_start:asset_end]
                    # 차원 확인 및 조정 (이제 asset_shap은 1D 형태여야 함)
                    if len(asset_shap) == n_features:
                        feature_importance_by_asset[asset_idx] = asset_shap
                        feature_importance_by_feature += asset_shap
                    else:
                        logger.warning(f"SHAP 값 차원 불일치: 예상 {n_features}, 실제 {len(asset_shap)}. 스킵합니다.")
                else:
                    logger.warning(f"SHAP 값의 인덱스 범위를 벗어남: {asset_start}:{asset_end} > {len(abs_shap_avg_outputs)}")
            
            # 정규화
            feature_importance_by_feature /= max(1, n_assets)  # 0으로 나누기 방지
            feature_importance_sum = feature_importance_by_feature.sum()
            
            if feature_importance_sum > 0:
                feature_importance_by_feature /= feature_importance_sum
            
            # SHAP 요약 플롯
            try:
                plt.figure(figsize=(12, 8))
                
                # 2D 배열로 변환하여 시각화
                feature_names = []
                for i in range(min(n_assets, len(STOCK_TICKERS))):
                    for j in range(min(n_features, len(FEATURE_NAMES))):
                        feature_names.append(f"{STOCK_TICKERS[i]} - {FEATURE_NAMES[j]}")
                
                # 특성 이름 길이 제한
                max_features = min(len(feature_names), abs_shap.shape[0])
                
                shap.summary_plot(
                    shap_values,
                    test_samples,
                    feature_names=feature_names[:max_features],
                    max_display=20,  # 표시할 최대 특성 수 제한
                    show=False
                )
                plt.title(f"SHAP Feature Importance", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"SHAP 요약 플롯 생성 중 오류: {e}")
            
            # 특성별 중요도 시각화
            plot_feature_importance(
                feature_importance_by_feature, 
                FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))], 
                save_path=os.path.join(save_dir, "feature_importance.png")
            )
            
            # 자산-특성 중요도 히트맵
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                feature_importance_by_asset,
                annot=True,
                cmap='viridis',
                xticklabels=FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))],
                yticklabels=STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))],
                fmt=".3f"
            )
            plt.title("Feature Importance by Asset", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "feature_importance_by_asset.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 결과 저장
            result = {
                "feature_importance_by_asset": feature_importance_by_asset,
                "feature_importance_by_feature": feature_importance_by_feature,
                "feature_names": FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))],
                "asset_names": STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))],
                "n_samples": n_samples,
                "sampled_dates": dates_sampled,
            }
            
            # CSV로 저장
            fi_df = pd.DataFrame(
                feature_importance_by_asset,
                index=STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))],
                columns=FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))]
            )
            fi_df.to_csv(os.path.join(save_dir, "feature_importance_by_asset.csv"))
            
            feature_df = pd.DataFrame({
                "Feature": FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))],
                "Importance": feature_importance_by_feature[:min(n_features, len(FEATURE_NAMES))]
            })
            feature_df.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)
            
            logger.info(f"SHAP analysis completed: results saved to {save_dir}")
            return result
            
        except Exception as e:
            logger.error(f"Error during SHAP analysis: {e}")
            logger.error(traceback.format_exc())  # 상세 오류 정보 출력
            return None
    elif method == "permutation":
        # 순열 중요도 계산
        baseline_actions = []
        feature_importance = np.zeros(states.shape[1])
        feature_importance_per_asset = []
        
        # 기준 행동 계산
        for state in states:
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, use_ema=use_ema)
            else:
                action, _, _ = agent.policy_old.act(state)
            baseline_actions.append(action)
        
        baseline_actions = np.array(baseline_actions)
        
        # 각 특성에 대해 순열 중요도 계산
        for feature_idx in tqdm(range(states.shape[1]), desc="Permutation Importance"):
            # 특성 값 셔플링
            permuted_states = states.copy()
            permuted_states[:, feature_idx] = np.random.permutation(permuted_states[:, feature_idx])
            
            permuted_actions = []
            for state in permuted_states:
                if hasattr(agent, 'select_action'):
                    action = agent.select_action(state, use_ema=use_ema)
                else:
                    action, _, _ = agent.policy_old.act(state)
                permuted_actions.append(action)
            
            permuted_actions = np.array(permuted_actions)
            
            # 액션 변화량 계산
            action_diff = np.mean(np.abs(baseline_actions - permuted_actions))
            feature_importance[feature_idx] = action_diff
        
        # 자산별 특성 중요도로 변환 (2D 형태로)
        feature_importance_2d = feature_importance.reshape(n_assets, n_features)
        feature_importance_per_asset = [feature_importance_2d[i] for i in range(n_assets)]
    
    elif method == "lime":
        # LIME 설명기 초기화
        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=states,
            feature_names=[f"Asset {i} - {f}" for i in range(n_assets) for f in FEATURE_NAMES],
            mode="regression"
        )
        
        # 모든 자산에 대한 중요도 합산
        lime_importances = np.zeros(states.shape[1])
        n_lime_samples = min(20, len(states))
        
        # 함수 래핑
        predict_fn = lambda x: lime_model_fn(agent, x, use_ema)
        
        for i in tqdm(range(n_lime_samples), desc="LIME Analysis"):
            # 각 자산별로 설명 생성
            for asset_idx in range(n_assets):
                exp = lime_explainer.explain_instance(
                    states[i], 
                    lambda x: predict_fn(x)[:, asset_idx],
                    num_features=states.shape[1]
                )
                # 중요도 합산
                for feature_idx, importance in exp.as_list():
                    idx = int(feature_idx.split(" - ")[0].split(" ")[1]) * n_features + \
                          FEATURE_NAMES.index(feature_idx.split(" - ")[1])
                    lime_importances[idx] += abs(importance)
        
        # 평균 계산
        feature_importance = lime_importances / (n_lime_samples * n_assets)
        
        # 자산별 특성 중요도로 변환
        feature_importance_2d = feature_importance.reshape(n_assets, n_features)
        feature_importance_per_asset = [feature_importance_2d[i] for i in range(n_assets)]
    
    # 결과 저장
    if feature_importance is not None:
        # 2D 형태로 변환 (자산 x 특성)
        feature_importance_2d = feature_importance.reshape(n_assets, n_features)
        
        # 특성별 중요도 (모든 자산에 대한 평균)
        feature_importance_by_feature = feature_importance_2d.mean(axis=0)
        
        # 자산별 중요도 (모든 특성에 대한 합)
        feature_importance_by_asset = feature_importance_2d.sum(axis=1)
        
        # 정규화 (합이 1이 되도록)
        feature_importance_by_feature = feature_importance_by_feature / feature_importance_by_feature.sum()
        feature_importance_by_asset = feature_importance_by_asset / feature_importance_by_asset.sum()
        
        # 시각화
        plot_feature_importance(
            feature_importance_by_feature, 
            FEATURE_NAMES, 
            save_path=os.path.join(save_dir, "feature_importance_by_feature.png")
        )
        
        plot_feature_importance(
            feature_importance_by_asset, 
            STOCK_TICKERS[:n_assets], 
            save_path=os.path.join(save_dir, "feature_importance_by_asset.png")
        )
        
        # 특성 간 상관관계 분석
        try:
            features_data = states.reshape(-1, n_features)
            plot_feature_correlation(
                features_data, 
                FEATURE_NAMES, 
                save_path=os.path.join(save_dir, "feature_correlation.png")
            )
        except Exception as e:
            logger.warning(f"특성 상관관계 시각화 중 오류: {e}")
        
        # 결과 저장
        result = {
            "method": method,
            "feature_importance": feature_importance,
            "feature_importance_2d": feature_importance_2d,
            "feature_importance_by_feature": feature_importance_by_feature,
            "feature_importance_by_asset": feature_importance_by_asset,
            "feature_importance_per_asset": feature_importance_per_asset,
            "feature_names": FEATURE_NAMES,
            "asset_names": STOCK_TICKERS[:n_assets],
            "n_samples": len(states),
            "sampled_dates": dates_sampled,
        }
        
        # 결과 저장 (CSV)
        feature_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Importance": feature_importance_by_feature
        })
        feature_df.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)
        
        asset_df = pd.DataFrame({
            "Asset": STOCK_TICKERS[:n_assets],
            "Importance": feature_importance_by_asset
        })
        asset_df.to_csv(os.path.join(save_dir, "asset_importance.csv"), index=False)
        
        logger.info(f"XAI 분석 완료: 결과 저장 위치 = {save_dir}")
        return result
    else:
        logger.warning(f"XAI 분석 실패: {method} 방법으로 특성 중요도를 계산할 수 없었습니다.")
        return None

def shapley_model_fn(agent, states, use_ema=True):
    """
    SHAP 계산을 위한 모델 래퍼 함수
    
    Args:
        agent: 평가할 학습된 PPO 에이전트
        states: 상태 배열
        use_ema: EMA 모델 사용 여부
        
    Returns:
        np.ndarray: 행동 배열
    """
    actions = []
    with torch.no_grad():
        for state in states:
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, use_ema=use_ema)
            else:
                action, _, _ = agent.policy_old.act(state)
            actions.append(action)
    return np.array(actions)

def lime_model_fn(agent, states, use_ema=True):
    """
    LIME 계산을 위한 모델 래퍼 함수
    
    Args:
        agent: 평가할 학습된 PPO 에이전트
        states: 상태 배열
        use_ema: EMA 모델 사용 여부
        
    Returns:
        np.ndarray: 행동 배열
    """
    return shapley_model_fn(agent, states, use_ema)

def sensitivity_analysis(
    agent, 
    test_data, 
    test_dates, 
    n_samples=10, 
    perturbation_range=0.1, 
    save_dir=None, 
    logger=None,
    use_ema=True
):
    """
    민감도 분석을 수행합니다.
    
    Args:
        agent: 평가할 학습된 PPO 에이전트
        test_data: 테스트 데이터 배열 (시간 x 자산 x 피처)
        test_dates: 테스트 날짜 인덱스
        n_samples: 분석할 샘플 수
        perturbation_range: 교란 범위 (예: 0.1은 ±10% 교란)
        save_dir: 결과 저장 디렉토리
        logger: 로깅 객체
        use_ema: EMA 모델 사용 여부
        
    Returns:
        dict: 민감도 분석 결과 딕셔너리
    """
    import logging
    if logger is None:
        logger = logging.getLogger("PortfolioRL")
    
    # 저장 디렉토리 생성
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(RESULTS_BASE_PATH, f"sensitivity_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Sensitivity analysis started: samples={n_samples}, perturbation={perturbation_range:.2f}")
    
    # 테스트 환경 생성 및 샘플 수집
    env = StockPortfolioEnv(test_data)
    n_assets = env.n_assets
    n_features = env.n_features
    
    # 샘플 날짜 랜덤 선택
    if len(test_dates) > n_samples:
        sample_indices = np.random.choice(len(test_dates) - 1, size=n_samples, replace=False)
    else:
        sample_indices = np.arange(len(test_dates) - 1)
        n_samples = len(sample_indices)
    
    # 상태, 액션, 날짜 샘플 수집
    states = []
    base_actions = []
    dates_sampled = []
    
    for i in tqdm(sample_indices, desc="Sensitivity analysis"):
        state, _ = env.reset(start_index=i)
        
        if hasattr(agent, 'select_action'):
            action = agent.select_action(state, use_ema=use_ema)
        else:
            action, _, _ = agent.policy_old.act(state)
        
        states.append(state)
        base_actions.append(action)
        dates_sampled.append(test_dates[i])
    
    states = np.array(states)
    base_actions = np.array(base_actions)
    
    # 민감도 분석 수행
    sensitivity = np.zeros((n_assets, n_features))
    n_perturb = 5  # 다른 교란 값으로 테스트할 횟수
    
    for asset_idx in tqdm(range(min(n_assets, len(STOCK_TICKERS))), desc="Asset sensitivity"):
        for feature_idx in range(min(n_features, len(FEATURE_NAMES))):
            # 기준 상태에 대한 교란 적용
            perturbed_actions = []
            
            for perturb_val in np.linspace(-perturbation_range, perturbation_range, n_perturb):
                perturbed_states = states.copy()
                
                # 해당 자산-특성에 대한 교란 적용
                for s_idx in range(len(states)):
                    orig_val = states[s_idx][asset_idx, feature_idx]
                    perturb_amount = orig_val * perturb_val
                    
                    perturbed_state = perturbed_states[s_idx].copy()
                    perturbed_state[asset_idx, feature_idx] += perturb_amount
                    perturbed_states[s_idx] = perturbed_state
                
                # 교란된 상태에 대한 예측
                perturbed_action_batch = []
                for s_idx in range(len(perturbed_states)):
                    current_state = perturbed_states[s_idx]
                    if hasattr(agent, 'select_action'):
                        action = agent.select_action(current_state, use_ema=use_ema)
                    else:
                        action, _, _ = agent.policy_old.act(current_state)
                    perturbed_action_batch.append(action)
                
                perturbed_actions.append(np.array(perturbed_action_batch))
            
            # 액션 변화의 표준편차 계산
            perturbed_actions = np.array(perturbed_actions)  # (n_perturb, n_samples, n_assets)
            action_std = np.std(perturbed_actions, axis=0)   # (n_samples, n_assets)
            mean_std = np.mean(action_std)
            
            sensitivity[asset_idx, feature_idx] = mean_std
    
    # 민감도 점수 정규화
    if np.sum(sensitivity) > 0:
        sensitivity = sensitivity / np.sum(sensitivity)
    
    # 특성별, 자산별 민감도 계산
    asset_sensitivity = np.sum(sensitivity, axis=1)
    feature_sensitivity = np.sum(sensitivity, axis=0)
    
    # 정규화
    asset_sensitivity = asset_sensitivity / np.sum(asset_sensitivity) if np.sum(asset_sensitivity) > 0 else asset_sensitivity
    feature_sensitivity = feature_sensitivity / np.sum(feature_sensitivity) if np.sum(feature_sensitivity) > 0 else feature_sensitivity
    
    # 자산-특성 민감도 히트맵
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        sensitivity, 
        annot=True, 
        cmap='viridis',
        xticklabels=FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))],
        yticklabels=STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))],
        fmt=".3f"
    )
    plt.title("Feature-Asset Sensitivity Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sensitivity_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 자산별 민감도 막대 그래프
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(asset_sensitivity)), asset_sensitivity)
    plt.xticks(range(len(asset_sensitivity)), STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))])
    plt.title("Asset Sensitivity", fontsize=16)
    plt.xlabel("Assets")
    plt.ylabel("Sensitivity Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "asset_sensitivity.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 특성별 민감도 시각화
    plot_feature_importance(
        feature_sensitivity, 
        FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))], 
        save_path=os.path.join(save_dir, "feature_sensitivity.png")
    )
    
    # 결과 저장
    result = {
        "sensitivity": sensitivity,
        "asset_sensitivity": asset_sensitivity,
        "feature_sensitivity": feature_sensitivity,
        "feature_names": FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))],
        "asset_names": STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))],
        "n_samples": len(states),
        "perturbation_range": perturbation_range,
        "sampled_dates": dates_sampled,
    }
    
    # CSV로 저장
    sensitivity_df = pd.DataFrame(sensitivity, 
                                 index=STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))], 
                                 columns=FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))])
    sensitivity_df.to_csv(os.path.join(save_dir, "sensitivity_matrix.csv"))
    
    feature_df = pd.DataFrame({
        "Feature": FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))],
        "Sensitivity": feature_sensitivity[:min(n_features, len(FEATURE_NAMES))]
    })
    feature_df.to_csv(os.path.join(save_dir, "feature_sensitivity.csv"), index=False)
    
    asset_df = pd.DataFrame({
        "Asset": STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))],
        "Sensitivity": asset_sensitivity[:min(n_assets, len(STOCK_TICKERS))]
    })
    asset_df.to_csv(os.path.join(save_dir, "asset_sensitivity.csv"), index=False)
    
    logger.info(f"Sensitivity analysis completed: results saved to {save_dir}")
    return result

def plot_decision_process(
    agent, 
    test_data, 
    test_dates, 
    date_index=None, 
    save_dir=None, 
    logger=None,
    use_ema=True
):
    """
    특정 날짜에 대한 의사결정 과정을 시각화합니다.
    
    Args:
        agent: 평가할 학습된 PPO 에이전트
        test_data: 테스트 데이터 배열 (시간 x 자산 x 피처)
        test_dates: 테스트 날짜 인덱스
        date_index: 날짜 인덱스 (None이면 무작위 선택)
        save_dir: 결과 저장 디렉토리
        logger: 로깅 객체
        use_ema: EMA 모델 사용 여부
        
    Returns:
        dict: 의사결정 분석 결과 딕셔너리
    """
    import logging
    if logger is None:
        logger = logging.getLogger("PortfolioRL")
    
    # 저장 디렉토리 생성
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(RESULTS_BASE_PATH, f"decision_{timestamp}")
        
    # 저장 디렉토리 생성 (중요!)
    os.makedirs(save_dir, exist_ok=True)
    
    # 날짜 인덱스 선택
    if date_index is None:
        date_index = np.random.randint(0, len(test_dates) - 1)
    
    # 범위 확인
    date_index = min(date_index, len(test_dates) - 1)
    
    selected_date = test_dates[date_index]
    logger.info(f"Decision analysis started: date={selected_date}, index={date_index}")
    
    try:
        # 테스트 환경 생성
        env = StockPortfolioEnv(test_data)
        n_assets = env.n_assets
        n_features = env.n_features
        
        # 환경 리셋 (선택한 날짜부터 시작)
        state, _ = env.reset(start_index=date_index)
        
        # 기본 액션 얻기
        if hasattr(agent, 'select_action'):
            action = agent.select_action(state, use_ema=use_ema)
        else:
            action, _, _ = agent.policy_old.act(state)
        
        # LIME 설명기 초기화
        try:
            # 특성 이름 생성 (n_assets와 n_features 기반)
            feature_names = []
            for i in range(min(n_assets, len(STOCK_TICKERS))):
                for j in range(min(n_features, len(FEATURE_NAMES))):
                    feature_names.append(f"{STOCK_TICKERS[i]} - {FEATURE_NAMES[j]}")
            
            # 배경 데이터 생성
            background_data = test_data.reshape(-1, n_assets * n_features)
            n_background = min(100, len(background_data))
            
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=background_data[:n_background],
                feature_names=feature_names[:n_assets * n_features],  # 특성 이름 수 제한
                mode="regression"
            )
            
            # 모델 래핑 함수 - 에러 핸들링 추가
            def safe_lime_predict(x):
                try:
                    # 입력 형태 조정
                    if x.ndim == 1:
                        x = x.reshape(1, -1)
                    
                    # 모델 입력 차원으로 재구성
                    reshaped_x = x.reshape(x.shape[0], n_assets, n_features)
                    
                    return lime_model_fn(agent, reshaped_x, use_ema)
                except Exception as e:
                    logger.error(f"LIME 예측 중 오류: {e}")
                    # 오류 발생 시 0으로 채운 결과 반환
                    return np.zeros((x.shape[0], n_assets))
            
            # 각 자산별 설명 생성
            explanations = []
            
            for asset_idx in range(min(n_assets, len(STOCK_TICKERS))):
                try:
                    # 특정 자산에 대한 예측 함수
                    def predict_asset(x):
                        full_pred = safe_lime_predict(x)
                        return full_pred[:, asset_idx]
                    
                    # LIME 설명 생성
                    exp = explainer.explain_instance(
                        state.reshape(-1), 
                        predict_asset,
                        num_features=min(10, n_assets * n_features)  # 특성 수 제한
                    )
                    explanations.append(exp)
                    
                    # 설명 시각화
                    fig = plt.figure(figsize=(10, 6))
                    exp.as_pyplot_figure()
                    plt.close() # lime 내부에서 생성된 figure 닫기
                    plt.title(f"{STOCK_TICKERS[asset_idx]} Asset Allocation Factors", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"decision_{STOCK_TICKERS[asset_idx]}.png"), dpi=300, bbox_inches='tight')
                    plt.close(fig) # 생성된 fig 닫기
                except Exception as e:
                    logger.warning(f"자산 {STOCK_TICKERS[asset_idx]} LIME 설명 생성 중 오류: {e}")
        except Exception as e:
            logger.warning(f"Error during LIME explanation generation: {e}")
            explanations = []
        
        # 상태 데이터와 행동 데이터 시각화
        
        # 1. 상태 데이터 (자산별 특성) 히트맵
        state_2d = state.reshape(n_assets, n_features)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(state_2d, annot=True, cmap='viridis', 
                    xticklabels=FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))], 
                    yticklabels=STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))], 
                    fmt=".2f")
        plt.title(f'State Data ({selected_date})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "state_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close() # 현재 figure 닫기
        
        # 2. 행동 (자산 배분) 막대 그래프
        fig_action = plt.figure(figsize=(10, 6))
        action_labels = STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))] + ["Cash"]
        
        # 액션 길이 확인 및 시각화 처리
        if len(action) == len(action_labels):
            # 모든 자산 + 현금에 대한 액션 (예상 길이가 맞음)
            plt.bar(range(len(action)), action)
            plt.xticks(range(len(action)), action_labels)
        elif len(action) == len(action_labels) - 1:
            # 현금이 없는 경우 (자산만 있음)
            plt.bar(range(len(action)), action)
            plt.xticks(range(len(action)), action_labels[:len(action)])
            logger.info(f"현금 할당이 없는 액션: {len(action)} 자산")
        elif len(action) > len(action_labels):
            # 액션이 더 많은 경우, 자르기
            logger.warning(f"액션 길이({len(action)})가 레이블 길이({len(action_labels)})보다 김. 일부만 표시합니다.")
            plt.bar(range(len(action_labels)), action[:len(action_labels)])
            plt.xticks(range(len(action_labels)), action_labels)
        else:
            # 그 외 불일치 케이스
            logger.warning(f"액션 길이({len(action)})가 레이블 길이({len(action_labels)})와 일치하지 않습니다. 레이블 없이 표시합니다.")
            plt.bar(range(len(action)), action)
            if len(action) <= 20:  # 레이블이 너무 많지 않은 경우에만 표시
                plt.xticks(range(len(action)), [f"Asset {i}" for i in range(len(action))])

        plt.title(f'Action (Portfolio Allocation) - {selected_date}', fontsize=16)
        plt.xlabel('Assets & Cash')
        plt.ylabel('Allocation')
        plt.ylim(0, max(action) * 1.2 if len(action) > 0 else 1)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "action_bar.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_action) # 현재 figure 닫기
        
        # 결과 저장
        result = {
            "date": selected_date,
            "date_index": date_index,
            "state": state.tolist(),  # ndarray를 리스트로 변환
            "action": action.tolist(),  # ndarray를 리스트로 변환
            "asset_names": STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))],
            "feature_names": FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))],
        }
        
        # CSV로 저장
        state_df = pd.DataFrame(state_2d, index=STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))], 
                               columns=FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))])
        state_df.to_csv(os.path.join(save_dir, "state_data.csv"))
        
        # 행동 데이터프레임 생성 - 배열 크기 확인 및 맞춤
        asset_names = STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))]
        
        # action 배열 길이에 따른 처리
        if len(action) == len(asset_names) + 1:
            # 현금 할당을 포함한 경우
            assets_with_cash = asset_names.copy()
            assets_with_cash.append("Cash")
            action_df = pd.DataFrame({
                "Asset": assets_with_cash,
                "Allocation": action
            })
        elif len(action) == len(asset_names):
            # 현금 할당 없는 경우
            action_df = pd.DataFrame({
                "Asset": asset_names,
                "Allocation": action
            })
        else:
            # 길이가 맞지 않는 경우, 자동 조정
            logger.warning(f"액션 길이({len(action)})가 자산 이름 길이({len(asset_names)})와 맞지 않습니다. 자동 조정합니다.")
            
            # 사용할 레이블 결정
            if len(action) > len(asset_names):
                # 액션이 더 많은 경우, 기존 자산 이름 + 추가 인덱스
                labels = asset_names.copy()
                for i in range(len(asset_names), len(action)):
                    if i == len(action) - 1 and len(action) == len(asset_names) + 1:
                        labels.append("Cash")  # 마지막 하나가 더 있는 경우 현금으로 가정
                    else:
                        labels.append(f"Asset {i}")
            else:
                # 액션이 더 적은 경우, 필요한 만큼만 자산 이름 사용
                labels = asset_names[:len(action)]
            
            action_df = pd.DataFrame({
                "Asset": labels,
                "Allocation": action
            })
        
        action_df.to_csv(os.path.join(save_dir, "action_data.csv"), index=False)
        
        logger.info(f"Decision analysis completed: results saved to = {save_dir}")
        return result
    
    except Exception as e:
        logger.error(f"Error during decision process analysis: {e}")
        logger.error(traceback.format_exc())  # 상세 오류 정보 출력

def integrated_gradients(
    agent,
    test_data,
    test_dates,
    n_samples=10,
    n_steps=INTEGRATED_GRADIENTS_STEPS,
    save_dir=None,
    logger=None,
    use_ema=True
):
    """
    통합 그래디언트(Integrated Gradients) 분석을 수행합니다.
    
    Args:
        agent: 평가할 학습된 PPO 에이전트
        test_data: 테스트 데이터 배열 (시간 x 자산 x 피처)
        test_dates: 테스트 날짜 인덱스
        n_samples: 분석할 샘플 수
        n_steps: 적분 단계 수
        save_dir: 결과 저장 디렉토리
        logger: 로깅 객체
        use_ema: EMA 모델 사용 여부
        
    Returns:
        dict: 통합 그래디언트 분석 결과
    """
    import logging
    import traceback
    
    if logger is None:
        logger = logging.getLogger("PortfolioRL")
    
    # 저장 디렉토리 생성
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(RESULTS_BASE_PATH, f"integrated_gradients_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Integrated gradients analysis started: samples={n_samples}, steps={n_steps}")
    
    # 테스트 환경 생성 및 샘플 수집
    env = StockPortfolioEnv(test_data)
    n_assets = env.n_assets
    n_features = env.n_features
    
    # 샘플 날짜 랜덤 선택
    if len(test_dates) > n_samples:
        sample_indices = np.random.choice(len(test_dates) - 1, size=n_samples, replace=False)
    else:
        sample_indices = np.arange(len(test_dates) - 1)
        n_samples = len(sample_indices)
    
    # 상태, 날짜 샘플 수집
    states = []
    dates_sampled = []
    
    for i in tqdm(sample_indices, desc="Collecting samples"):
        state, _ = env.reset(start_index=i)
        states.append(state)
        dates_sampled.append(test_dates[i])
    
    states = np.array(states)
    
    try:
        # 모델을 평가 모드로 설정
        model = None
        
        # PPO 에이전트에서 모델 접근
        if use_ema and hasattr(agent, 'policy_ema'):
            model = agent.policy_ema
            logger.info("EMA 정책 모델 사용")
        elif hasattr(agent, 'policy_old'):
            model = agent.policy_old
            logger.info("기본 정책 모델 사용")
        elif hasattr(agent, 'policy'):
            model = agent.policy
            logger.info("정책 모델 사용")
        else:
            logger.error("적합한 모델을 찾을 수 없습니다.")
            return None
        
        # 모델이 PyTorch 모듈인지 확인
        if not isinstance(model, torch.nn.Module):
            logger.error(f"모델이 PyTorch 모듈이 아님: {type(model)}")
            return None
        
        # 모델 모드 저장 및 훈련 모드로 설정 (LSTM backward 동작을 위해 필요)
        was_training = model.training
        model.train()  # cudNN RNN의 backward는 training 모드에서만 작동
        logger.info(f"모델 평가 모드 설정: {model.__class__.__name__}")
        
        # 종합 중요도 저장 변수
        integrated_grads_total = np.zeros((n_assets, n_features))
        success_count = 0  # 성공적으로 처리된 샘플 수 추적
        
        # 각 샘플에 대한 통합 그래디언트 계산
        for s_idx, state in enumerate(tqdm(states, desc="Computing integrated gradients")):
            try:
                # 기준선(baseline) 설정 - 상태의 평균값으로 초기화
                state_mean = np.mean(state, keepdims=True)
                baseline = np.ones_like(state) * state_mean
                
                # 상태와 기준선 텐서 변환
                state_tensor = torch.from_numpy(state).float().to(DEVICE)
                baseline_tensor = torch.from_numpy(baseline).float().to(DEVICE)
                gradient_sum = torch.zeros_like(state_tensor)
                alphas = torch.linspace(0, 1, n_steps, device=DEVICE)
            
                for alpha in alphas:
                    # 1. 원본 형태로 보간
                    interpolated_state_orig = baseline_tensor + alpha * (state_tensor - baseline_tensor)

                    # 2. 모델 입력 형태로 변환 (배치 차원 추가)
                    interpolated_state_input = interpolated_state_orig.unsqueeze(0) # (1, n_assets, n_features)
                    interpolated_state_input.requires_grad_(True)

                    # 3. 모델 순전파 및 타겟 설정
                    # 중요: detach()나 torch.no_grad() 없이 순전파 진행
                    concentration, value = model(interpolated_state_input)
                    
                    # value에 대해 requires_grad 확인
                    if not value.requires_grad:
                        value = value.detach().requires_grad_(True)
                        
                    # 상태 가치(value)는 기대 반환과 직접적으로 연결되어 있으므로 IG 대상 스칼라로 사용
                    target_output = value.squeeze()

                    # 4. 그래디언트 계산
                    model.zero_grad()
                    target_output.backward()

                    # 5. 입력에 대한 그래디언트 추출 및 누적
                    if interpolated_state_input.grad is not None:
                        # 그래디언트는 입력 텐서와 동일한 형태 (1, n_assets, n_features) -> 배치 차원 제거
                        gradient = interpolated_state_input.grad.squeeze(0)
                        if gradient.shape == state_tensor.shape:
                             gradient_sum += gradient
                        else:
                             # 형태가 예상과 다를 경우 경고 로깅
                             logger.warning(f"IG: 그래디언트 형태 불일치 발생. grad shape: {gradient.shape}, expected: {state_tensor.shape}. 해당 스텝 건너뛰기.")
                    else: 
                        logger.warning(f"IG: Alpha {alpha:.2f}에서 그래디언트가 None입니다.")
                
                # 6. 최종 IG 계산
                integrated_grads_tensor = (state_tensor - baseline_tensor) * (gradient_sum / n_steps)
                integrated_grad_np = integrated_grads_tensor.cpu().numpy()
                
                # 7. 누적 및 개별 시각화
                integrated_grads_total += np.abs(integrated_grad_np)
                success_count += 1
                
                # 개별 샘플 시각화 (최대 3개만)
                if s_idx < 3:
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(
                        np.abs(integrated_grad_np),
                        annot=True,
                        cmap='viridis',
                        xticklabels=FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))],
                        yticklabels=STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))],
                        fmt=".3f"
                    )
                    plt.title(f"Integrated Gradients - Sample {s_idx+1} ({dates_sampled[s_idx].strftime('%Y-%m-%d')})", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"integrated_gradients_sample_{s_idx+1}.png"), dpi=300, bbox_inches='tight')
                    plt.close()
            
            except Exception as e:
                logger.warning(f"샘플 {s_idx} 처리 중 오류 발생: {e}")
                logger.warning(traceback.format_exc())
                continue
        
        # 원래의 모델 모드로 복원
        if not was_training:
            model.eval()

        # 성공적으로 처리된 샘플이 있는지 확인
        if success_count == 0:
            logger.error("모든 샘플에 대해 통합 그래디언트 계산 실패")
            return None
            
        logger.info(f"성공적으로 처리된 샘플: {success_count}/{n_samples}")
        
        # 평균 통합 그래디언트
        integrated_grads_avg = integrated_grads_total / success_count
        
        # 정규화
        if np.sum(integrated_grads_avg) > 0:
            integrated_grads_norm = integrated_grads_avg / np.sum(integrated_grads_avg)
        else:
            integrated_grads_norm = integrated_grads_avg
        
        # 자산별, 특성별 중요도 집계
        asset_importance = np.sum(integrated_grads_norm, axis=1)
        feature_importance = np.sum(integrated_grads_norm, axis=0)
        
        # 정규화
        asset_importance = asset_importance / np.sum(asset_importance) if np.sum(asset_importance) > 0 else asset_importance
        feature_importance = feature_importance / np.sum(feature_importance) if np.sum(feature_importance) > 0 else feature_importance
        
        # 히트맵 시각화
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            integrated_grads_norm,
            annot=True,
            cmap='viridis',
            xticklabels=FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))],
            yticklabels=STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))],
            fmt=".3f"
        )
        plt.title("Integrated Gradients - Feature-Asset Importance", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "integrated_gradients_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 자산별 중요도 시각화
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(asset_importance)), asset_importance)
        plt.xticks(range(len(asset_importance)), STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))])
        plt.title("Asset Importance by Integrated Gradients", fontsize=16)
        plt.xlabel("Assets")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "integrated_gradients_asset_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 특성별 중요도 시각화
        plot_feature_importance(
            feature_importance,
            FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))],
            save_path=os.path.join(save_dir, "integrated_gradients_feature_importance.png")
        )
        
        # 결과 저장
        result = {
            "integrated_grads": integrated_grads_norm,
            "asset_importance": asset_importance,
            "feature_importance": feature_importance,
            "feature_names": FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))],
            "asset_names": STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))],
            "n_samples": n_samples,
            "successful_samples": success_count,
            "n_steps": n_steps,
            "dates_sampled": [date.strftime("%Y-%m-%d") for date in dates_sampled]
        }
        
        # CSV로 저장
        integrated_df = pd.DataFrame(
            integrated_grads_norm,
            index=STOCK_TICKERS[:min(n_assets, len(STOCK_TICKERS))],
            columns=FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))]
        )
        integrated_df.to_csv(os.path.join(save_dir, "integrated_gradients.csv"))
        
        # 특성 중요도 CSV 저장
        features_df = pd.DataFrame({
            "Feature": FEATURE_NAMES[:min(n_features, len(FEATURE_NAMES))],
            "Importance": feature_importance
        })
        features_df.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)
        
        return result
        
    except Exception as e:
        logger.error(f"통합 그래디언트 분석 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
        return None

def compute_integrated_gradients_for_single_state(model, state, baseline=None, steps=INTEGRATED_GRADIENTS_STEPS, logger=None):
    """
    단일 상태에 대한 통합 그래디언트를 계산합니다.
    
    Args:
        model (ActorCritic): ActorCritic 모델 인스턴스
        state (np.ndarray): 분석할 입력 상태 (n_assets, n_features)
        baseline (np.ndarray, optional): 기준선 상태. Defaults to None (상태 평균값 사용)
        steps (int): 적분 단계 수
        logger (logging.Logger, optional): 로깅 객체
        
    Returns:
        np.ndarray: 각 특성에 대한 통합 그래디언트 값
    """
    if logger is None:
        logger = logging.getLogger('PortfolioRL')
    
    # 기준선(baseline) 정의 - 기본값은 상태의 평균값으로 초기화
    if baseline is None:
        state_mean = np.mean(state, keepdims=True)
        baseline = np.ones_like(state) * state_mean
    
    try:
        if state.shape != baseline.shape:
            raise ValueError(f"State shape {state.shape}와 Baseline shape {baseline.shape} 불일치")

        # 텐서 변환
        state_tensor = torch.from_numpy(state).float().to(DEVICE)
        baseline_tensor = torch.from_numpy(baseline).float().to(DEVICE)
        gradient_sum = torch.zeros_like(state_tensor)
        alphas = torch.linspace(0, 1, steps, device=DEVICE)
    
        # 모델의 현재 모드 저장
        was_training = model.training
        
        # cudNN RNN의 backward는 training 모드에서만 작동하므로 임시로 모드 변경
        model.train()
        
        for alpha in alphas:
            # 1. 원본 형태로 보간
            interpolated_state_orig = baseline_tensor + alpha * (state_tensor - baseline_tensor)

            # 2. 모델 입력 형태로 변환 (배치 차원 추가)
            interpolated_state_input = interpolated_state_orig.unsqueeze(0) # (1, n_assets, n_features)
            interpolated_state_input.requires_grad_(True)

            # 3. 모델 순전파 및 타겟 설정
            # 중요: detach()나 torch.no_grad() 없이 순전파 진행
            concentration, value = model(interpolated_state_input)
            
            # value에 대해 requires_grad 확인
            if not value.requires_grad:
                value = value.detach().requires_grad_(True)
                
            # 상태 가치(value)는 기대 반환과 직접적으로 연결되어 있으므로 IG 대상 스칼라로 사용
            target_output = value.squeeze()

            # 4. 그래디언트 계산
            model.zero_grad()
            target_output.backward()

            # 5. 입력에 대한 그래디언트 추출 및 누적
            if interpolated_state_input.grad is not None:
                # 그래디언트는 입력 텐서와 동일한 형태 (1, n_assets, n_features) -> 배치 차원 제거
                gradient = interpolated_state_input.grad.squeeze(0)
                if gradient.shape == state_tensor.shape:
                     gradient_sum += gradient
                else:
                     # 형태가 예상과 다를 경우 경고 로깅
                     logger.warning(f"IG: 그래디언트 형태 불일치 발생. grad shape: {gradient.shape}, expected: {state_tensor.shape}. 해당 스텝 건너뛰기.")
            else: 
                logger.warning(f"IG: Alpha {alpha:.2f}에서 그래디언트가 None입니다.")
        
        # 원래의 모델 모드로 복원
        if not was_training:
            model.eval()

        # 6. 최종 IG 계산
        integrated_grads_tensor = (state_tensor - baseline_tensor) * (gradient_sum / steps)
        return integrated_grads_tensor.cpu().numpy()

    except Exception as e:
        logger.error(f"Integrated Gradients 계산 중 오류: {e}")
        logger.error(traceback.format_exc())
        return np.zeros_like(state)

def run_model_interpretability(
    agent,
    test_data,
    test_dates,
    methods=None,
    samples=XAI_SAMPLE_COUNT,
    save_dir=None,
    logger=None,
    use_ema=True
):
    """
    모델 해석 가능성 분석을 실행하는 통합 함수입니다.
    
    Args:
        agent: 평가할 학습된 PPO 에이전트
        test_data: 테스트 데이터 배열 (시간 x 자산 x 피처)
        test_dates: 테스트 날짜 인덱스
        methods: 사용할 분석 방법 리스트 (기본값: ['shap', 'sensitivity', 'decision', 'integrated_gradients'])
        samples: 분석에 사용할 샘플 수
        save_dir: 결과 저장 기본 디렉토리
        logger: 로깅 객체
        use_ema: EMA 모델 사용 여부
        
    Returns:
        dict: 모든 분석 결과를 포함하는 딕셔너리
    """
    import logging
    if logger is None:
        logger = logging.getLogger("PortfolioRL")
    
    # 기본 분석 방법 설정
    if methods is None:
        methods = ['shap', 'sensitivity', 'decision', 'integrated_gradients']
    
    logger.info(f"Model interpretability analysis started: methods={methods}, samples={samples}")
    
    # 저장 디렉토리 생성
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(RESULTS_BASE_PATH, f"xai_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
    
    # 분석 결과 저장 딕셔너리
    results = {
        "methods": methods,
        "samples": samples,
        "save_dir": save_dir
    }
    
    # 통합 특성 중요도
    combined_importance = None
    feature_names = None
    asset_names = None
    importance_count = 0
    
    # SHAP 분석
    if 'shap' in methods:
        logger.info("=== SHAP 기반 특성 중요도 분석 시작 ===")
        shap_dir = os.path.join(save_dir, "shap")
        os.makedirs(shap_dir, exist_ok=True)
        
        shap_result = compute_feature_importance(
            agent=agent,
            test_data=test_data,
            test_dates=test_dates,
            method="shap",
            n_samples=samples,
            save_dir=shap_dir,
            logger=logger,
            use_ema=use_ema
        )
        
        if shap_result is not None:
            results["shap"] = shap_result
            
            # 통합 중요도 계산에 추가
            if "feature_importance_by_feature" in shap_result:
                if combined_importance is None:
                    combined_importance = shap_result["feature_importance_by_feature"]
                    feature_names = shap_result.get("feature_names", None)
                    asset_names = shap_result.get("asset_names", None)
                else:
                    combined_importance += shap_result["feature_importance_by_feature"]
                importance_count += 1
                logger.info("SHAP 결과가 통합 중요도 계산에 추가되었습니다.")
    
    # 민감도 분석
    if 'sensitivity' in methods:
        logger.info("=== 민감도 분석 시작 ===")
        sensitivity_dir = os.path.join(save_dir, "sensitivity")
        os.makedirs(sensitivity_dir, exist_ok=True)
        
        sensitivity_result = sensitivity_analysis(
            agent=agent,
            test_data=test_data,
            test_dates=test_dates,
            n_samples=samples,
            save_dir=sensitivity_dir,
            logger=logger,
            use_ema=use_ema
        )
        
        if sensitivity_result is not None:
            results["sensitivity"] = sensitivity_result
            
            # 통합 중요도 계산에 추가
            if "feature_sensitivity" in sensitivity_result:
                if combined_importance is None:
                    combined_importance = sensitivity_result["feature_sensitivity"]
                    feature_names = sensitivity_result.get("feature_names", None)
                    asset_names = sensitivity_result.get("asset_names", None)
                else:
                    combined_importance += sensitivity_result["feature_sensitivity"]
                importance_count += 1
                logger.info("민감도 분석 결과가 통합 중요도 계산에 추가되었습니다.")
    
    # 의사결정 과정 시각화
    if 'decision' in methods:
        logger.info("=== 의사결정 과정 시각화 시작 ===")
        decision_dir = os.path.join(save_dir, "decision")
        os.makedirs(decision_dir, exist_ok=True)
        
        # 여러 날짜에 대한 의사결정 분석
        n_decision_samples = min(3, len(test_dates) - 1)  # 최대 3개의 날짜만 분석
        decision_results = []
        
        for i in range(n_decision_samples):
            date_index = int(len(test_dates) * (i + 1) / (n_decision_samples + 1))
            
            decision_result = plot_decision_process(
                agent=agent,
                test_data=test_data,
                test_dates=test_dates,
                date_index=date_index,
                save_dir=os.path.join(decision_dir, f"sample_{i+1}"),
                logger=logger,
                use_ema=use_ema
            )
            
            if decision_result is not None:
                decision_results.append(decision_result)
        
        if decision_results:
            results["decision"] = decision_results

    # 통합 그래디언트 분석
    if 'integrated_gradients' in methods:
        logger.info("=== 통합 그래디언트 분석 시작 ===")
        ig_dir = os.path.join(save_dir, "integrated_gradients")
        os.makedirs(ig_dir, exist_ok=True)
        
        ig_result = integrated_gradients(
            agent=agent,
            test_data=test_data,
            test_dates=test_dates,
            n_samples=samples // 2,  # 통합 그래디언트는 계산량이 많아 샘플 수 줄임
            n_steps=INTEGRATED_GRADIENTS_STEPS,
            save_dir=ig_dir,
            logger=logger,
            use_ema=use_ema
        )
        
        if ig_result is not None:
            results["integrated_gradients"] = ig_result
            
            # 통합 중요도 계산에 추가
            if "feature_importance" in ig_result:
                if combined_importance is None:
                    combined_importance = ig_result["feature_importance"]
                    feature_names = ig_result.get("feature_names", None)
                    asset_names = ig_result.get("asset_names", None)
                else:
                    combined_importance += ig_result["feature_importance"]
                importance_count += 1
                logger.info("통합 그래디언트 결과가 통합 중요도 계산에 추가되었습니다.")
    
    # 통합 특성 중요도 계산 및 시각화
    if combined_importance is not None and importance_count > 0:
        # 평균 계산
        combined_importance = combined_importance / importance_count
        
        # 정규화
        if np.sum(combined_importance) > 0:
            combined_importance = combined_importance / np.sum(combined_importance)
        
        # 통합 결과 시각화
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(combined_importance)), combined_importance)
        if feature_names is not None:
            plt.xticks(range(len(combined_importance)), feature_names, rotation=45, ha='right')
        plt.title("Combined Feature Importance", fontsize=16)
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "combined_feature_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 결과 저장
        combined_feature_df = pd.DataFrame({
            "Feature": feature_names if feature_names is not None else [f"Feature_{i}" for i in range(len(combined_importance))],
            "Importance": combined_importance
        })
        combined_feature_df.to_csv(os.path.join(save_dir, "combined_feature_importance.csv"), index=False)
        
        # 통합 결과 딕셔너리에 추가
        results["combined_importance"] = combined_importance
        if feature_names is not None:
            results["feature_names"] = feature_names
        if asset_names is not None:
            results["asset_names"] = asset_names
    
    logger.info(f"Model interpretability analysis completed: methods={methods}, results saved to {save_dir}")
    return results