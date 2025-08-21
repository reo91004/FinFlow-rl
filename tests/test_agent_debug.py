# tests/debug_agent.py - SAC 에이전트 탐험 및 초기화 디버깅

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd
from datetime import datetime

from config import *
from agents.bcell import BCell
from data import DataLoader
from core.environment import PortfolioEnvironment
from data.features import FeatureExtractor


def debug_sac_agent():
    """SAC 에이전트의 탐험 전략 및 초기화 디버깅"""

    print("=" * 80)
    print("SAC 에이전트 디버깅 시작")
    print("=" * 80)

    # 1. 테스트 환경 설정
    print("[1] 테스트 환경 설정...")
    n_assets = 5
    state_dim = 12 + 1 + n_assets  # features + crisis + weights

    # B-Cell 초기화
    bcell = BCell("volatility", state_dim, n_assets)

    print(f"B-Cell 초기화 완료:")
    print(f"  - 위험 유형: {bcell.risk_type}")
    print(f"  - 상태 차원: {bcell.state_dim}")
    print(f"  - 행동 차원: {bcell.action_dim}")
    print(f"  - 초기 Alpha: {bcell.alpha.item():.4f}")
    print(f"  - Target Entropy: {bcell.target_entropy:.4f}")
    print(f"  - 업데이트 빈도: {bcell.update_frequency}")

    # 2. 탐험 행동 테스트
    print(f"\n[2] 탐험 행동 분석 (100회 샘플링)...")

    # 더미 상태 생성
    test_state = np.concatenate(
        [
            np.random.randn(12) * 0.1,  # 정규화된 시장 특성
            [0.3],  # 위기 수준
            np.ones(n_assets) / n_assets,  # 균등 가중치
        ]
    )

    # 100번 행동 샘플링
    actions_training = []
    actions_eval = []

    bcell.actor.train()  # 훈련 모드
    for _ in range(100):
        action = bcell.get_action(test_state, deterministic=False)
        actions_training.append(action)

    bcell.actor.eval()  # 평가 모드
    for _ in range(100):
        action = bcell.get_action(test_state, deterministic=True)
        actions_eval.append(action)

    actions_training = np.array(actions_training)
    actions_eval = np.array(actions_eval)

    # 탐험 정도 분석
    training_entropy = -np.sum(
        actions_training * np.log(actions_training + 1e-8), axis=1
    ).mean()
    eval_entropy = -np.sum(actions_eval * np.log(actions_eval + 1e-8), axis=1).mean()

    print(f"  훈련 모드:")
    print(f"    평균 엔트로피: {training_entropy:.4f}")
    print(f"    가중치 표준편차: {actions_training.std(axis=0).mean():.4f}")
    print(
        f"    최대 가중치 범위: [{actions_training.max():.3f}, {actions_training.min():.3f}]"
    )

    print(f"  평가 모드:")
    print(f"    평균 엔트로피: {eval_entropy:.4f}")
    print(f"    가중치 표준편차: {actions_eval.std(axis=0).mean():.4f}")
    print(f"    최대 가중치 범위: [{actions_eval.max():.3f}, {actions_eval.min():.3f}]")

    # 탐험 적절성 평가
    exploration_ratio = (
        training_entropy / eval_entropy if eval_entropy > 0 else float("inf")
    )
    print(f"    탐험 비율: {exploration_ratio:.2f}")

    if exploration_ratio < 1.5:
        print("    ⚠️  탐험이 부족할 수 있습니다!")
    elif exploration_ratio > 10:
        print("    ⚠️  탐험이 과도할 수 있습니다!")
    else:
        print("    ✅ 탐험 정도가 적절합니다.")

    # 3. 하이퍼파라미터 적절성 검사
    print(f"\n[3] 하이퍼파라미터 적절성 검사...")

    # 학습률 체크
    print(f"  학습률:")
    print(f"    Actor LR: {ACTOR_LR:.0e} (권장: 1e-4~3e-4)")
    print(f"    Critic LR: {CRITIC_LR:.0e} (권장: 1e-4~3e-4)")
    print(f"    Alpha LR: {ALPHA_LR:.0e} (권장: 1e-4~3e-4)")

    if ACTOR_LR < 1e-5:
        print("    ⚠️  Actor 학습률이 너무 낮을 수 있습니다!")
    elif ACTOR_LR > 1e-3:
        print("    ⚠️  Actor 학습률이 너무 높을 수 있습니다!")
    else:
        print("    ✅ Actor 학습률이 적절합니다.")

    # 할인율 체크
    print(f"\n  강화학습 파라미터:")
    print(f"    Gamma (할인율): {GAMMA} (권장: 0.95~0.99)")
    print(f"    Tau (타겟 업데이트): {TAU} (권장: 0.005~0.01)")
    print(f"    배치 크기: {BATCH_SIZE} (권장: 64~256)")
    print(f"    버퍼 크기: {BUFFER_SIZE} (권장: 10K~1M)")

    if GAMMA < 0.9:
        print("    ⚠️  할인율이 너무 낮아 장기 학습에 문제가 있을 수 있습니다!")
    elif GAMMA > 0.999:
        print("    ⚠️  할인율이 너무 높아 수렴이 어려울 수 있습니다!")
    else:
        print("    ✅ 할인율이 적절합니다.")

    # 4. 네트워크 초기화 상태 검사
    print(f"\n[4] 네트워크 초기화 상태 검사...")

    # Actor 네트워크 가중치 분포
    actor_weights = []
    for param in bcell.actor.parameters():
        actor_weights.extend(param.data.flatten().cpu().numpy())

    actor_weights = np.array(actor_weights)

    print(f"  Actor 네트워크:")
    print(f"    가중치 평균: {actor_weights.mean():.6f}")
    print(f"    가중치 표준편차: {actor_weights.std():.6f}")
    print(f"    가중치 범위: [{actor_weights.min():.6f}, {actor_weights.max():.6f}]")

    if actor_weights.std() < 1e-3:
        print("    ⚠️  가중치 초기화가 너무 작을 수 있습니다!")
    elif actor_weights.std() > 1:
        print("    ⚠️  가중치 초기화가 너무 클 수 있습니다!")
    else:
        print("    ✅ 가중치 초기화가 적절합니다.")

    # Critic 네트워크 가중치 분포
    critic_weights = []
    for param in bcell.critic1.parameters():
        critic_weights.extend(param.data.flatten().cpu().numpy())

    critic_weights = np.array(critic_weights)

    print(f"  Critic 네트워크:")
    print(f"    가중치 평균: {critic_weights.mean():.6f}")
    print(f"    가중치 표준편차: {critic_weights.std():.6f}")
    print(f"    가중치 범위: [{critic_weights.min():.6f}, {critic_weights.max():.6f}]")

    # 5. Dirichlet 분포 파라미터 검사
    print(f"\n[5] Dirichlet 분포 파라미터 검사...")

    with torch.no_grad():
        state_tensor = (
            torch.tensor(test_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )
        concentration, weights, log_prob = bcell.actor(state_tensor)

        concentration_np = concentration.cpu().numpy().flatten()
        weights_np = weights.cpu().numpy().flatten()

    print(f"  Concentration 파라미터:")
    print(f"    평균: {concentration_np.mean():.4f}")
    print(f"    표준편차: {concentration_np.std():.4f}")
    print(f"    범위: [{concentration_np.min():.4f}, {concentration_np.max():.4f}]")

    if concentration_np.mean() < 1.0:
        print("    ⚠️  Concentration이 너무 낮아 탐험이 과도할 수 있습니다!")
    elif concentration_np.mean() > 10.0:
        print("    ⚠️  Concentration이 너무 높아 탐험이 부족할 수 있습니다!")
    else:
        print("    ✅ Concentration이 적절합니다.")

    print(f"  생성된 가중치:")
    print(f"    가중치: {weights_np}")
    print(f"    가중치 합: {weights_np.sum():.6f}")
    print(f"    엔트로피: {-np.sum(weights_np * np.log(weights_np + 1e-8)):.4f}")

    print("\n" + "=" * 80)
    print("SAC 에이전트 디버깅 완료")
    print("=" * 80)

    return {
        "training_entropy": training_entropy,
        "eval_entropy": eval_entropy,
        "exploration_ratio": exploration_ratio,
        "actor_weights_std": actor_weights.std(),
        "concentration_mean": concentration_np.mean(),
    }


def analyze_learning_issues(debug_results):
    """학습 문제점 종합 분석"""

    print("\n" + "=" * 60)
    print("학습 문제점 종합 분석")
    print("=" * 60)

    issues = []

    # 1. 탐험 문제
    if debug_results["exploration_ratio"] < 1.5:
        issues.append("탐험 부족: 다양한 전략을 시도하지 않음")
    elif debug_results["exploration_ratio"] > 10:
        issues.append("과도한 탐험: 학습된 지식을 활용하지 않음")

    # 2. 네트워크 초기화 문제
    if debug_results["actor_weights_std"] < 1e-3:
        issues.append("네트워크 초기화 불량: 가중치가 너무 작음")
    elif debug_results["actor_weights_std"] > 1:
        issues.append("네트워크 초기화 불량: 가중치가 너무 큼")

    # 3. Concentration 문제
    if debug_results["concentration_mean"] < 1.0:
        issues.append("Dirichlet concentration 너무 낮음: 불안정한 정책")
    elif debug_results["concentration_mean"] > 10.0:
        issues.append("Dirichlet concentration 너무 높음: 경직된 정책")

    # 4. 하이퍼파라미터 문제
    if ACTOR_LR < 1e-5:
        issues.append("Actor 학습률 너무 낮음: 학습 속도 저하")
    elif ACTOR_LR > 1e-3:
        issues.append("Actor 학습률 너무 높음: 불안정한 학습")

    print(f"발견된 문제점: {len(issues)}개")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    if not issues:
        print("✅ 에이전트 초기 조건이 양호합니다.")

    # 권장 개선사항
    print(f"\n권장 개선사항:")
    if debug_results["exploration_ratio"] < 1.5:
        print("  - Target entropy 증가: -15 → -7.5")
        print("  - Alpha 학습률 증가: 1e-4 → 3e-4")

    if debug_results["concentration_mean"] < 1.0:
        print("  - Concentration 최소값 증가: 1.0 → 2.0")

    if ACTOR_LR < 3e-4:
        print("  - Actor 학습률 증가: 1e-4 → 3e-4")


if __name__ == "__main__":
    try:
        debug_results = debug_sac_agent()
        analyze_learning_issues(debug_results)

        print(f"\n🎯 핵심 발견사항:")
        print(f"   1. SAC 에이전트의 탐험/활용 균형 점검 완료")
        print(f"   2. 네트워크 초기화 상태 확인됨")
        print(f"   3. 다음 단계에서 실제 학습 신호를 분석해야 함")

    except Exception as e:
        print(f"\n❌ 에이전트 디버깅 실행 중 오류: {e}")
        import traceback

        traceback.print_exc()
