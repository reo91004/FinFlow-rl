# tests/debug_training.py - 학습 신호 디버깅 (Loss, Gradient, TD-error)

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from config import *
from agents.bcell import BCell
from data import DataLoader
from core.environment import PortfolioEnvironment
from core.system import ImmunePortfolioSystem
from data.features import FeatureExtractor


def debug_training_signals():
    """학습 신호 (Loss, Gradient, TD-error) 디버깅"""

    print("=" * 80)
    print("학습 신호 디버깅 시작")
    print("=" * 80)

    # 1. 테스트 환경 구성
    print("[1] 테스트 환경 구성...")

    # 작은 데이터셋 생성
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    symbols = SYMBOLS[:5]
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, (50, 5))
    prices = pd.DataFrame(
        np.cumprod(1 + returns, axis=0) * 100, index=dates, columns=symbols
    )

    # 환경 및 시스템 초기화
    feature_extractor = FeatureExtractor(lookback_window=20)
    env = PortfolioEnvironment(prices, feature_extractor, initial_capital=100000)
    immune_system = ImmunePortfolioSystem(n_assets=len(symbols), state_dim=18)

    # 더미 학습 데이터로 T-Cell 훈련
    dummy_features = np.random.randn(100, 12)
    immune_system.fit_tcell(dummy_features)

    print(f"환경 설정 완료: {len(symbols)}개 자산, {env.max_steps} 스텝")

    # 2. 짧은 에피소드 실행하여 경험 수집
    print(f"\n[2] 경험 수집 (20 스텝)...")

    state = env.reset()
    experiences = []

    for step in range(20):
        # 의사결정
        weights, decision_info = immune_system.decide(state, training=True)

        # 환경 스텝
        next_state, reward, done, info = env.step(weights)

        # 경험 저장
        experiences.append(
            {
                "state": state.copy(),
                "action": weights.copy(),
                "reward": reward,
                "next_state": next_state.copy(),
                "done": done,
                "step": step,
            }
        )

        # 시스템 업데이트 (경험 저장)
        immune_system.update(state, weights, reward, next_state, done)

        print(
            f"  스텝 {step+1:2d}: 보상={reward:+7.4f}, 선택={decision_info['selected_bcell'][:4]}"
        )

        state = next_state
        if done:
            break

    print(f"경험 수집 완료: {len(experiences)} 스텝")

    # 3. B-Cell별 학습 신호 분석
    print(f"\n[3] B-Cell 학습 신호 분석...")

    training_stats = {}

    for bcell_name, bcell in immune_system.bcells.items():
        print(f"\n  === {bcell_name} B-Cell ===")

        # 충분한 경험이 있는지 확인
        buffer_size = len(bcell.replay_buffer)
        print(f"    버퍼 크기: {buffer_size}")

        if buffer_size < bcell.batch_size:
            print(f"    ⚠️  경험 부족 (필요: {bcell.batch_size}, 보유: {buffer_size})")
            continue

        # 학습 전 상태 기록
        pre_losses = {
            "actor": bcell.actor_losses[-5:] if bcell.actor_losses else [],
            "critic": bcell.critic_losses[-5:] if bcell.critic_losses else [],
        }

        # 학습 실행 (5회)
        print(f"    5회 업데이트 실행...")
        for update_i in range(5):
            bcell.update()

        # 학습 후 상태 기록
        post_losses = {
            "actor": bcell.actor_losses[-5:],
            "critic": bcell.critic_losses[-5:],
        }

        # 통계 계산
        if post_losses["actor"] and post_losses["critic"]:
            avg_actor_loss = np.mean(post_losses["actor"])
            avg_critic_loss = np.mean(post_losses["critic"])

            print(f"    평균 Actor Loss: {avg_actor_loss:.6f}")
            print(f"    평균 Critic Loss: {avg_critic_loss:.6f}")
            print(f"    Alpha 값: {bcell.alpha.item():.4f}")
            print(f"    업데이트 횟수: {bcell.update_count}")

            # 손실 분석
            if avg_actor_loss > 1000:
                print(f"    ⚠️  Actor 손실이 매우 높습니다!")
            elif avg_actor_loss < 1e-6:
                print(f"    ⚠️  Actor 손실이 너무 낮아 학습이 안 될 수 있습니다!")
            else:
                print(f"    ✅ Actor 손실이 정상 범위입니다.")

            if avg_critic_loss > 1000:
                print(f"    ⚠️  Critic 손실이 매우 높습니다!")
            elif np.isnan(avg_critic_loss):
                print(f"    ❌ Critic 손실에 NaN이 발생했습니다!")
            else:
                print(f"    ✅ Critic 손실이 정상 범위입니다.")

            training_stats[bcell_name] = {
                "avg_actor_loss": avg_actor_loss,
                "avg_critic_loss": avg_critic_loss,
                "alpha": bcell.alpha.item(),
                "update_count": bcell.update_count,
            }
        else:
            print(f"    ⚠️  손실 데이터 없음")

    # 4. 그래디언트 분석
    print(f"\n[4] 그래디언트 분석...")

    # 하나의 B-Cell에서 그래디언트 상세 분석
    test_bcell = immune_system.bcells["volatility"]

    if len(test_bcell.replay_buffer) >= test_bcell.batch_size:
        print(f"  Volatility B-Cell 그래디언트 분석...")

        # 배치 샘플링
        batch, is_weights, indices = test_bcell.replay_buffer.sample(
            test_bcell.batch_size
        )
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states, dtype=np.float32)).to(DEVICE)
        actions = torch.tensor(np.array(actions, dtype=np.float32)).to(DEVICE)
        rewards = torch.tensor([float(r) for r in rewards]).to(DEVICE)
        next_states = torch.tensor(np.array(next_states, dtype=np.float32)).to(DEVICE)
        dones = torch.tensor([bool(d) for d in dones]).to(DEVICE)

        # Forward pass
        _, current_actions, current_log_probs = test_bcell.actor(states)
        q1_current = test_bcell.critic1(states, current_actions).squeeze()
        q2_current = test_bcell.critic2(states, current_actions).squeeze()

        # Actor 손실 계산
        q_current = torch.min(q1_current, q2_current)
        actor_loss = (test_bcell.alpha * current_log_probs - q_current).mean()

        # 그래디언트 계산
        test_bcell.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)

        # 그래디언트 통계
        actor_grad_norms = []
        for param in test_bcell.actor.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm().item()
                actor_grad_norms.append(grad_norm)

        if actor_grad_norms:
            avg_grad_norm = np.mean(actor_grad_norms)
            max_grad_norm = np.max(actor_grad_norms)

            print(f"    Actor 그래디언트:")
            print(f"      평균 norm: {avg_grad_norm:.6f}")
            print(f"      최대 norm: {max_grad_norm:.6f}")
            print(f"      레이어 수: {len(actor_grad_norms)}")

            if avg_grad_norm < 1e-6:
                print(f"      ⚠️  그래디언트가 너무 작습니다 (Vanishing)!")
            elif avg_grad_norm > 10:
                print(f"      ⚠️  그래디언트가 너무 큽니다 (Exploding)!")
            else:
                print(f"      ✅ 그래디언트가 적절합니다.")

    # 5. TD-error 분포 분석
    print(f"\n[5] TD-error 분포 분석...")

    for bcell_name in ["volatility", "correlation"]:
        bcell = immune_system.bcells[bcell_name]

        if (
            hasattr(bcell, "monitoring_stats")
            and bcell.monitoring_stats["td_error_stats"]["mean"]
        ):
            td_means = bcell.monitoring_stats["td_error_stats"]["mean"]
            td_maxs = bcell.monitoring_stats["td_error_stats"]["max"]

            print(f"  {bcell_name} B-Cell TD-error:")
            print(f"    평균 TD-error: {np.mean(td_means):.6f}")
            print(f"    최대 TD-error: {np.mean(td_maxs):.6f}")
            print(f"    TD-error 기록 수: {len(td_means)}")

            if np.mean(td_means) > 1.0:
                print(f"    ⚠️  TD-error가 높아 가치 추정이 불안정할 수 있습니다!")
            else:
                print(f"    ✅ TD-error가 적절합니다.")

    print("\n" + "=" * 80)
    print("학습 신호 디버깅 완료")
    print("=" * 80)

    return training_stats


def analyze_learning_convergence(training_stats):
    """학습 수렴성 분석"""

    print("\n" + "=" * 60)
    print("학습 수렴성 분석")
    print("=" * 60)

    convergence_issues = []

    for bcell_name, stats in training_stats.items():
        print(f"\n{bcell_name} B-Cell:")

        actor_loss = stats["avg_actor_loss"]
        critic_loss = stats["avg_critic_loss"]
        alpha = stats["alpha"]

        print(f"  Actor Loss: {actor_loss:.6f}")
        print(f"  Critic Loss: {critic_loss:.6f}")
        print(f"  Alpha: {alpha:.4f}")

        # 수렴성 평가
        if np.isnan(actor_loss) or np.isnan(critic_loss):
            convergence_issues.append(f"{bcell_name}: NaN 손실 발생")
        elif actor_loss > 100 or critic_loss > 100:
            convergence_issues.append(f"{bcell_name}: 손실이 너무 큼")
        elif actor_loss < 1e-8 and critic_loss < 1e-8:
            convergence_issues.append(f"{bcell_name}: 손실이 너무 작음 (학습 정체)")
        else:
            print(f"  ✅ 학습 신호가 양호합니다.")

    if convergence_issues:
        print(f"\n발견된 수렴 문제:")
        for i, issue in enumerate(convergence_issues, 1):
            print(f"  {i}. {issue}")

        print(f"\n권장 해결책:")
        print(f"  - 학습률 조정 (Actor/Critic: 3e-4)")
        print(f"  - 그래디언트 클리핑 강화 (1.0 → 0.5)")
        print(f"  - Target 네트워크 업데이트 빈도 조정")
        print(f"  - 보상 정규화 추가")
    else:
        print(f"\n✅ 모든 B-Cell이 정상적으로 학습 중입니다.")


if __name__ == "__main__":
    try:
        training_stats = debug_training_signals()

        if training_stats:
            analyze_learning_convergence(training_stats)

        print(f"\n🎯 핵심 발견사항:")
        print(f"   1. 학습 신호 (Loss, Gradient) 상태 점검 완료")
        print(f"   2. TD-error 분포 확인됨")
        print(f"   3. 수렴성 문제 여부 파악됨")

    except Exception as e:
        print(f"\n❌ 학습 신호 디버깅 실행 중 오류: {e}")
        import traceback

        traceback.print_exc()
