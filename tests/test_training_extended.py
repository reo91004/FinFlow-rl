# tests/debug_training_extended.py - 충분한 경험으로 학습 신호 재분석

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
from core.system import ImmunePortfolioSystem
from data.features import FeatureExtractor


def debug_training_with_sufficient_data():
    """충분한 데이터로 학습 신호 디버깅"""

    print("=" * 80)
    print("확장된 학습 신호 디버깅 (충분한 경험)")
    print("=" * 80)

    # 1. 더 큰 테스트 환경 구성
    print("[1] 확장된 테스트 환경 구성...")

    # 더 큰 데이터셋 생성 (100 거래일)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    symbols = SYMBOLS[:5]
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, (100, 5))
    prices = pd.DataFrame(
        np.cumprod(1 + returns, axis=0) * 100, index=dates, columns=symbols
    )

    # 환경 및 시스템 초기화
    feature_extractor = FeatureExtractor(lookback_window=20)
    env = PortfolioEnvironment(prices, feature_extractor, initial_capital=100000)
    immune_system = ImmunePortfolioSystem(n_assets=len(symbols), state_dim=18)

    # 더미 학습 데이터로 T-Cell 훈련
    dummy_features = np.random.randn(200, 12)
    immune_system.fit_tcell(dummy_features)

    print(f"환경 설정 완료: {len(symbols)}개 자산, {env.max_steps} 스텝")

    # 2. 충분한 경험 수집 (75 스텝 - 배치 크기보다 많이)
    print(f"\n[2] 충분한 경험 수집 (75 스텝)...")

    state = env.reset()
    total_reward = 0
    bcell_usage = {}

    for step in range(75):
        # 의사결정
        weights, decision_info = immune_system.decide(state, training=True)

        # 환경 스텝
        next_state, reward, done, info = env.step(weights)
        total_reward += reward

        # B-Cell 사용 통계
        selected = decision_info["selected_bcell"]
        bcell_usage[selected] = bcell_usage.get(selected, 0) + 1

        # 시스템 업데이트 (경험 저장)
        immune_system.update(state, weights, reward, next_state, done)

        if step % 15 == 0:
            print(
                f"  스텝 {step+1:2d}: 보상={reward:+7.4f}, 누적보상={total_reward:+8.4f}, 선택={selected[:4]}"
            )

        state = next_state
        if done:
            print(f"  에피소드 조기 종료 (스텝 {step+1})")
            break

    print(f"\n경험 수집 완료:")
    print(f"  총 스텝: {step+1}")
    print(f"  누적 보상: {total_reward:+8.4f}")
    print(f"  평균 보상: {total_reward/(step+1):+7.4f}")
    print(f"  B-Cell 사용 분포: {bcell_usage}")

    # 3. 각 B-Cell의 버퍼 크기 확인
    print(f"\n[3] B-Cell 버퍼 상태 확인...")

    active_bcells = []
    for bcell_name, bcell in immune_system.bcells.items():
        buffer_size = len(bcell.replay_buffer)
        print(f"  {bcell_name:10s}: {buffer_size:3d}/64 경험 보유")

        if buffer_size >= bcell.batch_size:
            active_bcells.append((bcell_name, bcell))
            print(f"    ✅ 학습 가능")
        else:
            print(f"    ⚠️  경험 부족")

    print(f"\n학습 가능한 B-Cell: {len(active_bcells)}개")

    # 4. 학습 가능한 B-Cell들의 학습 신호 분석
    print(f"\n[4] 학습 신호 상세 분석...")

    training_results = {}

    for bcell_name, bcell in active_bcells:
        print(f"\n  === {bcell_name} B-Cell 학습 신호 ===")

        # 학습 전 손실 기록
        initial_actor_losses = len(bcell.actor_losses)
        initial_critic_losses = len(bcell.critic_losses)
        initial_alpha = bcell.alpha.item()

        print(f"    학습 전 상태:")
        print(f"      Alpha: {initial_alpha:.4f}")
        print(f"      업데이트 횟수: {bcell.update_count}")
        print(f"      버퍼 크기: {len(bcell.replay_buffer)}")

        # 10회 연속 업데이트
        print(f"    10회 업데이트 실행중...")

        actor_losses = []
        critic_losses = []
        alpha_values = []

        for update_i in range(10):
            # 손실 기록을 위한 업데이트 전 값
            pre_actor_count = len(bcell.actor_losses)
            pre_critic_count = len(bcell.critic_losses)
            pre_alpha = bcell.alpha.item()

            # 업데이트 실행
            bcell.update()

            # 손실 값 수집
            if len(bcell.actor_losses) > pre_actor_count:
                actor_losses.append(bcell.actor_losses[-1])
            if len(bcell.critic_losses) > pre_critic_count:
                critic_losses.append(bcell.critic_losses[-1])

            alpha_values.append(bcell.alpha.item())

        # 학습 후 분석
        if actor_losses and critic_losses:
            avg_actor_loss = np.mean(actor_losses)
            avg_critic_loss = np.mean(critic_losses)
            final_alpha = bcell.alpha.item()
            alpha_change = final_alpha - initial_alpha

            print(f"    학습 후 분석:")
            print(f"      평균 Actor Loss: {avg_actor_loss:.6f}")
            print(f"      평균 Critic Loss: {avg_critic_loss:.6f}")
            print(f"      최종 Alpha: {final_alpha:.4f} (변화: {alpha_change:+.4f})")
            print(f"      총 업데이트: {bcell.update_count}")

            # 손실 안정성 체크
            actor_loss_std = np.std(actor_losses) if len(actor_losses) > 1 else 0
            critic_loss_std = np.std(critic_losses) if len(critic_losses) > 1 else 0

            print(f"      Actor Loss 변동: {actor_loss_std:.6f}")
            print(f"      Critic Loss 변동: {critic_loss_std:.6f}")

            # 문제 진단
            issues = []
            if np.isnan(avg_actor_loss) or np.isnan(avg_critic_loss):
                issues.append("NaN 손실 발생")
            if avg_actor_loss > 1000:
                issues.append("Actor 손실 과도함")
            if avg_critic_loss > 1000:
                issues.append("Critic 손실 과도함")
            if actor_loss_std > avg_actor_loss:
                issues.append("Actor 손실 불안정")
            if critic_loss_std > avg_critic_loss:
                issues.append("Critic 손실 불안정")
            if abs(alpha_change) < 1e-4:
                issues.append("Alpha 학습 정체")

            if issues:
                print(f"      ⚠️  발견된 문제:")
                for issue in issues:
                    print(f"        - {issue}")
            else:
                print(f"      ✅ 학습 신호 양호")

            training_results[bcell_name] = {
                "avg_actor_loss": avg_actor_loss,
                "avg_critic_loss": avg_critic_loss,
                "final_alpha": final_alpha,
                "alpha_change": alpha_change,
                "actor_loss_std": actor_loss_std,
                "critic_loss_std": critic_loss_std,
                "issues": issues,
            }
        else:
            print(f"      ⚠️  손실 데이터 수집 실패")

    # 5. 종합 진단
    print(f"\n[5] 종합 진단...")

    total_issues = []
    healthy_bcells = 0

    for bcell_name, results in training_results.items():
        if not results["issues"]:
            healthy_bcells += 1
        else:
            total_issues.extend(
                [f"{bcell_name}: {issue}" for issue in results["issues"]]
            )

    print(f"  정상 B-Cell: {healthy_bcells}/{len(training_results)}")
    print(f"  총 문제점: {len(total_issues)}개")

    if total_issues:
        print(f"  발견된 문제점들:")
        for i, issue in enumerate(total_issues[:10], 1):  # 최대 10개만 표시
            print(f"    {i}. {issue}")
        if len(total_issues) > 10:
            print(f"    ... 외 {len(total_issues)-10}개")
    else:
        print(f"  ✅ 모든 B-Cell이 정상적으로 학습 중")

    print("\n" + "=" * 80)
    print("확장된 학습 신호 디버깅 완료")
    print("=" * 80)

    return {
        "training_results": training_results,
        "total_issues": total_issues,
        "bcell_usage": bcell_usage,
        "total_reward": total_reward,
        "steps_completed": step + 1,
    }


def recommend_improvements(debug_results):
    """구체적 개선 방안 제안"""

    print("\n" + "=" * 60)
    print("구체적 개선 방안")
    print("=" * 60)

    training_results = debug_results["training_results"]
    total_issues = debug_results["total_issues"]

    recommendations = []

    # 1. 경험 부족 문제
    if len(training_results) < 3:
        recommendations.append(
            {
                "category": "데이터 수집",
                "issue": "충분한 경험 부족",
                "solution": "초기 에피소드 길이 증가 (252 → 500 스텝)",
                "priority": "HIGH",
            }
        )

    # 2. 손실 관련 문제
    nan_issues = [issue for issue in total_issues if "NaN" in issue]
    if nan_issues:
        recommendations.append(
            {
                "category": "수치 안정성",
                "issue": "NaN 손실 발생",
                "solution": "Gradient clipping 강화, 학습률 감소 (1e-4 → 5e-5)",
                "priority": "CRITICAL",
            }
        )

    # 3. 불안정성 문제
    unstable_issues = [issue for issue in total_issues if "불안정" in issue]
    if unstable_issues:
        recommendations.append(
            {
                "category": "학습 안정성",
                "issue": "손실 불안정",
                "solution": "Target network 업데이트 빈도 감소 (TAU: 0.005 → 0.001)",
                "priority": "MEDIUM",
            }
        )

    # 4. Alpha 학습 정체
    alpha_issues = [issue for issue in total_issues if "Alpha" in issue]
    if alpha_issues:
        recommendations.append(
            {
                "category": "탐험 전략",
                "issue": "Alpha 자동 조정 정체",
                "solution": "Alpha 학습률 증가 (1e-4 → 3e-4), Target entropy 조정",
                "priority": "MEDIUM",
            }
        )

    # 5. 보상 문제
    avg_reward = debug_results["total_reward"] / debug_results["steps_completed"]
    if abs(avg_reward) < 1e-4:
        recommendations.append(
            {
                "category": "보상 함수",
                "issue": "보상 신호 미약",
                "solution": "보상 스케일링, Sharpe window 감소 (20 → 10)",
                "priority": "HIGH",
            }
        )

    # 출력
    if recommendations:
        print(f"우선순위별 개선 방안:")

        for priority in ["CRITICAL", "HIGH", "MEDIUM"]:
            priority_recs = [r for r in recommendations if r["priority"] == priority]
            if priority_recs:
                print(f"\n{priority} 우선순위:")
                for i, rec in enumerate(priority_recs, 1):
                    print(f"  {i}. [{rec['category']}] {rec['issue']}")
                    print(f"     해결책: {rec['solution']}")
    else:
        print("✅ 추가 개선이 필요하지 않습니다.")

    # 구체적 설정값 제안
    print(f"\n권장 설정값 변경:")
    print(f"  config.py 수정:")
    print(f"    LR_ACTOR = 3e-4  # 현재: {LR_ACTOR}")
    print(f"    LR_CRITIC = 3e-4  # 현재: {LR_CRITIC}")
    print(f"    ALPHA_LR = 3e-4  # 현재: {ALPHA_LR}")
    print(f"    TAU = 0.001  # 현재: {TAU}")
    print(f"    BATCH_SIZE = 32  # 현재: {BATCH_SIZE} (더 빠른 업데이트)")

    if avg_reward < 1e-4:
        print(f"  environment.py 수정:")
        print(f"    sharpe_window = 10  # 현재: 20")
        print(f"    보상 스케일링 추가")


if __name__ == "__main__":
    try:
        debug_results = debug_training_with_sufficient_data()
        recommend_improvements(debug_results)

        print(f"\n🎯 핵심 발견사항:")
        print(f"   1. 배치 크기 vs 경험 부족 문제 해결됨")
        print(f"   2. 실제 학습 신호 분석 완료")
        print(f"   3. 구체적 개선 방안 도출됨")

    except Exception as e:
        print(f"\n❌ 확장된 학습 디버깅 실행 중 오류: {e}")
        import traceback

        traceback.print_exc()
