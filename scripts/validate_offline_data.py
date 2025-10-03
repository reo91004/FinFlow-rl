# scripts/validate_offline_data.py

"""
오프라인 데이터 품질 검증 스크립트

목적: BC warm-start를 위한 데이터 품질 확인
의존성: numpy, matplotlib, sklearn
사용처: BC 리팩터링 Phase 0

검증 메트릭:
- Action entropy > 2.0 (다양성)
- State coverage (PCA)
- Return distribution (극단값 체크)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import argparse


def analyze_offline_data(data_path: str, output_path: str = 'offline_data_quality.png'):
    """
    오프라인 데이터 품질 분석 및 시각화

    Args:
        data_path: NPZ 파일 경로
        output_path: 시각화 저장 경로

    Returns:
        품질 검증 통과 여부 (bool)
    """
    print("=" * 60)
    print("Offline Data Quality Report")
    print("=" * 60)

    # 데이터 로드
    if not Path(data_path).exists():
        print(f"오류: 데이터 파일이 없습니다: {data_path}")
        print("\n재수집 방법:")
        print("  python main.py --mode train --config configs/default_irt.yaml")
        print("  (첫 실행 시 자동으로 오프라인 데이터 수집)")
        return False

    data = np.load(data_path)

    states = data['states']      # [N, state_dim]
    actions = data['actions']    # [N, action_dim]
    rewards = data['rewards']    # [N]

    N, state_dim = states.shape
    action_dim = actions.shape[1]

    print(f"샘플 수: {N:,}")
    print(f"State 차원: {state_dim}")
    print(f"Action 차원: {action_dim}")

    # ===== 1. Action Diversity (Entropy) =====
    # Dirichlet 분포의 엔트로피: -Σ p_i log(p_i)
    action_entropy_per_sample = -np.sum(
        actions * np.log(actions + 1e-8), axis=1
    )
    mean_entropy = action_entropy_per_sample.mean()
    max_entropy = np.log(action_dim)

    print(f"\nAction Entropy:")
    print(f"  평균: {mean_entropy:.3f} / {max_entropy:.3f} (max)")
    print(f"  표준편차: {action_entropy_per_sample.std():.3f}")

    # 품질 기준: 평균 엔트로피 > 2.0 (30자산 기준 균등 분포는 3.4)
    if mean_entropy < 2.0:
        print("  ⚠️  WARNING: 낮은 다양성 (< 2.0)")
        print("      → BC가 균등 분포를 학습할 위험")
        quality_pass = False
    else:
        print("  ✓  다양성 양호")
        quality_pass = True

    # ===== 2. Return Statistics =====
    print(f"\nReturn 통계:")
    print(f"  평균: {rewards.mean():.4f}")
    print(f"  표준편차: {rewards.std():.4f}")
    print(f"  최소: {rewards.min():.4f}")
    print(f"  최대: {rewards.max():.4f}")
    print(f"  중앙값: {np.median(rewards):.4f}")

    # 극단값 체크
    n_extreme = np.sum((rewards < -5.0) | (rewards > 5.0))
    if n_extreme > N * 0.01:
        print(f"  ⚠️  WARNING: 극단값 {n_extreme}개 ({n_extreme/N*100:.2f}%)")
        print("      → 보상 스케일링 확인 필요")
    else:
        print("  ✓  극단값 없음")

    # ===== 3. Action Distribution (Top Assets) =====
    mean_weights = actions.mean(axis=0)
    top_10_indices = np.argsort(mean_weights)[-10:]
    top_10_weights = mean_weights[top_10_indices]

    print(f"\nTop 10 자산 (평균 가중치):")
    for idx, weight in zip(top_10_indices, top_10_weights):
        print(f"  Asset {idx}: {weight:.4f}")

    # 편향 체크 (특정 자산에 과도하게 집중)
    max_weight = mean_weights.max()
    if max_weight > 0.2:
        print(f"  ⚠️  WARNING: 최대 가중치 {max_weight:.4f} > 0.2")
        print("      → 특정 자산 편향 가능성")
    else:
        print("  ✓  균형잡힌 분포")

    # ===== 4. State Coverage (PCA) =====
    print(f"\nState Coverage (PCA):")
    if N > 1000:
        # 샘플링 (시각화용)
        sample_idx = np.random.choice(N, 10000, replace=False)
        states_sampled = states[sample_idx]
    else:
        states_sampled = states

    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(states_sampled)

    explained_var = pca.explained_variance_ratio_
    print(f"  PC1: {explained_var[0]*100:.2f}% 설명력")
    print(f"  PC2: {explained_var[1]*100:.2f}% 설명력")
    print(f"  누적: {explained_var.sum()*100:.2f}%")

    # ===== 5. Visualization =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (0, 0): Action Entropy Distribution
    axes[0, 0].hist(action_entropy_per_sample, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(2.0, color='red', linestyle='--', linewidth=2, label='Threshold (2.0)')
    axes[0, 0].axvline(mean_entropy, color='green', linestyle='-', linewidth=2, label=f'Mean ({mean_entropy:.2f})')
    axes[0, 0].set_title('Action Entropy Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Entropy')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # (0, 1): Top Assets by Average Weight
    axes[0, 1].barh(range(10), top_10_weights, color='steelblue', edgecolor='black')
    axes[0, 1].set_yticks(range(10))
    axes[0, 1].set_yticklabels([f'Asset {i}' for i in top_10_indices])
    axes[0, 1].set_title('Top 10 Assets by Avg Weight', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Average Weight')
    axes[0, 1].grid(axis='x', alpha=0.3)

    # (1, 0): Return Distribution
    axes[1, 0].hist(rewards, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].axvline(rewards.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean ({rewards.mean():.2f})')
    axes[1, 0].axvline(np.median(rewards), color='green', linestyle='-', linewidth=2, label=f'Median ({np.median(rewards):.2f})')
    axes[1, 0].set_title('Return Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # (1, 1): State Coverage (PCA)
    scatter = axes[1, 1].scatter(
        states_2d[:, 0], states_2d[:, 1],
        c=rewards[:len(states_2d)],
        cmap='RdYlGn',
        alpha=0.3,
        s=5
    )
    axes[1, 1].set_title('State Coverage (PCA)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
    axes[1, 1].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
    axes[1, 1].grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Reward', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓  시각화 저장 완료: {output_path}")

    # ===== 6. Final Report =====
    print("\n" + "=" * 60)
    print("최종 판정")
    print("=" * 60)

    if quality_pass and N >= 50000:
        print("✅ 데이터 품질 양호 - BC warm-start 가능")
        return True
    elif N < 50000:
        print(f"⚠️  샘플 부족: {N} < 50,000")
        print("   → 오프라인 데이터 재수집 권장")
        return False
    else:
        print("⚠️  품질 이슈 발견 - 재수집 권장")
        print("   → Action entropy 또는 분포 편향 확인")
        return False


def main():
    parser = argparse.ArgumentParser(description='오프라인 데이터 품질 검증')
    parser.add_argument(
        '--data',
        type=str,
        default='data/offline_data.npz',
        help='NPZ 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='offline_data_quality.png',
        help='시각화 저장 경로'
    )

    args = parser.parse_args()

    is_valid = analyze_offline_data(args.data, args.output)

    if not is_valid:
        print("\n권장 조치:")
        print("  1. 오프라인 데이터 재수집:")
        print("     python main.py --mode train --config configs/default_irt.yaml")
        print("  2. 데이터 수집 설정 확인:")
        print("     - OfflineDataset.collect_from_env() 파라미터")
        print("     - 전략 다양성 (momentum, mean_reversion, risk_parity)")
        exit(1)
    else:
        print("\n다음 단계:")
        print("  python main.py --mode train --config configs/default_irt.yaml")


if __name__ == '__main__':
    main()
