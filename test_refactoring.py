# test_refactoring.py

import sys
import torch
import numpy as np
from pathlib import Path

print("FinFlow-RL 리팩토링 테스트")
print("="*60)

# 1. Import 테스트
print("\n1. Import 테스트...")
try:
    from src.agents.b_cell import BCell
    print("✓ B-Cell import 성공")
except Exception as e:
    print(f"✗ B-Cell import 실패: {e}")

try:
    from src.agents.t_cell import TCell
    print("✓ T-Cell import 성공")
except Exception as e:
    print(f"✗ T-Cell import 실패: {e}")

try:
    from src.agents.memory import MemoryCell
    print("✓ Memory Cell import 성공")
except Exception as e:
    print(f"✗ Memory Cell import 실패: {e}")

try:
    from src.core.offline_trainer import OfflineTrainer
    print("✓ Offline Trainer import 성공")
except Exception as e:
    print(f"✗ Offline Trainer import 실패: {e}")

try:
    from src.core.td3bc import TD3BCAgent
    print("✓ TD3+BC import 성공")
except Exception as e:
    print(f"✗ TD3+BC import 실패: {e}")

try:
    from src.core.trainer import FinFlowTrainer
    print("✓ FinFlow Trainer import 성공")
except Exception as e:
    print(f"✗ FinFlow Trainer import 실패: {e}")

try:
    from src.experiments.ablation import AblationStudy
    print("✓ Ablation Study import 성공")
except Exception as e:
    print(f"✗ Ablation Study import 실패: {e}")

try:
    from src.baselines.equal_weight import EqualWeightStrategy
    print("✓ Equal Weight Strategy import 성공")
except Exception as e:
    print(f"✗ Equal Weight Strategy import 실패: {e}")

try:
    from src.baselines.standard_sac import StandardSAC
    print("✓ Standard SAC import 성공")
except Exception as e:
    print(f"✗ Standard SAC import 실패: {e}")

# 2. 삭제된 파일 확인
print("\n2. 삭제된 파일 확인...")
deleted_files = [
    'src/agents/gating.py',
    'src/core/distributional.py',
    'scripts/pretrain_iql.py'
]

for file in deleted_files:
    if Path(file).exists():
        print(f"✗ {file} 아직 존재함")
    else:
        print(f"✓ {file} 성공적으로 삭제됨")

# 3. 컴포넌트 초기화 테스트
print("\n3. 컴포넌트 초기화 테스트...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# B-Cell 초기화
try:
    config = {
        'hidden_dims': [128, 128],
        'n_critics': 3,
        'm_sample': 2,
        'utd_ratio': 10,
        'buffer_size': 10000
    }
    b_cell = BCell(state_dim=43, action_dim=5, config=config, device=device)
    print("✓ B-Cell 초기화 성공")
except Exception as e:
    print(f"✗ B-Cell 초기화 실패: {e}")

# T-Cell 초기화
try:
    t_cell = TCell(feature_dim=12, contamination=0.1)
    print("✓ T-Cell 초기화 성공")

    # 더미 데이터로 학습 테스트
    dummy_features = np.random.randn(100, 12)
    t_cell.fit(dummy_features)
    print("✓ T-Cell 학습 성공")

    # 위기 감지 테스트
    test_features = np.random.randn(12)
    crisis_level, explanation = t_cell.detect_crisis(test_features)
    print(f"✓ T-Cell 위기 감지 성공: level={crisis_level:.3f}")
except Exception as e:
    print(f"✗ T-Cell 테스트 실패: {e}")

# Memory Cell 초기화
try:
    memory = MemoryCell(capacity=100, k_neighbors=5)
    print("✓ Memory Cell 초기화 성공")

    # 메모리 저장 테스트
    for i in range(10):
        state = np.random.randn(43)
        action = np.random.dirichlet(np.ones(5))
        reward = np.random.randn()
        crisis = np.random.random()
        memory.store(state, action, reward, crisis)
    print("✓ Memory Cell 저장 성공")
except Exception as e:
    print(f"✗ Memory Cell 테스트 실패: {e}")

# TD3+BC 초기화
try:
    td3bc_config = {
        'hidden_dims': [128, 128],
        'bc_weight': 2.5,
        'policy_delay': 2
    }
    td3bc = TD3BCAgent(state_dim=43, action_dim=5, config=td3bc_config, device=device)
    print("✓ TD3+BC 초기화 성공")
except Exception as e:
    print(f"✗ TD3+BC 초기화 실패: {e}")

print("\n" + "="*60)
print("리팩토링 테스트 완료!")
print("="*60)