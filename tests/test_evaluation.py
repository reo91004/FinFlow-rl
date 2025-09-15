#!/usr/bin/env python
"""평가 스크립트 테스트"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from scripts.evaluate import FinFlowEvaluator

# 변환된 체크포인트로 평가기 생성
evaluator = FinFlowEvaluator(
    checkpoint_path="logs/20250913_015247/models/checkpoint_best_migrated.pt",
    data_path="data/test",
    device="cpu"
)

print("✓ Evaluator created successfully")
print(f"Model loaded: {evaluator.b_cell.specialization}")
print(f"Memory size: {len(evaluator.memory_cell.memories)}")
print(f"Device: {evaluator.device}")