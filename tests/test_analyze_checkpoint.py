import torch
import numpy as np

# 체크포인트 로드
checkpoint = torch.load('logs/20250913_015247/models/checkpoint_best.pt', map_location='cpu', weights_only=False)

print("=== Checkpoint Keys ===")
print(list(checkpoint.keys()))

print("\n=== B-Cell Structure ===")
if 'b_cell' in checkpoint:
    print("b_cell keys:", list(checkpoint['b_cell'].keys()))
    print("\nTypes in b_cell:")
    for k, v in checkpoint['b_cell'].items():
        if isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} keys")
            # 첫 몇 개 키만 출력
            for subk in list(v.keys())[:3]:
                print(f"    - {subk}: {type(v[subk])}")
        else:
            print(f"  {k}: {type(v)}")
            if k == 'log_alpha':
                print(f"    Value: {v}")
            elif k == 'specialization':
                print(f"    Value: {v}")

print("\n=== Memory Cell Structure ===")
if 'memory_cell' in checkpoint:
    mem = checkpoint['memory_cell']
    print(f"memory_cell keys: {list(mem.keys())}")
    if 'memories' in mem:
        print(f"  memories: list of {len(mem['memories'])} items")
        if mem['memories']:
            first_mem = mem['memories'][0]
            print("  First memory item keys:", list(first_mem.keys()))
            for k, v in first_mem.items():
                print(f"    - {k}: {type(v)} shape={getattr(v, 'shape', 'N/A')}")

print("\n=== Problematic Types ===")
# state_dict 내의 모든 타입 확인
def check_types(obj, path=""):
    problems = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            problems.extend(check_types(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            problems.extend(check_types(v, f"{path}[{i}]"))
    elif isinstance(obj, np.ndarray):
        problems.append(f"{path}: numpy.ndarray")
    elif isinstance(obj, (np.integer, np.floating)):
        problems.append(f"{path}: numpy scalar ({type(obj).__name__})")
    elif isinstance(obj, str) and path.endswith('.state_dict'):
        # state_dict 내부에 문자열이 있는 경우
        problems.append(f"{path}: string in state_dict")
    return problems

problems = check_types(checkpoint)
if problems:
    print("Found problematic types:")
    for p in problems[:20]:  # 처음 20개만
        print(f"  - {p}")
else:
    print("No numpy arrays or scalars found in checkpoint")