# scripts/migrate_checkpoint.py

import torch
import numpy as np
import argparse
from pathlib import Path
import logging
import json
import copy


def setup_logger():
    """로거 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("CheckpointMigration")


def convert_state_dict(state_dict):
    """state_dict 내부의 모든 numpy 배열을 텐서로 변환"""
    converted = {}
    for key, value in state_dict.items():
        if isinstance(value, np.ndarray):
            converted[key] = torch.from_numpy(value.copy())
        elif isinstance(value, dict):
            converted[key] = convert_state_dict(value)
        elif isinstance(value, (list, tuple)):
            converted[key] = type(value)(
                torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v for v in value
            )
        else:
            converted[key] = value
    return converted


def deep_convert_to_tensor(obj, path="root"):
    """재귀적으로 모든 numpy 객체를 torch tensor로 변환"""
    logger = logging.getLogger("CheckpointMigration")

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            result[k] = deep_convert_to_tensor(v, f"{path}.{k}")
        return result
    elif isinstance(obj, list):
        return [deep_convert_to_tensor(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    elif isinstance(obj, tuple):
        return tuple(deep_convert_to_tensor(v, f"{path}[{i}]") for i, v in enumerate(obj))
    elif isinstance(obj, np.ndarray):
        return torch.from_numpy(obj.copy()).float()
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, torch.Tensor):
        return obj
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    else:
        logger.debug(f"Unknown type at {path}: {type(obj)}")
        return obj


def migrate_checkpoint(input_path: str, output_path: str = None) -> None:
    """기존 체크포인트를 새 형식으로 변환"""
    logger = setup_logger()

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_migrated{input_path.suffix}"
    else:
        output_path = Path(output_path)

    logger.info(f"체크포인트 로드 중: {input_path}")

    # 기존 체크포인트 로드
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    # 원본 체크포인트 키 확인
    logger.info(f"원본 체크포인트 키: {list(checkpoint.keys())}")

    # 새 체크포인트 생성
    new_checkpoint = {}

    # 1. B-Cell state_dict 변환 (메타데이터 분리)
    if "b_cell" in checkpoint:
        logger.info("B-Cell state_dict 변환 중...")
        b_cell_data = checkpoint["b_cell"]

        # B-Cell은 state_dict가 아니라 복합 구조
        new_b_cell = {}

        # 실제 state_dict들 변환
        if "actor" in b_cell_data:
            new_b_cell["actor"] = convert_state_dict(b_cell_data["actor"])
        if "critic_q1" in b_cell_data:
            new_b_cell["critic_q1"] = convert_state_dict(b_cell_data["critic_q1"])
        if "critic_q2" in b_cell_data:
            new_b_cell["critic_q2"] = convert_state_dict(b_cell_data["critic_q2"])

        # 메타데이터 처리
        if "log_alpha" in b_cell_data:
            # numpy array를 tensor로 변환
            if isinstance(b_cell_data["log_alpha"], np.ndarray):
                new_b_cell["log_alpha"] = torch.from_numpy(b_cell_data["log_alpha"].copy()).float()
            else:
                new_b_cell["log_alpha"] = b_cell_data["log_alpha"]

        # 문자열과 숫자 메타데이터는 그대로 유지
        for key in ["specialization", "training_step", "performance_score"]:
            if key in b_cell_data:
                new_b_cell[key] = b_cell_data[key]

        new_checkpoint["b_cell"] = new_b_cell

    # 2. Gating Network state_dict 변환
    if "gating_network" in checkpoint:
        logger.info("Gating Network state_dict 변환 중...")
        new_checkpoint["gating_network"] = convert_state_dict(checkpoint["gating_network"])

    # 3. T-Cell state 변환
    if "t_cell" in checkpoint:
        logger.info("T-Cell state 변환 중...")
        new_checkpoint["t_cell"] = deep_convert_to_tensor(checkpoint["t_cell"], "t_cell")

    # 4. Device 수정
    device_str = str(checkpoint.get("device", "cpu"))
    if "auto" in device_str.lower():
        if torch.cuda.is_available():
            device_str = f"cuda:{torch.cuda.current_device()}"
        else:
            device_str = "cpu"
        logger.info(f"Device 수정: {checkpoint.get('device')} -> {device_str}")
    new_checkpoint["device"] = device_str

    # 5. Memory cell 변환
    if "memory_cell" in checkpoint:
        logger.info("메모리 셀 변환 중...")
        new_memory_cell = {}

        if "memories" in checkpoint["memory_cell"]:
            memories = checkpoint["memory_cell"]["memories"]
            converted_memories = []

            logger.info(f"메모리 변환 중 ({len(memories)}개)...")

            for i, memory in enumerate(memories):
                converted_memory = deep_convert_to_tensor(memory, f"memory[{i}]")
                converted_memories.append(converted_memory)

                if (i + 1) % 10000 == 0:
                    logger.info(f"  진행: {i + 1}/{len(memories)}")

            new_memory_cell["memories"] = converted_memories

        # stats 변환
        if "stats" in checkpoint["memory_cell"]:
            new_memory_cell["stats"] = deep_convert_to_tensor(
                checkpoint["memory_cell"]["stats"], "memory_cell.stats"
            )

        new_checkpoint["memory_cell"] = new_memory_cell

    # 6. Config 변환
    if "config" in checkpoint:
        logger.info("Config 변환 중...")
        new_checkpoint["config"] = deep_convert_to_tensor(checkpoint["config"], "config")
    elif "config_params" in checkpoint:
        logger.info("Config params 변환 중...")
        new_checkpoint["config_params"] = deep_convert_to_tensor(
            checkpoint["config_params"], "config_params"
        )

    # 7. Metrics 변환
    if "metrics" in checkpoint:
        logger.info("Metrics 변환 중...")
        new_checkpoint["metrics"] = deep_convert_to_tensor(checkpoint["metrics"], "metrics")

    # 8. 기타 필드 변환
    other_fields = [
        "episode",
        "global_step",
        "stability_report",
        "metadata",
        "best_sharpe",
        "epoch",
        "best_reward",
        "timestamp",
        "current_epoch",
    ]

    for key in other_fields:
        if key in checkpoint:
            value = checkpoint[key]
            if isinstance(value, (np.ndarray, np.number)):
                new_checkpoint[key] = float(value.item() if hasattr(value, "item") else value)
            elif isinstance(value, torch.Tensor):
                new_checkpoint[key] = float(value.item())
            elif key in ["episode", "global_step", "epoch", "current_epoch"]:
                new_checkpoint[key] = int(value) if not isinstance(value, int) else value
            else:
                new_checkpoint[key] = deep_convert_to_tensor(value, key)

    # 저장
    logger.info(f"변환된 체크포인트 저장 중: {output_path}")
    torch.save(new_checkpoint, output_path)

    # 검증: weights_only=True로 로드 테스트
    logger.info("검증 중...")
    try:
        test_checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)

        # 주요 필드 검증
        assert (
            "b_cell" in test_checkpoint or "model_state_dict" in test_checkpoint
        ), "모델 데이터 누락"
        assert "device" in test_checkpoint, "device 누락"
        assert (
            test_checkpoint["device"] != "auto"
        ), f"device가 여전히 auto: {test_checkpoint['device']}"

        # 메모리 검증
        if "memory_cell" in test_checkpoint:
            memories = test_checkpoint["memory_cell"].get("memories", [])
            if memories:
                first_memory = memories[0]
                for key in ["state", "action", "reward"]:
                    if key in first_memory:
                        value_type = type(first_memory[key])
                        assert isinstance(
                            first_memory[key], (torch.Tensor, float, int, bool)
                        ), f"메모리 필드 {key}가 잘못된 타입: {value_type}"

        # B-Cell state_dict 검증
        if "b_cell" in test_checkpoint:
            b_cell = test_checkpoint["b_cell"]
            # state_dict들만 검증
            for key in ["actor", "critic_q1", "critic_q2"]:
                if key in b_cell:
                    assert isinstance(b_cell[key], dict), f"b_cell.{key}가 dict가 아님: {type(b_cell[key])}"
                    for param_key, param_value in b_cell[key].items():
                        assert isinstance(
                            param_value, torch.Tensor
                        ), f"b_cell.{key}.{param_key}가 텐서가 아님: {type(param_value)}"

            # 메타데이터 타입 검증
            if "specialization" in b_cell:
                assert isinstance(b_cell["specialization"], str), "specialization이 문자열이 아님"
            if "training_step" in b_cell:
                assert isinstance(b_cell["training_step"], int), "training_step이 정수가 아님"
            if "performance_score" in b_cell:
                assert isinstance(b_cell["performance_score"], (float, int)), "performance_score가 숫자가 아님"
            if "log_alpha" in b_cell:
                assert isinstance(b_cell["log_alpha"], torch.Tensor), "log_alpha가 텐서가 아님"

        logger.info("✓ 변환 성공!")
        logger.info(f"  - Episode: {test_checkpoint.get('episode', 'N/A')}")
        logger.info(f"  - Global Step: {test_checkpoint.get('global_step', 'N/A')}")
        logger.info(f"  - Device: {test_checkpoint.get('device', 'N/A')}")
        logger.info(
            f"  - Memory size: {len(test_checkpoint.get('memory_cell', {}).get('memories', []))}"
        )

    except Exception as e:
        logger.error(f"✗ 검증 실패: {e}")
        logger.error("변환된 체크포인트를 weights_only=True로 로드할 수 없습니다.")

        # 디버깅 정보
        logger.info("\n디버깅 정보:")
        try:
            debug_checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
            for key in debug_checkpoint.keys():
                logger.info(f"  {key}: {type(debug_checkpoint[key])}")
        except:
            pass

        raise


def main():
    parser = argparse.ArgumentParser(description="FinFlow 체크포인트 migration")
    parser.add_argument("input", type=str, help="기존 체크포인트 경로")
    parser.add_argument("--output", type=str, default=None, help="변환된 체크포인트 저장 경로")
    parser.add_argument("--batch", action="store_true", help="디렉토리 내 모든 체크포인트 변환")
    parser.add_argument("--force", action="store_true", help="이미 변환된 파일도 재변환")

    args = parser.parse_args()

    if args.batch:
        input_dir = Path(args.input)
        checkpoint_files = list(input_dir.glob("*.pt"))

        logger = setup_logger()
        logger.info(f"{len(checkpoint_files)}개 체크포인트 발견")

        success_count = 0
        fail_count = 0

        for cp_file in checkpoint_files:
            if not args.force and "_migrated" in cp_file.stem:
                logger.info(f"건너뜀 (이미 변환됨): {cp_file.name}")
                continue

            logger.info(f"\n처리 중: {cp_file.name}")
            try:
                migrate_checkpoint(cp_file)
                success_count += 1
            except Exception as e:
                logger.error(f"실패: {cp_file.name} - {e}")
                fail_count += 1

        logger.info(f"\n완료: 성공 {success_count}, 실패 {fail_count}")
    else:
        migrate_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
