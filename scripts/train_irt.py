# scripts/train_irt.py

"""
IRT 학습 스크립트

사용법:
python scripts/train_irt.py --config configs/default_irt.yaml
"""

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path

from src.training.trainer_irt import TrainerIRT
from src.utils.logger import FinFlowLogger

def set_seed(seed: int):
    """재현성을 위한 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='IRT Training Script')
    parser.add_argument('--config', type=str, default='configs/default_irt.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 시드 설정
    seed = config.get('seed', 42)
    set_seed(seed)

    logger = FinFlowLogger("Main")
    logger.info(f"설정 파일 로드: {args.config}")
    logger.info(f"시드: {seed}")

    # 트레이너 생성 및 학습
    trainer = TrainerIRT(config)
    best_model = trainer.train()

    logger.info("학습 완료!")

if __name__ == '__main__':
    main()