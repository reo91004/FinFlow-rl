#!/usr/bin/env python3
# scripts/train.py

"""
FinFlow-RL 메인 학습 스크립트
IQL 사전학습 → Distributional SAC 온라인 학습
"""

import argparse
import yaml
import os
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.device_manager import set_seed, DeviceManager
from src.utils.logger import FinFlowLogger, get_session_directory
from src.data import DataLoader, FeatureExtractor
from src.environments.portfolio_env import PortfolioEnv
from src.environments.reward_functions import PortfolioObjective, RewardNormalizer
from src.training.trainer import FinFlowTrainer

def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='FinFlow-RL Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with verbose output')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'offline', 'online'],
                       help='Training mode: full (Offline+Online), offline (IQL/TD3BC), or online (REDQ/TQC)')
    parser.add_argument('--use-trainer', action='store_true', default=True,
                       help='Use integrated trainer (default: True)')
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 통합 Trainer 사용 (권장)
    if args.use_trainer:
        logger = FinFlowLogger("Main")
        logger.info("=" * 80)
        logger.info("FinFlow-RL (BIPD 2.0) 통합 학습 시작")
        logger.info("=" * 80)
        logger.info(f"설정 파일: {args.config}")
        logger.info(f"학습 모드: {args.mode}")
        
        # Trainer 생성 (config를 직접 전달)
        trainer = FinFlowTrainer(config)
        
        # 체크포인트 로드 (있을 경우)
        if args.resume:
            trainer.load_checkpoint(args.resume)
            logger.info(f"체크포인트 로드: {args.resume}")
        
        # 학습 실행
        if args.mode == 'full':
            trainer.train()  # IQL + Online (REDQ/TQC)
        elif args.mode == 'offline':
            trainer._offline_pretrain()  # 오프라인 학습만 (IQL/TD3BC)
        elif args.mode == 'online':
            trainer._online_finetune()  # Online 학습만 (REDQ/TQC)
        
        logger.info("\n🎉 FinFlow-RL 학습이 완료되었습니다!")
        return
    
    # Trainer를 사용하지 않는 경우 경고
    logger = FinFlowLogger("Training")
    logger.error("=" * 80)
    logger.error("경고: 통합 Trainer를 사용하지 않고 있습니다!")
    logger.error("--use-trainer 옵션을 사용하여 실제 강화학습을 실행하세요.")
    logger.error("=" * 80)
    logger.info("\n사용법:")
    logger.info("  python scripts/train.py --use-trainer --mode full")
    logger.info("  python scripts/train.py --use-trainer --mode offline  # 오프라인 학습 (IQL/TD3BC)")
    logger.info("  python scripts/train.py --use-trainer --mode online  # Online (REDQ/TQC)")
    return
    
    # 더 이상 랜덤 정책 테스트를 지원하지 않음
    # 실제 강화학습은 Trainer를 통해서만 수행

if __name__ == "__main__":
    main()