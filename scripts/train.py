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

from src.utils.seed import set_seed, DeviceManager
from src.utils.logger import FinFlowLogger, get_session_directory
from src.data.loader import DataLoader
from src.data.loader import FeatureExtractor
from src.core.env import PortfolioEnv
from src.core.objectives import PortfolioObjective, RewardNormalizer
from src.core.trainer import FinFlowTrainer, TrainingConfig

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
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'iql', 'sac'],
                       help='Training mode: full (IQL+SAC), iql (only), or sac (only)')
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
        
        # TrainingConfig 생성
        training_config = TrainingConfig(
            # Environment
            env_config={
                'initial_balance': config['env']['initial_capital'],
                'transaction_cost': config['env']['turnover_cost'],
                'max_leverage': config['env']['max_leverage'],
                'window_size': config['features']['window']
            },
            # Data
            data_config={
                'tickers': config['data']['symbols'],
                'period': '5y',  # 5년 데이터
                'interval': '1d',
                'auto_download': True,
                'use_cache': True
            },
            # IQL
            iql_epochs=config['train']['offline_steps'] // 1000,
            iql_batch_size=config['train']['offline_batch_size'],
            iql_expectile=config['bcell']['iql_expectile'],
            iql_temperature=config['bcell']['iql_temperature'],
            # SAC
            sac_episodes=config['train']['online_steps'] // 100,
            sac_batch_size=config['train']['online_batch_size'],
            sac_gamma=config['bcell']['gamma'],
            sac_tau=config['bcell']['tau'],
            sac_alpha=config['bcell']['alpha_init'],
            sac_cql_weight=config['bcell']['cql_alpha_start'],
            # Memory
            memory_capacity=config['train']['buffer_size'],
            memory_k_neighbors=config['memory']['k_neighbors'],
            # Monitoring
            eval_interval=config['train']['eval_interval'] // 100,
            checkpoint_interval=config['train']['save_interval'] // 100,
            log_interval=config['train']['log_interval'] // 100,
            # Target metrics
            target_sharpe=config['objectives']['sharpe_beta'] * 1.5,
            target_cvar=config['objectives']['cvar_target'],
            # Device & Seed
            device=config['device'],
            seed=config['seed'],
            # Paths
            data_path='data/processed',
            checkpoint_dir='checkpoints'
        )
        
        # Trainer 생성
        trainer = FinFlowTrainer(training_config)
        
        # 체크포인트 로드 (있을 경우)
        if args.resume:
            trainer.load_checkpoint(args.resume)
            logger.info(f"체크포인트 로드: {args.resume}")
        
        # 학습 실행
        if args.mode == 'full':
            trainer.train()  # IQL + SAC
        elif args.mode == 'iql':
            trainer._pretrain_iql()  # IQL만
        elif args.mode == 'sac':
            trainer._train_sac()  # SAC만
        
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
    logger.info("  python scripts/train.py --use-trainer --mode iql  # IQL만")
    logger.info("  python scripts/train.py --use-trainer --mode sac  # SAC만")
    return
    
    # 더 이상 랜덤 정책 테스트를 지원하지 않음
    # 실제 강화학습은 Trainer를 통해서만 수행

if __name__ == "__main__":
    main()