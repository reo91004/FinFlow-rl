#!/usr/bin/env python3
# scripts/pretrain_iql.py

"""
IQL 오프라인 사전학습 스크립트
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.utils.seed import set_seed, DeviceManager
from src.utils.logger import FinFlowLogger, get_session_directory
from src.data import DataLoader, FeatureExtractor
from src.core.env import PortfolioEnv
from src.core.replay import OfflineDataset
from src.core.iql import IQLAgent
from src.agents.t_cell import TCell

def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def collect_offline_data(env, n_episodes: int = 100) -> OfflineDataset:
    """
    오프라인 데이터 수집 (랜덤 정책)
    """
    dataset = OfflineDataset(capacity=100000)
    logger = FinFlowLogger("DataCollection")
    
    for episode in tqdm(range(n_episodes), desc="Collecting offline data"):
        state, info = env.reset()
        trajectory = []
        done = False
        truncated = False
        
        while not done and not truncated:
            # Random policy (Dirichlet)
            action = np.random.dirichlet(np.ones(env.n_assets))
            next_state, reward, done, truncated, info = env.step(action)
            
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done or truncated
            })
            
            state = next_state
        
        # Add trajectory to dataset
        from src.core.replay import Transition
        transitions = [Transition(**t) for t in trajectory]
        dataset.add_trajectory(transitions)
    
    logger.info(f"오프라인 데이터 수집 완료: {len(dataset)} transitions")
    return dataset

def main():
    parser = argparse.ArgumentParser(description='IQL Pretraining')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--collect-episodes', type=int, default=100,
                       help='Number of episodes to collect')
    parser.add_argument('--train-steps', type=int, default=200000,
                       help='Number of training steps')
    parser.add_argument('--eval-interval', type=int, default=10000,
                       help='Evaluation interval')
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 시드 설정
    set_seed(config['seed'])
    
    # 디바이스 설정
    device_manager = DeviceManager(config['device'])
    device = device_manager.device
    
    # 로거 초기화
    logger = FinFlowLogger("IQLPretraining")
    session_dir = get_session_directory()
    
    logger.info("=" * 80)
    logger.info("IQL 오프라인 사전학습 시작")
    logger.info("=" * 80)
    logger.info(f"설정 파일: {args.config}")
    logger.info(f"디바이스: {device}")
    logger.info(f"학습 스텝: {args.train_steps}")
    
    # 데이터 로드
    logger.info("시장 데이터를 로드합니다...")
    data_loader = DataLoader(cache_dir=config['data']['cache_dir'])
    market_data = data_loader.get_market_data(
        symbols=config['data']['symbols'],
        train_start=config['data']['start'],
        train_end=config['data']['end'],
        test_start=config['data']['test_start'],
        test_end=config['data']['test_end']
    )
    
    train_data = market_data['train_data']
    
    # 특성 추출기 및 환경 초기화
    feature_extractor = FeatureExtractor(window=config['features']['window'])
    env = PortfolioEnv(
        price_data=train_data,
        feature_extractor=feature_extractor,
        initial_capital=config['env']['initial_capital'],
        transaction_cost=config['env']['turnover_cost'],
        slippage=config['env']['slip_coeff']
    )
    
    # T-Cell 학습 (정상 시장 패턴)
    logger.info("T-Cell을 학습합니다...")
    tcell = TCell(
        feature_dim=12,
        contamination=config['tcell']['contamination'],
        n_estimators=config['tcell']['n_estimators']
    )
    
    # 특성 수집
    features_list = []
    for i in range(len(train_data) - feature_extractor.window):
        features = feature_extractor.extract_features(train_data, current_idx=i + feature_extractor.window)
        features_list.append(features)
    
    tcell.fit(np.array(features_list))
    logger.info("T-Cell 학습 완료")
    
    # 오프라인 데이터 수집 또는 로드
    dataset_path = os.path.join(session_dir, "offline_dataset.pt")
    
    if os.path.exists(dataset_path):
        logger.info(f"기존 데이터셋 로드: {dataset_path}")
        dataset = OfflineDataset()
        dataset.load(dataset_path)
    else:
        logger.info(f"오프라인 데이터 수집 중... ({args.collect_episodes} 에피소드)")
        dataset = collect_offline_data(env, n_episodes=args.collect_episodes)
        dataset.save(dataset_path)
        logger.info(f"데이터셋 저장: {dataset_path}")
    
    # IQL 초기화
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    logger.info(f"IQL 초기화 - State dim: {state_dim}, Action dim: {action_dim}")
    iql = IQL(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config['bcell'],
        device=device
    )
    
    # 학습 루프
    logger.info("IQL 학습 시작...")
    batch_size = config['train']['offline_batch_size']
    
    pbar = tqdm(range(args.train_steps), desc="IQL Training")
    
    for step in pbar:
        # Get batch
        batch = dataset.get_batch(batch_size)
        
        if batch is None:
            logger.warning("데이터셋이 비어있습니다!")
            break
        
        # Train step
        losses = iql.train_step(batch)
        
        # Update progress bar
        pbar.set_postfix({
            'V': f"{losses['value_loss']:.4f}",
            'Q': f"{losses['q_loss']:.4f}",
            'π': f"{losses['actor_loss']:.4f}"
        })
        
        # Log metrics
        if step % 1000 == 0:
            logger.log_metrics({
                'step': step,
                'value_loss': losses['value_loss'],
                'q_loss': losses['q_loss'],
                'actor_loss': losses['actor_loss']
            }, step=step)
        
        # Evaluation
        if step % args.eval_interval == 0 and step > 0:
            logger.info(f"Step {step}: 평가 중...")
            
            # Run evaluation episode
            state, info = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            
            while not done and not truncated:
                action = iql.get_action(state, deterministic=True)
                next_state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                state = next_state
            
            logger.info(f"평가 에피소드 보상: {episode_reward:.4f}")
            logger.info(f"최종 포트폴리오 가치: {info['portfolio_value']:,.0f}")
            logger.info(f"샤프 비율: {info['sharpe_ratio']:.3f}")
    
    # 모델 저장
    model_path = os.path.join(session_dir, "models", "iql_pretrained.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    iql.save(model_path)
    
    # T-Cell 저장
    tcell_path = os.path.join(session_dir, "models", "tcell.pkl")
    import pickle
    with open(tcell_path, 'wb') as f:
        pickle.dump(tcell, f)
    logger.info(f"T-Cell 저장: {tcell_path}")
    
    logger.info("=" * 80)
    logger.info("IQL 사전학습 완료!")
    logger.info("=" * 80)
    logger.info(f"모델 저장 위치: {model_path}")
    
    # 최종 통계
    if len(iql.losses['value_loss']) > 0:
        avg_value_loss = np.mean(iql.losses['value_loss'][-1000:])
        avg_q_loss = np.mean(iql.losses['q_loss'][-1000:])
        avg_actor_loss = np.mean(iql.losses['actor_loss'][-1000:])
        
        logger.info(f"최종 평균 손실:")
        logger.info(f"  - Value Loss: {avg_value_loss:.4f}")
        logger.info(f"  - Q Loss: {avg_q_loss:.4f}")
        logger.info(f"  - Actor Loss: {avg_actor_loss:.4f}")

if __name__ == "__main__":
    main()