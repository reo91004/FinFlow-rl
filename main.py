# main.py

"""
FinFlow: Biologically-Inspired Portfolio Defense 2.0

IQL → Distributional SAC 파이프라인 기반 포트폴리오 최적화 시스템
"""

import argparse
import json
import yaml
from pathlib import Path
from src.core.trainer import FinFlowTrainer, TrainingConfig
from src.utils.logger import FinFlowLogger
from src.utils.seed import set_seed, get_device_info

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='FinFlow Training System')
    
    # Training mode
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'demo'],
                       help='Execution mode')
    
    # Config file
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    
    # Paths
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to training data')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    # Data parameters (override config if provided)
    parser.add_argument('--tickers', type=str, nargs='+', default=None,
                       help='Stock tickers to use (overrides config)')
    parser.add_argument('--data-period', type=str, default=None,
                       help='Data period (1y, 2y, 5y, etc.)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Force re-download data')
    
    # Training parameters
    parser.add_argument('--iql-epochs', type=int, default=100,
                       help='Number of IQL pretraining epochs')
    parser.add_argument('--sac-episodes', type=int, default=1000,
                       help='Number of SAC fine-tuning episodes')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension for networks')
    parser.add_argument('--num-quantiles', type=int, default=32,
                       help='Number of quantiles for distributional RL')
    parser.add_argument('--memory-capacity', type=int, default=50000,
                       help='Memory cell capacity')
    
    # Environment parameters
    parser.add_argument('--initial-balance', type=float, default=1000000,
                       help='Initial portfolio balance')
    parser.add_argument('--transaction-cost', type=float, default=0.001,
                       help='Transaction cost rate')
    parser.add_argument('--max-weight', type=float, default=0.2,
                       help='Maximum weight per asset')
    
    # Target metrics
    parser.add_argument('--target-sharpe', type=float, default=1.5,
                       help='Target Sharpe ratio')
    parser.add_argument('--target-cvar', type=float, default=-0.02,
                       help='Target CVaR (5%)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, mps, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Load config file
    config_dict = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        print(f"Warning: Config file {args.config} not found, using defaults")
    
    # Override config with command line arguments
    if args.tickers:
        if 'data' not in config_dict:
            config_dict['data'] = {}
        config_dict['data']['tickers'] = args.tickers
    
    if args.data_period:
        if 'data' not in config_dict:
            config_dict['data'] = {}
        config_dict['data']['period'] = args.data_period
    
    # Get values from config or use defaults
    # Support both 'tickers' and 'symbols' field names
    tickers = config_dict.get('data', {}).get('tickers') or \
              config_dict.get('data', {}).get('symbols', 
                                             ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'])
    data_period = config_dict.get('data', {}).get('period', '2y')
    
    # Set seed
    seed = args.seed or config_dict.get('system', {}).get('seed', 42)
    set_seed(seed)
    
    # Initialize logger
    logger = FinFlowLogger("Main")
    
    # Print header
    print("=" * 60)
    print("FinFlow: Biologically-Inspired Portfolio Defense 2.0")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Tickers: {tickers}")
    print(f"Device: {get_device_info()}")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    if args.mode == 'train':
        # Create training config from loaded config
        config = TrainingConfig(
            env_config={
                'initial_balance': args.initial_balance or config_dict.get('env', {}).get('initial_capital', 1000000),
                'transaction_cost': args.transaction_cost or config_dict.get('env', {}).get('turnover_cost', 0.001),
                'max_leverage': config_dict.get('env', {}).get('max_leverage', 1.0),
                'window_size': config_dict.get('features', {}).get('window', 30)
            },
            data_config={
                'tickers': tickers,
                'period': data_period,
                'interval': config_dict.get('data', {}).get('interval', '1d'),
                'auto_download': config_dict.get('data', {}).get('auto_download', True),
                'use_cache': not args.no_cache
            },
            iql_epochs=args.iql_epochs or config_dict.get('training', {}).get('iql_epochs', 100),
            sac_episodes=args.sac_episodes or config_dict.get('training', {}).get('sac_episodes', 1000),
            iql_batch_size=args.batch_size or config_dict.get('training', {}).get('iql_batch_size', 256),
            sac_batch_size=args.batch_size or config_dict.get('training', {}).get('sac_batch_size', 256),
            iql_lr=args.lr or config_dict.get('training', {}).get('iql_lr', 3e-4),
            sac_lr=args.lr or config_dict.get('training', {}).get('sac_lr', 3e-4),
            memory_capacity=args.memory_capacity or config_dict.get('training', {}).get('memory_capacity', 50000),
            data_path=args.data_path or config_dict.get('system', {}).get('data_path', 'data/processed'),
            checkpoint_dir=args.checkpoint_dir or config_dict.get('system', {}).get('checkpoint_dir', 'checkpoints'),
            target_sharpe=args.target_sharpe or config_dict.get('targets', {}).get('sharpe_ratio', 1.5),
            target_cvar=args.target_cvar or config_dict.get('targets', {}).get('cvar_5', -0.02),
            device=args.device or config_dict.get('system', {}).get('device', 'auto'),
            seed=seed
        )
        
        # Create trainer
        trainer = FinFlowTrainer(config)
        
        # Resume if checkpoint provided
        if args.resume:
            logger.info(f"체크포인트에서 재개: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Train
        trainer.train()
        
    elif args.mode == 'evaluate':
        # Import evaluation module
        from scripts.evaluate import FinFlowEvaluator
        
        if not args.resume:
            print("평가 모드는 --resume 체크포인트가 필요합니다")
            return
        
        # Create evaluator
        evaluator = FinFlowEvaluator(
            checkpoint_path=args.resume,
            data_path=args.data_path,
            device=args.device
        )
        
        # Run evaluation
        evaluator.evaluate()
        
    elif args.mode == 'demo':
        # Demo mode
        print("\n데모 모드:")
        print("1. 합성 데이터로 빠른 학습 실행")
        print("2. 학습된 모델로 실시간 거래 시뮬레이션")
        print("3. XAI 설명 생성 및 시각화")
        
        # Quick training with synthetic data
        config = TrainingConfig(
            iql_epochs=10,  # Reduced for demo
            sac_episodes=50,  # Reduced for demo
            device=args.device,
            seed=args.seed
        )
        
        trainer = FinFlowTrainer(config)
        
        print("\n빠른 학습 시작 (데모용)...")
        trainer.train()
        
        print("\n학습 완료! 결과는 logs/ 디렉토리에서 확인하세요.")
        
    else:
        print(f"알 수 없는 모드: {args.mode}")


if __name__ == "__main__":
    main()