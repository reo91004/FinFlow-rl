# main.py

"""
메인 실행 파일: FinFlow 시스템 진입점

목적: FinFlow 학습/평가 시스템 실행
의존: trainer.py, 모든 core/agents/data 모듈
사용처: 직접 실행 (python main.py)
역할: CLI 인터페이스 제공 및 파이프라인 시작

파이프라인:
1. 설정 로드 (YAML 또는 CLI 인자)
2. 데이터 로드 (다우존스 30 기본값)
3. Phase 1: IQL 오프라인 사전학습
4. Phase 2: REDQ 온라인 미세조정
5. 평가 및 결과 저장

주요 모드:
- train: 전체 학습 파이프라인 실행
- evaluate: 학습된 모델 평가
- demo: 실시간 거래 데모 (미구현)
"""

import argparse
import json
import yaml
import torch
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
    
    # Training parameters (None = use config file values)
    parser.add_argument('--iql-epochs', type=int, default=None,
                       help='Number of IQL pretraining epochs (overrides config)')
    parser.add_argument('--sac-episodes', type=int, default=None,
                       help='Number of SAC fine-tuning episodes (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    # Model parameters (None = use config file values)
    parser.add_argument('--hidden-dim', type=int, default=None,
                       help='Hidden dimension for networks (overrides config)')
    parser.add_argument('--num-quantiles', type=int, default=None,
                       help='Number of quantiles for distributional RL (overrides config)')
    parser.add_argument('--memory-capacity', type=int, default=None,
                       help='Memory cell capacity (overrides config)')
    
    # Environment parameters (None = use config file values)
    parser.add_argument('--initial-balance', type=float, default=None,
                       help='Initial portfolio balance (overrides config)')
    parser.add_argument('--transaction-cost', type=float, default=None,
                       help='Transaction cost rate (overrides config)')
    parser.add_argument('--max-weight', type=float, default=None,
                       help='Maximum weight per asset (overrides config)')
    
    # Target metrics (None = use config file values)
    parser.add_argument('--target-sharpe', type=float, default=None,
                       help='Target Sharpe ratio (overrides config)')
    parser.add_argument('--target-cvar', type=float, default=None,
                       help='Target CVaR (5%) (overrides config)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
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
        config_dict['data']['symbols'] = args.tickers
    
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
        # CLI 오버라이드 수집
        overrides = {}
        
        # CLI 인자들을 오버라이드로 변환
        if args.iql_epochs is not None:
            overrides['offline_training_epochs'] = args.iql_epochs
        if args.sac_episodes is not None:
            overrides['online_episodes'] = args.sac_episodes
        if args.batch_size is not None:
            overrides['offline_batch_size'] = args.batch_size
            overrides['online_batch_size'] = args.batch_size
        if args.lr is not None:
            overrides['iql_lr'] = args.lr
            overrides['sac_lr'] = args.lr
        if args.memory_capacity is not None:
            overrides['memory_capacity'] = args.memory_capacity
        if args.data_path is not None:
            overrides['data_path'] = args.data_path
        if args.checkpoint_dir is not None:
            overrides['checkpoint_dir'] = args.checkpoint_dir
        if args.target_sharpe is not None:
            overrides['target_sharpe'] = args.target_sharpe
        if args.target_cvar is not None:
            overrides['target_cvar'] = args.target_cvar
        if args.device is not None:
            overrides['device'] = args.device
        if args.initial_balance is not None:
            if 'env_config' not in overrides:
                overrides['env_config'] = {}
            overrides['env_config']['initial_balance'] = args.initial_balance
        if args.transaction_cost is not None:
            if 'env_config' not in overrides:
                overrides['env_config'] = {}
            overrides['env_config']['transaction_cost'] = args.transaction_cost
        # seed는 명시적으로 지정된 경우만
        if args.seed != 42:  # 기본값이 아니면 오버라이드
            overrides['seed'] = seed
        
        # 티커 오버라이드 처리
        if tickers:
            if 'data_config' not in overrides:
                overrides['data_config'] = {}
            overrides['data_config']['tickers'] = tickers
            overrides['data_config']['symbols'] = tickers  # 호환성
        
        # Create training config with YAML and overrides
        config = TrainingConfig(
            config_path=args.config,
            override_params=overrides
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

        # device='auto'를 실제 디바이스로 변환
        if args.device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        else:
            device = args.device

        # Create evaluator
        evaluator = FinFlowEvaluator(
            checkpoint_path=args.resume,
            data_path=args.data_path,
            config_path=args.config,  # Pass the config file
            device=device
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
            data_config={
                'symbols': ['AAPL', 'GOOGL', 'MSFT'],  # 데모용 3개 심볼
                'start': '2019-01-01',
                'end': '2020-12-31',
                'test_start': '2021-01-01',
                'test_end': '2021-12-31'
            },
            offline_training_epochs=10,  # Reduced for demo
            online_episodes=50,  # Reduced for demo
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