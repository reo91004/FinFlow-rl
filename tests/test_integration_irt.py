# tests/test_integration_irt.py

"""
IRT 통합 테스트

전체 파이프라인 테스트:
- 데이터 로드
- 환경 생성
- 학습 실행 (1 에피소드)
- 평가 실행
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import tempfile
import shutil

from src.training.trainer_irt import TrainerIRT
from src.agents.bcell_irt import BCellIRTActor
from src.algorithms.critics.redq import REDQCritic
from src.environments.portfolio_env import PortfolioEnv
from src.data.market_loader import DataLoader
from src.data.feature_extractor import FeatureExtractor

def create_test_config():
    """테스트용 간단한 설정 생성"""
    config = {
        'seed': 42,
        'device': 'cpu',
        'data': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],  # 3개 자산만
            'start': '2023-01-01',
            'end': '2023-03-31',
            'test_start': '2023-04-01',
            'test_end': '2023-06-30',
            'val_ratio': 0.2,
            'cache': True
        },
        'env': {
            'initial_balance': 1000000,
            'transaction_cost': 0.001,
            'max_leverage': 1.0
        },
        'feature_dim': 12,
        'offline': {
            'method': 'iql',
            'epochs': 1,  # 빠른 테스트
            'batch_size': 32,
            'expectile': 0.7,
            'temperature': 1.0
        },
        'irt': {
            'emb_dim': 64,  # 작은 차원
            'm_tokens': 4,
            'M_proto': 6,
            'alpha': 0.3
        },
        'redq': {
            'n_critics': 2,  # 최소
            'm_sample': 1,
            'utd_ratio': 1,
            'hidden_dims': [64, 64],  # 작은 네트워크
            'actor_lr': 1e-3,
            'critic_lr': 1e-3,
            'gamma': 0.99,
            'tau': 0.005,
            'buffer_size': 1000
        },
        'online_episodes': 1,  # 1 에피소드만
        'skip_offline': True,  # 오프라인 스킵
        'objectives': {
            'sharpe_beta': 1.0,
            'sharpe_ema_alpha': 0.99,
            'cvar_alpha': 0.05,
            'cvar_target': -0.02,
            'lambda_cvar': 1.0,
            'lambda_turn': 0.1,
            'lambda_dd': 0.0,
            'r_clip': 5.0
        }
    }
    return config

def test_data_loading():
    """데이터 로딩 테스트"""
    loader = DataLoader(cache_dir='data/cache')

    # 간단한 데이터 로드
    data = loader.download_data(
        symbols=['AAPL', 'MSFT'],
        start_date='2023-01-01',
        end_date='2023-01-31',
        use_cache=True
    )

    assert isinstance(data, pd.DataFrame)
    assert len(data.columns) == 2
    assert len(data) > 0

def test_environment_creation():
    """환경 생성 테스트"""
    # 더미 데이터 생성
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame(
        np.random.randn(100, 3) * 0.01 + 1.001,
        index=dates,
        columns=['AAPL', 'MSFT', 'GOOGL']
    )
    data = data.cumprod()  # 가격 데이터로 변환

    # 환경 생성
    env = PortfolioEnv(
        price_data=data,
        feature_extractor=FeatureExtractor(window=20),
        initial_capital=1000000,
        transaction_cost=0.001
    )

    # 리셋 테스트
    state, info = env.reset()
    assert state.shape[0] == 12 + 3 + 1  # features + weights + crisis

    # 스텝 테스트
    action = np.ones(3) / 3  # 균등 가중치
    next_state, reward, done, truncated, info = env.step(action)
    assert next_state.shape == state.shape
    assert isinstance(reward, float)

def test_actor_critic_compatibility():
    """Actor-Critic 호환성 테스트"""
    state_dim = 16  # 12 features + 3 assets + 1 crisis
    action_dim = 3

    # Actor 생성
    actor = BCellIRTActor(
        state_dim=state_dim,
        action_dim=action_dim,
        emb_dim=64,
        m_tokens=4,
        M_proto=6
    )

    # Critic 생성
    critic = REDQCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        n_critics=2,
        m_sample=1,
        hidden_dims=[64, 64]
    )

    # 배치 생성
    B = 4
    state = torch.randn(B, state_dim)

    # Actor forward
    action, info = actor(state, critics=critic.get_all_critics())
    assert action.shape == (B, action_dim)

    # Critic forward
    q_values = critic(state, action)
    assert len(q_values) == 2

def test_full_training_pipeline():
    """전체 학습 파이프라인 테스트 (1 에피소드)"""
    config = create_test_config()

    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        # 로그 디렉토리 설정
        import os
        os.environ['FINFLOW_LOG_DIR'] = temp_dir

        try:
            # 트레이너 생성
            trainer = TrainerIRT(config)

            # 1 에피소드 학습
            config['online_episodes'] = 1
            config['skip_offline'] = True

            # 학습 실행
            best_model = trainer._online_finetune()

            # 모델이 반환되었는지 확인
            assert best_model is not None
            assert isinstance(best_model, BCellIRTActor)

            print("전체 파이프라인 테스트 통과!")

        except Exception as e:
            print(f"파이프라인 테스트 실패: {e}")
            raise

def test_checkpoint_save_load():
    """체크포인트 저장/로드 테스트"""
    actor = BCellIRTActor(state_dim=16, action_dim=3, emb_dim=64)
    critic = REDQCritic(state_dim=16, action_dim=3, n_critics=2)

    # 임시 파일에 저장
    with tempfile.NamedTemporaryFile(suffix='.pth') as f:
        # 저장
        torch.save({
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict()
        }, f.name)

        # 새 모델 생성
        actor2 = BCellIRTActor(state_dim=16, action_dim=3, emb_dim=64)
        critic2 = REDQCritic(state_dim=16, action_dim=3, n_critics=2)

        # 로드
        checkpoint = torch.load(f.name)
        actor2.load_state_dict(checkpoint['actor_state_dict'])
        critic2.load_state_dict(checkpoint['critic_state_dict'])

        # 같은 출력을 내는지 확인
        state = torch.randn(2, 16)
        with torch.no_grad():
            action1, _ = actor(state, deterministic=True)
            action2, _ = actor2(state, deterministic=True)
        assert torch.allclose(action1, action2)

def test_memory_efficiency():
    """메모리 효율성 테스트"""
    import psutil
    import os

    # 초기 메모리
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # 모델 생성
    actor = BCellIRTActor(state_dim=43, action_dim=30)
    critic = REDQCritic(state_dim=43, action_dim=30, n_critics=10)

    # 여러 배치 처리
    for _ in range(10):
        state = torch.randn(32, 43)
        action, _ = actor(state)
        q_values = critic(state, action)

    # 최종 메모리
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = mem_after - mem_before

    print(f"메모리 사용량: {mem_used:.2f} MB")
    assert mem_used < 500  # 500MB 이하

if __name__ == '__main__':
    print("통합 테스트 시작...")

    # 각 테스트 실행
    print("1. 데이터 로딩 테스트...")
    test_data_loading()

    print("2. 환경 생성 테스트...")
    test_environment_creation()

    print("3. Actor-Critic 호환성 테스트...")
    test_actor_critic_compatibility()

    print("4. 체크포인트 저장/로드 테스트...")
    test_checkpoint_save_load()

    print("5. 메모리 효율성 테스트...")
    test_memory_efficiency()

    print("6. 전체 파이프라인 테스트...")
    test_full_training_pipeline()

    print("\n✅ 모든 통합 테스트 통과!")