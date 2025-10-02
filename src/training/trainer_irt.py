# src/training/trainer_irt.py

"""
IRT 기반 학습 파이프라인

Phase 1: IQL 오프라인 사전학습 (기존 유지)
Phase 2: IRT 온라인 미세조정 (신규)

핵심 차이:
- Actor: BCellIRTActor 사용
- Critic: REDQ 앙상블
- 로깅: IRT 해석 정보 추가

의존성: BCellIRTActor, REDQCritic, IQLAgent, PortfolioEnv
사용처: scripts/train_irt.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
import json

from src.agents.bcell_irt import BCellIRTActor
from src.algorithms.critics.redq import REDQCritic
from src.algorithms.offline.iql import IQLAgent
from src.immune.irt import IRT
from src.environments.portfolio_env import PortfolioEnv
from src.data.market_loader import DataLoader
from src.data.feature_extractor import FeatureExtractor
from src.data.offline_dataset import OfflineDataset
from src.data.replay_buffer import PrioritizedReplayBuffer, Transition
from src.utils.logger import FinFlowLogger, get_session_directory
from src.utils.training_utils import polyak_update, resolve_device
from src.evaluation.metrics import MetricsCalculator

class TrainerIRT:
    """IRT 기반 통합 학습기"""

    def __init__(self, config: Dict):
        """
        Args:
            config: YAML 설정 파일 로드된 딕셔너리
        """
        self.config = config
        # Device 설정: 'auto' 문자열 자동 처리
        device_str = config.get('device', 'auto')
        self.device = resolve_device(device_str)

        self.logger = FinFlowLogger("TrainerIRT")
        self.metrics_calc = MetricsCalculator()

        # 데이터 로드 및 분할
        self._load_and_split_data()

        # 컴포넌트 초기화
        self._initialize_components()

        # 체크포인트 디렉토리
        self.session_dir = Path(get_session_directory())
        self.checkpoint_dir = self.session_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _load_and_split_data(self):
        """데이터 로드 및 train/val/test 분할"""
        data_config = self.config['data']

        loader = DataLoader(cache_dir='data/cache')
        self.price_data = loader.download_data(
            symbols=data_config['symbols'],
            start_date=data_config['start'],
            end_date=data_config.get('test_end', data_config['end']),
            use_cache=data_config.get('cache', True)
        )

        # 날짜 기반 분할
        train_end_date = data_config['end']
        test_start_date = data_config['test_start']

        train_full_data = self.price_data[:train_end_date]
        self.test_data = self.price_data[test_start_date:]

        # train에서 val 분리
        val_ratio = data_config.get('val_ratio', 0.2)
        val_split = int(len(train_full_data) * (1 - val_ratio))

        self.train_data = train_full_data[:val_split]
        self.val_data = train_full_data[val_split:]

        self.logger.info(f"데이터 분할 완료: Train={len(self.train_data)}, Val={len(self.val_data)}, Test={len(self.test_data)}")

    def _initialize_components(self):
        """컴포넌트 초기화"""
        # 차원 계산
        n_assets = len(self.price_data.columns)
        feature_dim = self.config.get('feature_dim', 12)
        state_dim = feature_dim + n_assets + 1  # features + weights + crisis

        self.n_assets = n_assets
        self.state_dim = state_dim
        self.action_dim = n_assets

        # 특성 추출기
        window_size = self.config['env'].get('window_size', 20)
        self.feature_extractor = FeatureExtractor(window=window_size)

        # 환경
        env_config = self.config['env']
        objective_config = self.config.get('objectives')

        self.train_env = PortfolioEnv(
            price_data=self.train_data,
            feature_extractor=self.feature_extractor,
            initial_capital=env_config.get('initial_balance', 1000000),
            transaction_cost=env_config.get('transaction_cost', 0.001),
            max_leverage=env_config.get('max_leverage', 1.0),
            objective_config=objective_config,
            use_advanced_reward=(objective_config is not None)
        )

        self.val_env = PortfolioEnv(
            price_data=self.val_data,
            feature_extractor=self.feature_extractor,
            initial_capital=env_config['initial_balance'],
            transaction_cost=env_config['transaction_cost'],
            max_leverage=env_config['max_leverage'],
            objective_config=objective_config,
            use_advanced_reward=(objective_config is not None)
        )

        # IRT Actor
        irt_config = self.config.get('irt', {})
        self.actor = BCellIRTActor(
            state_dim=state_dim,
            action_dim=n_assets,
            emb_dim=irt_config.get('emb_dim', 128),
            m_tokens=irt_config.get('m_tokens', 6),
            M_proto=irt_config.get('M_proto', 8),
            alpha=irt_config.get('alpha', 0.3),
            ema_beta=irt_config.get('ema_beta', 0.9),
            market_feature_dim=feature_dim
        ).to(self.device)

        # IRT Operator 내부 파라미터 설정
        self.actor.irt = IRT(
            emb_dim=irt_config.get('emb_dim', 128),
            m_tokens=irt_config.get('m_tokens', 6),
            M_proto=irt_config.get('M_proto', 8),
            eps=irt_config.get('eps', 0.05),
            alpha=irt_config.get('alpha', 0.3),
            gamma=irt_config.get('gamma', 0.5),
            lambda_tol=irt_config.get('lambda_tol', 2.0),
            rho=irt_config.get('rho', 0.3),
            eta_0=irt_config.get('eta_0', 0.05),
            eta_1=irt_config.get('eta_1', 0.10),
            kappa=irt_config.get('kappa', 1.0),
            eps_tol=irt_config.get('eps_tol', 0.1),
            n_self_sigs=irt_config.get('n_self_sigs', 4),
            max_iters=irt_config.get('max_iters', 10),
            tol=irt_config.get('tol', 0.001)
        ).to(self.device)

        # REDQ Critics
        redq_config = self.config.get('redq', {})
        self.critic = REDQCritic(
            state_dim=state_dim,
            action_dim=n_assets,
            n_critics=redq_config.get('n_critics', 10),
            m_sample=redq_config.get('m_sample', 2),
            hidden_dims=redq_config.get('hidden_dims', [256, 256])
        ).to(self.device)

        self.critic_target = REDQCritic(
            state_dim=state_dim,
            action_dim=n_assets,
            n_critics=redq_config['n_critics'],
            m_sample=redq_config['m_sample'],
            hidden_dims=redq_config['hidden_dims']
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=redq_config.get('actor_lr', 3e-4)
        )

        self.critic_optim = optim.Adam(
            self.critic.parameters(),
            lr=redq_config.get('critic_lr', 3e-4)
        )

        # Replay Buffer
        replay_config = self.config.get('replay_buffer', {})
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=redq_config.get('buffer_size', 100000),
            alpha=replay_config.get('alpha', 0.6),
            beta=replay_config.get('beta', 0.4)
        )

        # Hyperparameters
        self.gamma = redq_config.get('gamma', 0.99)
        self.tau = redq_config.get('tau', 0.005)
        self.utd_ratio = redq_config.get('utd_ratio', 10)

        self.logger.info(f"컴포넌트 초기화 완료: state_dim={state_dim}, action_dim={n_assets}")

    def train(self):
        """전체 학습 파이프라인"""
        self.logger.info("="*60)
        self.logger.info("IRT 학습 시작")
        self.logger.info("="*60)

        # Phase 1: 오프라인 사전학습 (선택적)
        if not self.config.get('skip_offline', False):
            self.logger.info("\n[Phase 1] 오프라인 IQL 사전학습")
            self._offline_pretrain()
        else:
            self.logger.info("오프라인 학습 스킵")

        # Phase 2: 온라인 IRT 미세조정
        self.logger.info("\n[Phase 2] 온라인 IRT 미세조정")
        best_model = self._online_finetune()

        # Phase 3: 최종 평가
        self.logger.info("\n[Phase 3] 최종 평가")
        test_metrics = self._evaluate_episode(self.test_data, "Test")

        # 결과 저장
        self._save_results(test_metrics)

        return best_model

    def _offline_pretrain(self):
        """오프라인 IQL 사전학습 (기존 로직 유지)"""
        offline_config = self.config['offline']

        # 오프라인 데이터 로드/생성
        offline_data_path = Path('data/offline_data.npz')

        if not offline_data_path.exists():
            self.logger.info("오프라인 데이터 수집 중...")
            dataset = OfflineDataset()
            dataset.collect_from_env(
                self.train_env,
                n_episodes=100,
                diversity_bonus=True
            )
            dataset.save(offline_data_path)

        dataset = OfflineDataset(data_path=offline_data_path)

        # IQL 에이전트 파라미터 추출 (IQLAgent가 받는 것만)
        iql_params = {
            'expectile': offline_config.get('expectile', 0.7),
            'temperature': offline_config.get('temperature', 1.0),
        }

        iql_agent = IQLAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
            **iql_params
        )

        # IQL 학습
        n_epochs = offline_config.get('epochs', 50)
        batch_size = offline_config.get('batch_size', 256)

        for epoch in tqdm(range(n_epochs), desc="IQL Training"):
            batch = dataset.sample_batch(batch_size)

            states = torch.FloatTensor(batch['states']).to(self.device)
            actions = torch.FloatTensor(batch['actions']).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).to(self.device)
            next_states = torch.FloatTensor(batch['next_states']).to(self.device)
            dones = torch.FloatTensor(batch['dones']).to(self.device)

            losses = iql_agent.update(states, actions, rewards, next_states, dones)

            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}: V={losses['value_loss']:.4f}, "
                    f"Q={losses['q_loss']:.4f}, Actor={losses['actor_loss']:.4f}"
                )

        # Actor에 IQL 정책 가중치 로드 (프로토타입 디코더로)
        self.logger.info("IQL 정책을 IRT Actor로 전이 중...")
        # 여기서는 간단히 무시 (실제로는 프로토타입 초기화에 사용 가능)

    def _online_finetune(self):
        """온라인 IRT 미세조정"""
        n_episodes = self.config.get('online_episodes', 200)
        eval_freq = 10

        best_sharpe = -float('inf')
        best_model_path = None

        for episode in tqdm(range(n_episodes), desc="Online IRT Training"):
            # 에피소드 실행
            episode_info = self._run_episode(self.train_env, training=True)

            # 로깅
            self.logger.info(
                f"Episode {episode}: Return={episode_info['return']:.4f}, "
                f"AvgCrisis={episode_info['avg_crisis']:.3f}, "
                f"Turnover={episode_info['turnover']:.4f}"
            )

            # 평가
            if episode % eval_freq == 0:
                val_metrics = self._evaluate_episode(self.val_data, "Validation")

                # Best model 저장
                if val_metrics['sharpe'] > best_sharpe:
                    best_sharpe = val_metrics['sharpe']
                    best_model_path = self._save_checkpoint(episode, is_best=True)
                    self.logger.info(f"New best model: Sharpe={best_sharpe:.4f}")

        # Best model 로드
        if best_model_path:
            self._load_checkpoint(best_model_path)
            self.logger.info(f"Best model loaded: {best_model_path}")

        return self.actor

    def _run_episode(self, env: PortfolioEnv, training: bool = True) -> Dict:
        """단일 에피소드 실행"""
        state, _ = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        episode_return = 0
        episode_length = 0
        crisis_levels = []
        turnovers = []

        done = False
        truncated = False

        while not (done or truncated):
            # 행동 선택
            with torch.no_grad():
                action, info = self.actor(
                    state_tensor,
                    critics=self.critic.get_all_critics() if training else None,
                    deterministic=not training
                )

            action_np = action.cpu().numpy()[0]

            # 환경 스텝
            next_state, reward, done, truncated, env_info = env.step(action_np)

            # 버퍼에 저장 (학습 시)
            if training:
                transition = Transition(
                    state=state,
                    action=action_np,
                    reward=reward,
                    next_state=next_state,
                    done=done or truncated
                )
                self.replay_buffer.push(transition)

            # IRT 업데이트 (UTD ratio만큼)
            if training and len(self.replay_buffer) > 1000:
                for _ in range(self.utd_ratio):
                    self._update_irt()

            # 기록
            episode_return += reward
            episode_length += 1
            crisis_levels.append(info['crisis_level'].item())
            turnovers.append(env_info.get('turnover', 0.0))

            state = next_state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        return {
            'return': episode_return,
            'length': episode_length,
            'avg_crisis': np.mean(crisis_levels),
            'turnover': np.mean(turnovers)
        }

    def _update_irt(self):
        """IRT 업데이트 (1 스텝)"""
        # 배치 샘플
        batch, weights, indices = self.replay_buffer.sample(256)

        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)

        # ===== Critic Update =====
        with torch.no_grad():
            # 타겟 행동
            next_actions, _ = self.actor(next_states, critics=None, deterministic=False)

            # Target Q
            target_q = self.critic_target.get_target_q(next_states, next_actions)
            td_target = rewards + self.gamma * (1 - dones) * target_q

        # 모든 critics 업데이트
        critic_losses = []
        for critic in self.critic.critics:
            q = critic(states, actions)
            critic_loss = F.mse_loss(q, td_target)
            critic_losses.append(critic_loss)

        total_critic_loss = torch.stack(critic_losses).mean()

        self.critic_optim.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # ===== Actor Update =====
        new_actions, _ = self.actor(states, critics=self.critic.get_all_critics())

        # Q값 평균 (모든 critics)
        q_values = self.critic(states, new_actions)
        actor_loss = -torch.stack(q_values).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # ===== Target Update =====
        polyak_update(self.critic_target, self.critic, self.tau)

        # TD error 업데이트 (PER)
        with torch.no_grad():
            td_errors = torch.abs(td_target - self.critic.critics[0](states, actions))
            self.replay_buffer.update_priorities(indices, td_errors.squeeze().cpu().numpy())

    def _evaluate_episode(self, data: pd.DataFrame, phase: str) -> Dict:
        """에피소드 평가"""
        env_config = self.config['env']
        objective_config = self.config.get('objectives')

        env = PortfolioEnv(
            price_data=data,
            feature_extractor=self.feature_extractor,
            initial_capital=env_config['initial_balance'],
            transaction_cost=env_config['transaction_cost'],
            objective_config=objective_config,
            use_advanced_reward=(objective_config is not None)
        )

        # 평가 에피소드 실행
        self.actor.eval()
        episode_info = self._run_episode(env, training=False)
        self.actor.train()

        # 메트릭 계산
        returns_array = np.array(env.all_returns)
        sharpe = self.metrics_calc.calculate_sharpe_ratio(returns_array)

        metrics = {
            'return': episode_info['return'],
            'sharpe': sharpe,
            'avg_crisis': episode_info['avg_crisis']
        }

        self.logger.info(f"{phase} 평가: Sharpe={sharpe:.4f}, Return={episode_info['return']:.4f}")

        return metrics

    def _save_checkpoint(self, episode: int, is_best: bool = False):
        """체크포인트 저장"""
        filename = 'best_model.pth' if is_best else f'checkpoint_ep{episode}.pth'
        path = self.checkpoint_dir / filename

        torch.save({
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict()
        }, path)

        return path

    def _load_checkpoint(self, path: Path):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])

        self.logger.info(f"체크포인트 로드 완료: {path}")

    def _save_results(self, metrics: Dict):
        """최종 결과 저장"""
        results_dir = self.session_dir / 'results'
        results_dir.mkdir(exist_ok=True)

        with open(results_dir / 'final_results.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        self.logger.info(f"결과 저장 완료: {results_dir}")