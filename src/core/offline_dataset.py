# src/core/offline_dataset.py

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from src.utils.logger import FinFlowLogger

class OfflineDataset:
    """
    오프라인 강화학습을 위한 데이터셋 클래스
    
    NPZ 형식만 지원 (IQL 사전학습용)
    """
    
    def __init__(self, data_path: str = None):
        """
        Args:
            data_path: NPZ 파일 경로 또는 디렉토리
        """
        self.logger = FinFlowLogger("OfflineDataset")
        # 데이터 초기화
        self.states = np.array([])
        self.actions = np.array([])
        self.rewards = np.array([])
        self.next_states = np.array([])
        self.dones = np.array([])
        self.size = 0
        self.state_dim = 0
        self.action_dim = 0
        
        # 데이터 로드
        if data_path:
            data_path = Path(data_path)
            
            if data_path.is_dir():
                # 디렉토리면 offline_data.npz 찾기
                npz_file = data_path / 'offline_data.npz'
                if npz_file.exists():
                    self._load_npz(npz_file)
                else:
                    print(f"데이터 파일이 없음: {npz_file}")
            elif str(data_path).endswith('.npz'):
                # NPZ 파일 직접 지정
                if data_path.exists():
                    self._load_npz(data_path)
                else:
                    print(f"데이터 파일이 없음: {data_path}")
            else:
                print(f"지원하지 않는 형식: {data_path}")
    
    def _load_npz(self, file_path: Path):
        """NPZ 파일 로드"""
        data = np.load(file_path)
        self.states = data['states']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.next_states = data['next_states']
        self.dones = data['dones']
        
        self.size = len(self.states)
        self.state_dim = self.states.shape[1] if self.size > 0 else 0
        self.action_dim = self.actions.shape[1] if self.size > 0 else 0
        
        print(f"NPZ 데이터셋 로드 완료: {file_path}")
        print(f"  - 샘플 수: {self.size}")
        print(f"  - State 차원: {self.state_dim}")
        print(f"  - Action 차원: {self.action_dim}")
    
    
    def __len__(self) -> int:
        """데이터셋 크기 반환"""
        return self.size
    
    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        랜덤 배치 샘플링
        
        Args:
            batch_size: 배치 크기
            
        Returns:
            배치 데이터 딕셔너리
        """
        if batch_size > self.size:
            raise ValueError(f"배치 크기({batch_size})가 데이터셋 크기({self.size})보다 큽니다")
        
        # 랜덤 인덱스 선택
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
    
    
    def get_batch(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        배치 데이터 반환 (pretrain_iql.py 호환)
        
        Args:
            batch_size: 배치 크기
            device: PyTorch 디바이스
            
        Returns:
            텐서 형태의 배치 데이터
        """
        if self.size == 0:
            return None
        
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        
        batch = {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.FloatTensor(self.dones[indices])
        }
        
        if device:
            batch = {k: v.to(device) for k, v in batch.items()}
        
        return batch
    
    def save(self, path: str):
        """데이터셋 저장 (NPZ 형식만)"""
        path = Path(path)
        
        # NPZ 확장자 강제
        if not str(path).endswith('.npz'):
            path = Path(str(path) + '.npz')
        
        # NPZ 형식으로 저장
        np.savez(
            path,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            next_states=self.next_states,
            dones=self.dones
        )
        print(f"데이터셋 저장: {path}")
    
    def collect_from_env(self, env, n_episodes: int = 500, diversity_bonus: bool = True,
                        verbose: bool = True) -> None:
        """
        다양한 투자 전략으로 데이터 수집
        
        Args:
            env: 거래 환경 (PortfolioEnv)
            n_episodes: 수집할 에피소드 수
            diversity_bonus: 다양성 증강 여부
            verbose: 진행 상황 출력 여부
        """
        strategies = {
            'random': 0.2,           # 20% - 탐색용
            'momentum': 0.2,         # 20% - 모멘텀 전략
            'mean_reversion': 0.2,   # 20% - 평균회귀
            'equal_weight': 0.15,    # 15% - 균등가중
            'risk_parity': 0.15,     # 15% - 위험균등
            'min_variance': 0.1      # 10% - 최소분산
        }
        
        all_transitions = []
        
        for strategy_name, ratio in strategies.items():
            n_eps = int(n_episodes * ratio)
            self.logger.info(f"{strategy_name} 전략으로 {n_eps} 에피소드 수집")
            
            iterator = tqdm(range(n_eps), desc=strategy_name) if verbose else range(n_eps)
            
            for episode in iterator:
                transitions = self._collect_episode(
                    env, 
                    strategy=self._get_strategy(strategy_name, env.n_assets),
                    add_noise=diversity_bonus
                )
                all_transitions.extend(transitions)
        
        # 데이터 품질 검증
        self._validate_data_quality(all_transitions)
        
        # 데이터 증강 (선택적)
        if diversity_bonus:
            augmented = self._augment_data(all_transitions)
            all_transitions.extend(augmented)
        
        # 전환을 배열로 변환
        self._transitions_to_arrays(all_transitions)
        self.logger.info(f"총 {len(all_transitions)} 전환 수집 완료")
        
    def _collect_episode(self, env, strategy, add_noise: bool = False) -> List:
        """단일 에피소드 수집"""
        transitions = []
        state, info = env.reset()
        done = False
        truncated = False
        
        while not done and not truncated:
            # 전략에 따른 액션 선택
            action = strategy(state)
            
            # 노이즈 추가 (선택적)
            if add_noise:
                noise = np.random.randn(len(action)) * 0.02
                action = action + noise
                action = np.maximum(action, 0)
                action = action / (action.sum() + 1e-8)
            
            next_state, reward, done, truncated, info = env.step(action)
            
            transitions.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done or truncated
            })
            
            state = next_state
        
        return transitions
    
    def _get_strategy(self, name: str, n_assets: int):
        """전략별 정책 함수 반환"""
        if name == 'random':
            return lambda s: np.random.dirichlet(np.ones(n_assets))
        
        elif name == 'momentum':
            def momentum_strategy(state):
                # 최근 수익률 기반 모멘텀
                if len(state) > n_assets:
                    # state에서 최근 수익률 추출 (간단한 휴리스틱)
                    recent_returns = state[:n_assets]
                    weights = np.exp(recent_returns * 2)
                else:
                    weights = np.ones(n_assets)
                return weights / weights.sum()
            return momentum_strategy
        
        elif name == 'mean_reversion':
            def mean_reversion_strategy(state):
                # 평균회귀 전략
                if len(state) > n_assets:
                    recent_returns = state[:n_assets]
                    weights = np.exp(-recent_returns * 2)
                else:
                    weights = np.ones(n_assets)
                return weights / weights.sum()
            return mean_reversion_strategy
        
        elif name == 'equal_weight':
            return lambda s: np.ones(n_assets) / n_assets
        
        elif name == 'risk_parity':
            def risk_parity_strategy(state):
                # 간단한 위험균등 (역변동성 가중)
                if len(state) > n_assets * 2:
                    # 변동성 추정 (간단한 휴리스틱)
                    volatilities = np.abs(state[n_assets:n_assets*2]) + 0.01
                    inv_vol = 1 / volatilities
                else:
                    inv_vol = np.ones(n_assets)
                return inv_vol / inv_vol.sum()
            return risk_parity_strategy
        
        elif name == 'min_variance':
            def min_variance_strategy(state):
                # 최소분산 포트폴리오 (간단한 버전)
                if len(state) > n_assets:
                    # 낮은 변동성 자산에 높은 가중치
                    volatilities = np.abs(state[:n_assets]) + 0.01
                    weights = 1 / (volatilities ** 2)
                else:
                    weights = np.ones(n_assets)
                weights = np.maximum(weights, 0)  # Long only
                return weights / weights.sum()
            return min_variance_strategy
        
        else:
            raise ValueError(f"Unknown strategy: {name}")
    
    def _augment_data(self, transitions: List) -> List:
        """데이터 증강으로 다양성 추가"""
        augmented = []
        
        # 일부 전환만 증강 (최대 1000개)
        for t in transitions[:min(1000, len(transitions) // 10)]:
            # 노이즈 추가
            noisy_state = t['state'] + np.random.randn(*t['state'].shape) * 0.01
            noisy_action = t['action'] + np.random.randn(*t['action'].shape) * 0.05
            noisy_action = np.maximum(noisy_action, 0)
            noisy_action = noisy_action / (noisy_action.sum() + 1e-8)
            
            augmented.append({
                'state': noisy_state,
                'action': noisy_action,
                'reward': t['reward'] * 0.95,  # 약간 할인
                'next_state': t['next_state'],
                'done': t['done']
            })
        
        self.logger.info(f"데이터 증강: {len(augmented)}개 샘플 추가")
        return augmented
    
    def _validate_data_quality(self, transitions: List):
        """데이터 품질 검증"""
        if not transitions:
            raise ValueError("수집된 데이터가 없습니다")
        
        # 보상 분포 확인
        rewards = [t['reward'] for t in transitions]
        self.logger.info(f"보상 통계: mean={np.mean(rewards):.6f}, std={np.std(rewards):.6f}")
        
        # 액션 다양성 확인
        actions = np.array([t['action'] for t in transitions])
        action_std = np.std(actions, axis=0)
        self.logger.info(f"액션 다양성: min_std={np.min(action_std):.6f}, max_std={np.max(action_std):.6f}")
    
    def _transitions_to_arrays(self, transitions: List):
        """전환 리스트를 numpy 배열로 변환"""
        if transitions:
            self.states = np.array([t['state'] for t in transitions])
            self.actions = np.array([t['action'] for t in transitions])
            self.rewards = np.array([t['reward'] for t in transitions])
            self.next_states = np.array([t['next_state'] for t in transitions])
            self.dones = np.array([t['done'] for t in transitions], dtype=float)
            
            self.size = len(self.states)
            self.state_dim = self.states.shape[1] if self.size > 0 else 0
            self.action_dim = self.actions.shape[1] if self.size > 0 else 0
            
            # verbose 파라미터 제거 (undefined)
            self.logger.info(f"오프라인 데이터 수집 완료: {self.size} transitions")
            self.logger.info(f"  - State 차원: {self.state_dim}")
            self.logger.info(f"  - Action 차원: {self.action_dim}")
            self.logger.info(f"  - 평균 보상: {np.mean(self.rewards):.4f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        데이터셋 통계 정보 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        if self.size == 0:
            return {'size': 0}
        
        return {
            'size': self.size,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'reward_mean': float(np.mean(self.rewards)),
            'reward_std': float(np.std(self.rewards)),
            'reward_min': float(np.min(self.rewards)),
            'reward_max': float(np.max(self.rewards)),
            'done_ratio': float(np.mean(self.dones))
        }