# src/core/offline_dataset.py

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

class OfflineDataset:
    """
    오프라인 강화학습을 위한 통합 데이터셋 클래스
    
    .npz와 .pt 형식 모두 지원하여 IQL 사전학습에 사용
    """
    
    def __init__(self, data_path: str = None, capacity: int = 100000):
        """
        Args:
            data_path: 데이터 디렉토리 경로 (None이면 메모리만 사용)
            capacity: 메모리 버퍼 크기
        """
        self.capacity = capacity
        self.transitions = []
        
        # 데이터 파일이 지정된 경우 로드
        if data_path:
            data_path = Path(data_path)
            
            # .npz 파일 우선 확인
            npz_file = data_path / 'offline_data.npz'
            pt_file = data_path / 'offline_dataset.pt'
            
            if npz_file.exists():
                self._load_npz(npz_file)
            elif pt_file.exists():
                self._load_pt(pt_file)
            elif data_path.suffix == '.npz':
                self._load_npz(data_path)
            elif data_path.suffix == '.pt':
                self._load_pt(data_path)
            else:
                # 데이터 파일이 없으면 빈 데이터셋으로 시작
                self.states = np.array([])
                self.actions = np.array([])
                self.rewards = np.array([])
                self.next_states = np.array([])
                self.dones = np.array([])
                self.size = 0
                print(f"빈 데이터셋 생성 (capacity={capacity})")
        else:
            # 메모리 버퍼로만 사용
            self.states = np.array([])
            self.actions = np.array([])
            self.rewards = np.array([])
            self.next_states = np.array([])
            self.dones = np.array([])
            self.size = 0
    
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
    
    def _load_pt(self, file_path: Path):
        """PT 파일 로드 및 변환"""
        data = torch.load(file_path, map_location='cpu')
        
        # transitions 리스트 형식인 경우
        if 'transitions' in data:
            transitions = data['transitions']
            if transitions:
                # numpy 배열로 변환
                self.states = np.array([t['state'] for t in transitions])
                self.actions = np.array([t['action'] for t in transitions])
                self.rewards = np.array([t['reward'] for t in transitions])
                self.next_states = np.array([t['next_state'] for t in transitions])
                self.dones = np.array([t['done'] for t in transitions])
        # 직접 배열 형식인 경우
        elif 'states' in data:
            self.states = data['states'].numpy() if torch.is_tensor(data['states']) else data['states']
            self.actions = data['actions'].numpy() if torch.is_tensor(data['actions']) else data['actions']
            self.rewards = data['rewards'].numpy() if torch.is_tensor(data['rewards']) else data['rewards']
            self.next_states = data['next_states'].numpy() if torch.is_tensor(data['next_states']) else data['next_states']
            self.dones = data['dones'].numpy() if torch.is_tensor(data['dones']) else data['dones']
        
        self.size = len(self.states) if len(self.states) > 0 else 0
        self.state_dim = self.states.shape[1] if self.size > 0 else 0
        self.action_dim = self.actions.shape[1] if self.size > 0 else 0
        
        print(f"PT 데이터셋 로드 완료: {file_path}")
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
    
    def add_trajectory(self, trajectory: list):
        """전체 에피소드 추가 (replay.py 호환)"""
        for transition in trajectory:
            if len(self.transitions) >= self.capacity:
                self.transitions.pop(0)
            self.transitions.append(transition)
    
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
        """데이터셋 저장 (npz 또는 pt 형식)"""
        path = Path(path)
        
        if str(path).endswith('.npz'):
            # NPZ 형식으로 저장
            np.savez(
                path,
                states=self.states,
                actions=self.actions,
                rewards=self.rewards,
                next_states=self.next_states,
                dones=self.dones
            )
            print(f"데이터셋 저장 (NPZ): {path}")
        else:
            # PT 형식으로 저장
            torch.save({
                'states': self.states,
                'actions': self.actions,
                'rewards': self.rewards,
                'next_states': self.next_states,
                'dones': self.dones,
                'transitions': self.transitions if self.transitions else []
            }, path)
            print(f"데이터셋 저장 (PT): {path}")
    
    def load(self, path: str):
        """데이터셋 로드 (pt 형식)"""
        self._load_pt(Path(path))
    
    def collect_from_env(self, env, n_episodes: int = 100, policy: str = 'random', 
                        verbose: bool = True) -> None:
        """
        환경에서 직접 오프라인 데이터 수집
        
        Args:
            env: 거래 환경 (PortfolioEnv)
            n_episodes: 수집할 에피소드 수
            policy: 정책 타입 ('random', 'uniform' 등)
            verbose: 진행 상황 출력 여부
        """
        from tqdm import tqdm
        
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        
        iterator = tqdm(range(n_episodes), desc="Collecting offline data") if verbose else range(n_episodes)
        
        for episode in iterator:
            state, info = env.reset()
            done = False
            truncated = False
            
            while not done and not truncated:
                # 정책에 따른 액션 선택
                if policy == 'random':
                    # Dirichlet 분포로 유효한 포트폴리오 가중치 생성
                    action = np.random.dirichlet(np.ones(env.n_assets))
                elif policy == 'uniform':
                    # 균등 가중치
                    action = np.ones(env.n_assets) / env.n_assets
                else:
                    raise ValueError(f"Unknown policy: {policy}")
                
                next_state, reward, done, truncated, info = env.step(action)
                
                all_states.append(state)
                all_actions.append(action)
                all_rewards.append(reward)
                all_next_states.append(next_state)
                all_dones.append(done or truncated)
                
                state = next_state
        
        # numpy 배열로 변환 및 저장
        if all_states:
            self.states = np.array(all_states)
            self.actions = np.array(all_actions)
            self.rewards = np.array(all_rewards)
            self.next_states = np.array(all_next_states)
            self.dones = np.array(all_dones, dtype=float)
            
            self.size = len(self.states)
            self.state_dim = self.states.shape[1] if self.size > 0 else 0
            self.action_dim = self.actions.shape[1] if self.size > 0 else 0
            
            if verbose:
                print(f"오프라인 데이터 수집 완료: {self.size} transitions")
                print(f"  - State 차원: {self.state_dim}")
                print(f"  - Action 차원: {self.action_dim}")
                print(f"  - 평균 보상: {np.mean(self.rewards):.4f}")
    
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