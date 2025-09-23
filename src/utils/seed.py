# src/utils/seed.py

"""
시드 및 디바이스 관리: 재현성 및 하드웨어 설정

목적: 실험 재현성 보장 및 디바이스 관리
의존: torch, numpy
사용처: main.py, 모든 학습 스크립트
역할: 랜덤 시드 고정 및 GPU/CPU 자동 선택

구현 내용:
- 전역 시드 설정 (random, numpy, torch)
- CUDA deterministic 모드 설정
- GPU/CPU 자동 감지 및 선택
- 디바이스 정보 로깅
- 텐서/배열 변환 유틸리티
"""

import numpy as np
import torch
import random
import os
from typing import Optional

def get_best_device() -> torch.device:
    """사용 가능한 최적의 디바이스 자동 선택"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def set_seed(seed: int = 42) -> None:
    """재현 가능한 결과를 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # CUDA 최적화 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 환경 변수 설정
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device_info(device: Optional[torch.device] = None) -> str:
    """현재 사용 중인 device 정보 반환"""
    if device is None:
        device = get_best_device()

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        return f"CUDA GPU: {torch.cuda.get_device_name(0)} (Memory: {props.total_memory // 1024**3}GB)"
    else:
        return "CPU"

class DeviceManager:
    """디바이스 관리 헬퍼 클래스"""
    
    def __init__(self, device: Optional[str] = "auto"):
        if device == "auto":
            self.device = get_best_device()
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {get_device_info(self.device)}")
    
    def to_device(self, tensor):
        """텐서를 설정된 디바이스로 이동"""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        elif isinstance(tensor, np.ndarray):
            return torch.from_numpy(tensor).to(self.device)
        else:
            return torch.tensor(tensor).to(self.device)
    
    def to_numpy(self, tensor):
        """텐서를 numpy 배열로 변환"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        else:
            return np.array(tensor)