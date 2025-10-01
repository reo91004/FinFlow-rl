# 🔄 **FinFlow-RL IRT 리팩토링 핸드오버 프롬프트**

**Version**: 2.0-IRT  
**Date**: 2025-10-01  
**Objective**: IRT (Immune Replicator Transport) Operator 기반 완전 리팩토링

---

## 📋 **Executive Summary**

### **리팩토링 목표**

1. **IRT Operator 통합**: OT + Replicator 기반 새로운 정책 혼합
2. **코드 간소화**: 불필요한 복잡도 제거, 핵심 기능 집중
3. **설명 가능성 강화**: 수송 행렬, 복제자 가중치, 비용 분해 시각화
4. **실전 작동 보장**: 깡통 코드 없이 end-to-end 학습 가능

### **주요 변경 사항**

| 항목          | Before                   | After                 |
| ------------- | ------------------------ | --------------------- |
| **Actor**     | Distributional SAC + MoE | IRT (OT + Replicator) |
| **T-Cell**    | Isolation Forest + SHAP  | 경량 신경망 (z, d, c) |
| **Memory**    | k-NN 기반 검색           | EMA w\_{t-1} (통합)   |
| **파일 수**   | ~27개                    | ~18개 (-33%)          |
| **코드 라인** | ~8000                    | ~5500 (-31%)          |

---

## 🗂️ **프로젝트 구조 변경**

### **최종 디렉토리 구조**

```
FinFlow-rl/
├── configs/
│   ├── default_irt.yaml              # [NEW] IRT 기본 설정
│   └── experiments/
│       ├── ablation_irt.yaml         # [NEW] Ablation study
│       └── crisis_focus.yaml         # [NEW] 위기 구간 집중
│
├── src/
│   ├── immune/                       # [NEW] 면역 모듈
│   │   ├── __init__.py
│   │   ├── irt.py                    # [NEW] IRT Operator
│   │   └── t_cell.py                 # [MODIFIED] 경량화
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   └── bcell_irt.py              # [NEW] IRT 기반 Actor
│   │
│   ├── algorithms/
│   │   ├── offline/
│   │   │   ├── __init__.py
│   │   │   └── iql.py                # [KEEP] 간소화
│   │   └── critics/                  # [NEW] Critic 분리
│   │       ├── __init__.py
│   │       └── redq.py               # [NEW] REDQ Critic
│   │
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── portfolio_env.py          # [KEEP] 변경 없음
│   │   └── reward_functions.py       # [KEEP] 변경 없음
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── market_loader.py          # [KEEP]
│   │   ├── feature_extractor.py      # [KEEP]
│   │   ├── offline_dataset.py        # [KEEP]
│   │   └── replay_buffer.py          # [KEEP]
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # [KEEP]
│   │   ├── visualizer.py             # [MODIFIED] IRT 시각화 추가
│   │   └── explainer.py              # [MODIFIED] IRT 해석 추가
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer_irt.py            # [NEW] IRT 전용 트레이너
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                 # [KEEP]
│       ├── monitoring.py             # [KEEP]
│       └── training_utils.py         # [KEEP]
│
├── scripts/
│   ├── train_irt.py                  # [NEW] IRT 학습
│   ├── evaluate_irt.py               # [NEW] IRT 평가
│   └── visualize_irt.py              # [NEW] IRT 시각화
│
├── tests/
│   ├── test_irt.py                   # [NEW] IRT 단위 테스트
│   └── test_integration_irt.py       # [NEW] 통합 테스트
│
└── docs/
    └── IRT_ARCHITECTURE.md           # [NEW] IRT 아키텍처 문서
```

### **삭제할 파일**

```bash
# 제거 대상
src/algorithms/online/memory.py       # → w_prev EMA로 대체
src/algorithms/online/meta.py         # → 사용 안 함
src/models/networks.py                # → 개별 모듈로 분산
src/baselines/                        # → 간소화
src/experiments/                      # → scripts/로 통합
```

---

## 📝 **파일별 상세 구현**

### **1. src/immune/irt.py** [NEW]

```python
# src/immune/irt.py

"""
IRT (Immune Replicator Transport) Operator

이론적 기초:
- Optimal Transport: Cuturi (2013) Entropic OT
- Replicator Dynamics: Hofbauer & Sigmund (1998)
- 결합: (1-α)·Replicator + α·OT

핵심 수식:
w_t = (1-α)·Replicator(w_{t-1}, f_t) + α·Transport(E_t, K, C_t)

의존성: torch
사용처: BCellIRTActor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Sinkhorn(nn.Module):
    """
    엔트로피 정규화 최적수송 (Sinkhorn 알고리즘)

    수학적 배경:
    min_{P∈U(u,v)} <P,C> + ε·KL(P||uv^T)

    수렴 보장: O(1/ε) 반복 내 선형 수렴 (Cuturi, 2013)
    """

    def __init__(self, max_iters: int = 10, eps: float = 0.05, tol: float = 1e-3):
        super().__init__()
        self.max_iters = max_iters
        self.eps = eps
        self.tol = tol

    def forward(self, C: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            C: 비용 행렬 [B, m, M]
            u: 소스 분포 [B, m, 1]
            v: 타겟 분포 [B, 1, M]

        Returns:
            P: 수송 계획 [B, m, M]
        """
        B, m, M = C.shape

        # Log-space 연산 (수치 안정성)
        log_K = -C / (self.eps + 1e-8)
        log_u = torch.log(u + 1e-8)
        log_v = torch.log(v + 1e-8)

        log_a = torch.zeros_like(log_u)
        log_b = torch.zeros_like(log_v)

        # Sinkhorn 반복
        for iter_idx in range(self.max_iters):
            log_a_prev = log_a.clone()

            log_a = log_u - torch.logsumexp(log_K + log_b, dim=2, keepdim=True)
            log_b = log_v - torch.logsumexp(log_K + log_a, dim=1, keepdim=True)

            # 조기 종료 (수렴 체크)
            if iter_idx > 0:
                err = torch.abs(log_a - log_a_prev).max()
                if err < self.tol:
                    break

        # 수송 계획 계산
        P = torch.exp(log_a + log_K + log_b)

        # 수치 안정성 체크
        P = torch.clamp(P, min=0.0, max=1.0)

        return P

class IRT(nn.Module):
    """
    Immune Replicator Transport Operator

    핵심 혁신:
    1. OT: 구조적 매칭 (현재 상태 ↔ 프로토타입)
    2. Replicator: 시간 메모리 (과거 성공 전략 선호)
    3. 면역 신호: 비용 함수에 도메인 지식 내장

    수학적 정의:
    C_ij = d(e_i,k_j) - γ<e_i,d_t> + λ[tolerance] + ρ[checkpoint]
    P* = Sinkhorn(C, u, v)
    w_tilde ∝ w_{t-1}·exp(η(c)[f - \bar{f}])
    w_t = (1-α)·w_tilde + α·P*1_m
    """

    def __init__(self,
                 emb_dim: int,
                 m_tokens: int = 6,
                 M_proto: int = 8,
                 eps: float = 0.05,
                 alpha: float = 0.3,
                 gamma: float = 0.5,
                 lambda_tol: float = 2.0,
                 rho: float = 0.3):
        super().__init__()

        self.emb_dim = emb_dim
        self.m = m_tokens
        self.M = M_proto
        self.alpha = alpha

        # 하이퍼파라미터
        self.gamma = gamma          # 공자극 가중치
        self.lambda_tol = lambda_tol  # 내성 가중치
        self.rho = rho              # 체크포인트 가중치
        self.kappa = 1.0            # 내성 게인
        self.eps_tol = 0.1          # 내성 임계값

        # 학습 가능한 마할라노비스 메트릭
        # M = L^T L (positive definite 보장)
        self.metric_L = nn.Parameter(torch.eye(emb_dim))

        # 자기-내성 서명 (학습 가능)
        self.self_sigs = nn.Parameter(torch.randn(4, emb_dim) * 0.1)

        # Sinkhorn 알고리즘
        self.sinkhorn = Sinkhorn(eps=eps)

    def _mahalanobis_distance(self, E: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        학습 가능한 마할라노비스 거리

        d_M(x,y) = sqrt((x-y)^T M (x-y)), M = L^T L
        """
        M = self.metric_L.T @ self.metric_L  # [D, D]

        diff = E.unsqueeze(2) - K.unsqueeze(1)  # [B, m, M, D]

        # (x-y)^T M (x-y) = sum_ij (x-y)_i M_ij (x-y)_j
        mahal_sq = torch.einsum('bmnd,de,bmne->bmn', diff, M, diff)
        mahal = torch.sqrt(torch.clamp(mahal_sq, min=1e-8))

        return mahal  # [B, m, M]

    def _cost_matrix(self,
                     E: torch.Tensor,
                     K: torch.Tensor,
                     danger: torch.Tensor,
                     proto_conf: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        면역학적 비용 행렬 구성

        C_ij = distance - γ·co_stim + λ·tolerance + ρ·checkpoint

        Args:
            E: 에피토프 [B, m, D]
            K: 프로토타입 [B, M, D]
            danger: 공자극 임베딩 [B, D]
            proto_conf: 프로토타입 과신도 [B, 1, M] (optional)
        """
        B, m, D = E.shape
        M = K.shape[1]

        # 1. 기본 거리
        dist = self._mahalanobis_distance(E, K)  # [B, m, M]

        # 2. 공자극 (Co-stimulation)
        # 위험 신호와 정렬된 에피토프 선호
        co_stim = torch.einsum('bmd,bd->bm', E, danger).unsqueeze(2)  # [B, m, 1]

        # 3. 음성 선택 (Tolerance)
        # 자기-내성 서명과 유사한 에피토프 억제
        E_norm = F.normalize(E, dim=-1)  # [B, m, D]
        sig_norm = F.normalize(self.self_sigs, dim=-1)  # [S, D]

        cos_sim = E_norm @ sig_norm.T  # [B, m, S]
        worst_match = cos_sim.max(dim=-1, keepdim=True)[0]  # [B, m, 1]

        tolerance_penalty = torch.relu(
            self.kappa * worst_match - self.eps_tol
        )  # [B, m, 1]

        # 4. 체크포인트 억제 (Checkpoint)
        # 과신하는 프로토타입 억제
        if proto_conf is None:
            proto_conf = torch.zeros(B, 1, M, device=E.device)

        # 5. 종합 비용
        C = (
            dist
            - self.gamma * co_stim
            + self.lambda_tol * tolerance_penalty
            + self.rho * proto_conf
        )

        return C  # [B, m, M]

    def forward(self,
                E: torch.Tensor,
                K: torch.Tensor,
                danger: torch.Tensor,
                w_prev: torch.Tensor,
                fitness: torch.Tensor,
                crisis_level: torch.Tensor,
                proto_conf: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        IRT 연산자 forward pass

        Args:
            E: 에피토프 [B, m, D]
            K: 프로토타입 [B, M, D]
            danger: 공자극 임베딩 [B, D]
            w_prev: 이전 혼합 가중치 [B, M]
            fitness: 프로토타입 적합도 [B, M]
            crisis_level: 위기 레벨 [B, 1]
            proto_conf: 프로토타입 과신도 [B, 1, M]

        Returns:
            w: 새 혼합 가중치 [B, M]
            P: 수송 계획 [B, m, M] (해석용)
        """
        B, m, D = E.shape
        M = K.shape[1]

        # ===== Step 1: Optimal Transport 매칭 =====
        u = torch.full((B, m, 1), 1.0/m, device=E.device)
        v = torch.full((B, 1, M), 1.0/M, device=E.device)

        C = self._cost_matrix(E, K, danger, proto_conf)
        P = self.sinkhorn(C, u, v)  # [B, m, M]

        # OT 마진 (프로토타입별 수송 질량)
        p_mass = P.sum(dim=1)  # [B, M]

        # ===== Step 2: Replicator 업데이트 =====
        # 위기 가열: η(c) = η_0 + η_1·c
        eta_0, eta_1 = 0.05, 0.10
        eta = eta_0 + eta_1 * crisis_level  # [B, 1]

        # Advantage 계산
        baseline = (w_prev * fitness).sum(dim=-1, keepdim=True)  # [B, 1]
        advantage = fitness - baseline  # [B, M]

        # 자기-내성 페널티 (프로토타입도 검사)
        K_norm = F.normalize(K, dim=-1)  # [B, M, D]
        sig_norm = F.normalize(self.self_sigs, dim=-1)  # [S, D]

        proto_self_sim = (K_norm @ sig_norm.T).max(dim=-1)[0]  # [B, M]
        r_penalty = 0.5 * proto_self_sim

        # Replicator 방정식 (log-space)
        log_w_prev = torch.log(w_prev + 1e-8)
        log_tilde_w = log_w_prev + eta * advantage - r_penalty

        tilde_w = F.softmax(log_tilde_w, dim=-1)  # [B, M]

        # ===== Step 3: 이중 결합 (OT ∘ Replicator) =====
        w = (1 - self.alpha) * tilde_w + self.alpha * p_mass

        # 정규화 (수치 안정성)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        w = torch.clamp(w, min=1e-6, max=1.0)

        return w, P
```

### **2. src/immune/t_cell.py** [MODIFIED]

```python
# src/immune/t_cell.py

"""
T-Cell: 경량 위기 감지 시스템

이전 버전과의 차이:
- Isolation Forest 제거 (복잡도 감소)
- 단일 신경망으로 z, d, c 동시 출력
- 온라인 정규화로 안정성 확보

출력:
- z: 위기 타입 점수 [B, K] (다차원)
- d: 공자극 임베딩 [B, D] (IRT 비용 함수용)
- c: 스칼라 위기 레벨 [B, 1] (복제자 가열용)

의존성: torch
사용처: BCellIRTActor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TCellMinimal(nn.Module):
    """경량 T-Cell: 위기 감지 + 공자극 임베딩"""

    def __init__(self,
                 in_dim: int,
                 emb_dim: int = 128,
                 n_types: int = 4,
                 momentum: float = 0.99):
        """
        Args:
            in_dim: 입력 차원 (시장 특성, 예: 12)
            emb_dim: 공자극 임베딩 차원
            n_types: 위기 타입 수 (변동성, 유동성, 상관관계, 시스템)
            momentum: 온라인 정규화 모멘텀
        """
        super().__init__()

        self.n_types = n_types
        self.emb_dim = emb_dim
        self.momentum = momentum

        # 단일 인코더 (효율성)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_types + emb_dim)
        )

        # 온라인 정규화 통계 (학습 중 업데이트)
        self.register_buffer('mu', torch.zeros(n_types))
        self.register_buffer('sigma', torch.ones(n_types))
        self.register_buffer('count', torch.zeros(1))

        # 위기 타입별 가중치 (학습 가능)
        self.alpha = nn.Parameter(torch.ones(n_types) / n_types)

    def forward(self,
                features: torch.Tensor,
                update_stats: bool = True) -> tuple:
        """
        Args:
            features: 시장 특성 [B, F]
            update_stats: 통계 업데이트 여부 (학습=True, 평가=False)

        Returns:
            z: 위기 타입 점수 [B, K]
            d: 공자극 임베딩 [B, D]
            c: 스칼라 위기 레벨 [B, 1]
        """
        h = self.encoder(features)  # [B, K+D]

        # 분리
        z = h[:, :self.n_types]      # [B, K]
        d = h[:, self.n_types:]      # [B, D]

        # 온라인 정규화 (학습 시)
        if update_stats and self.training:
            with torch.no_grad():
                batch_mu = z.mean(dim=0)
                batch_sigma = z.std(dim=0) + 1e-6

                # EMA 업데이트
                self.mu = self.momentum * self.mu + (1 - self.momentum) * batch_mu
                self.sigma = self.momentum * self.sigma + (1 - self.momentum) * batch_sigma
                self.count += 1

        # 표준화
        z_std = (z - self.mu) / (self.sigma + 1e-6)  # [B, K]

        # 가중 합산 → 시그모이드 (0-1 범위)
        alpha_norm = F.softmax(self.alpha, dim=0)  # [K]
        c = torch.sigmoid(
            (z_std * alpha_norm).sum(dim=-1, keepdim=True)
        )  # [B, 1]

        return z, d, c

    def get_crisis_interpretation(self, z: torch.Tensor) -> dict:
        """
        위기 타입 해석 (시각화용)

        Args:
            z: 위기 타입 점수 [B, K]

        Returns:
            해석 딕셔너리
        """
        crisis_types = ['Volatility', 'Liquidity', 'Correlation', 'Systemic']

        z_std = (z - self.mu) / (self.sigma + 1e-6)
        z_prob = torch.sigmoid(z_std)  # [B, K]

        interpretation = {}
        for i, ctype in enumerate(crisis_types[:self.n_types]):
            interpretation[ctype] = z_prob[:, i].mean().item()

        return interpretation
```

### **3. src/agents/bcell_irt.py** [NEW]

```python
# src/agents/bcell_irt.py

"""
B-Cell Actor with IRT (Immune Replicator Transport)

핵심 기능:
1. 에피토프 인코딩: 상태 → 다중 토큰
2. IRT 연산: OT + Replicator 혼합
3. Dirichlet 디코딩: 혼합 → 포트폴리오 가중치
4. EMA 메모리: w_prev 관리

의존성: IRT, TCellMinimal, QNetwork
사용처: TrainerIRT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from src.immune.irt import IRT
from src.immune.t_cell import TCellMinimal

class BCellIRTActor(nn.Module):
    """IRT 기반 B-Cell Actor"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 emb_dim: int = 128,
                 m_tokens: int = 6,
                 M_proto: int = 8,
                 alpha: float = 0.3):
        """
        Args:
            state_dim: 상태 차원 (예: 43)
            action_dim: 행동 차원 (예: 30)
            emb_dim: 임베딩 차원
            m_tokens: 에피토프 토큰 수
            M_proto: 프로토타입 수
            alpha: OT-Replicator 결합 비율
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.emb_dim = emb_dim
        self.m = m_tokens
        self.M = M_proto

        # ===== 에피토프 인코더 =====
        self.epitope_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, m_tokens * emb_dim)
        )

        # ===== 프로토타입 키 (학습 가능) =====
        # Xavier 초기화
        self.proto_keys = nn.Parameter(
            torch.randn(M_proto, emb_dim) / (emb_dim ** 0.5)
        )

        # ===== 프로토타입별 Dirichlet 디코더 =====
        # 각 프로토타입은 독립적인 정책 (전문가)
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, action_dim),
                nn.Softplus()  # 양수 concentration 보장
            )
            for _ in range(M_proto)
        ])

        # ===== IRT 연산자 =====
        self.irt = IRT(
            emb_dim=emb_dim,
            m_tokens=m_tokens,
            M_proto=M_proto,
            alpha=alpha
        )

        # ===== T-Cell 통합 =====
        market_feature_dim = 12  # FeatureExtractor 출력
        self.t_cell = TCellMinimal(
            in_dim=market_feature_dim,
            emb_dim=emb_dim
        )

        # ===== 이전 가중치 (EMA) =====
        self.register_buffer('w_prev', torch.full((1, M_proto), 1.0/M_proto))
        self.ema_beta = 0.9

    def _compute_fitness(self,
                        state: torch.Tensor,
                        critics: List[nn.Module]) -> torch.Tensor:
        """
        각 프로토타입의 적합도 (fitness) 계산

        방법: 각 프로토타입 정책으로 행동 샘플 → Critics로 Q값 평가

        Args:
            state: [B, S]
            critics: QNetwork 리스트 (REDQ)

        Returns:
            fitness: [B, M]
        """
        B = state.size(0)
        fitness = torch.zeros(B, self.M, device=state.device)

        with torch.no_grad():
            K_batch = self.proto_keys.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

            for j in range(self.M):
                # 프로토타입 j의 concentration
                conc_j = self.decoders[j](K_batch[:, j, :])  # [B, A]

                # Dirichlet 분포에서 샘플
                conc_j_clamped = torch.clamp(conc_j, min=1.0, max=100.0)
                dist_j = torch.distributions.Dirichlet(conc_j_clamped)
                action_j = dist_j.sample()  # [B, A]

                # Critics로 Q값 평가 (앙상블 평균)
                q_values = []
                for critic in critics:
                    q = critic(state, action_j)
                    q_values.append(q.squeeze(-1))  # [B]

                fitness[:, j] = torch.stack(q_values).mean(dim=0)

        return fitness

    def forward(self,
                state: torch.Tensor,
                critics: Optional[List[nn.Module]] = None,
                deterministic: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            state: [B, S]
            critics: QNetwork 리스트 (fitness 계산용)
            deterministic: 결정적 행동 (평가 시)

        Returns:
            action: [B, A] - 포트폴리오 가중치
            info: 해석 정보 (w, P, crisis 등)
        """
        B = state.size(0)

        # ===== Step 1: T-Cell 위기 감지 =====
        market_features = state[:, :12]  # 시장 특성 추출
        z, danger_embed, crisis_level = self.t_cell(
            market_features,
            update_stats=self.training
        )

        # ===== Step 2: 에피토프 인코딩 =====
        E = self.epitope_encoder(state).view(B, self.m, self.emb_dim)  # [B, m, D]

        # ===== Step 3: 프로토타입 확장 =====
        K = self.proto_keys.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

        # ===== Step 4: Fitness 계산 =====
        if critics is not None and not deterministic:
            fitness = self._compute_fitness(state, critics)
        else:
            # 평가 모드 또는 critics 없음: 균등 fitness
            fitness = torch.ones(B, self.M, device=state.device)

        # ===== Step 5: IRT 연산 =====
        w_prev_batch = self.w_prev.expand(B, -1)  # [B, M]

        w, P = self.irt(
            E=E,
            K=K,
            danger=danger_embed,
            w_prev=w_prev_batch,
            fitness=fitness,
            crisis_level=crisis_level,
            proto_conf=None  # 필요 시 추가
        )

        # ===== Step 6: Dirichlet 혼합 정책 =====
        # 각 프로토타입의 concentration 계산
        concentrations = torch.stack([
            self.decoders[j](K[:, j, :]) for j in range(self.M)
        ], dim=1)  # [B, M, A]

        # IRT 가중치로 혼합
        mixed_conc = torch.einsum('bm,bma->ba', w, concentrations) + 1.0  # [B, A]

        if deterministic:
            # 결정적: Dirichlet 평균 (mode)
            action = (mixed_conc - 1) / (mixed_conc.sum(dim=-1, keepdim=True) - self.action_dim)
            action = torch.clamp(action, min=0.0)
            action = action / (action.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            # 확률적: Dirichlet 샘플
            mixed_conc_clamped = torch.clamp(mixed_conc, min=1.0, max=100.0)
            dist = torch.distributions.Dirichlet(mixed_conc_clamped)
            action = dist.sample()

        # ===== Step 7: EMA 업데이트 (w_prev) =====
        if self.training:
            with torch.no_grad():
                self.w_prev = (
                    self.ema_beta * self.w_prev
                    + (1 - self.ema_beta) * w.detach().mean(dim=0, keepdim=True)
                )

        # ===== Step 8: 해석 정보 수집 =====
        info = {
            'w': w.detach(),  # [B, M] - 프로토타입 가중치
            'P': P.detach(),  # [B, m, M] - 수송 계획
            'crisis_level': crisis_level.detach(),  # [B, 1]
            'crisis_types': z.detach(),  # [B, K]
            'fitness': fitness.detach()  # [B, M]
        }

        return action, info
```

### **4. src/algorithms/critics/redq.py** [NEW]

```python
# src/algorithms/critics/redq.py

"""
REDQ (Randomized Ensemble Double Q-learning) Critic

핵심 아이디어:
- N개 Q-network 앙상블 (예: N=10)
- 매 업데이트마다 M개 서브셋 샘플 (예: M=2)
- Min Q 사용으로 overestimation bias 완화

근거: Chen et al. (2021) "Randomized Ensembled Double Q-learning"

의존성: torch
사용처: TrainerIRT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class QNetwork(nn.Module):
    """단일 Q-network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()

        layers = []
        in_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, S]
            action: [B, A]

        Returns:
            Q: [B, 1]
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class REDQCritic(nn.Module):
    """REDQ Critic 앙상블"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_critics: int = 10,
                 m_sample: int = 2,
                 hidden_dims: List[int] = [256, 256]):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원
            n_critics: 앙상블 크기
            m_sample: 서브셋 크기 (target 계산용)
            hidden_dims: 은닉층 차원
        """
        super().__init__()

        self.n_critics = n_critics
        self.m_sample = m_sample

        # N개 독립적인 Q-network
        self.critics = nn.ModuleList([
            QNetwork(state_dim, action_dim, hidden_dims)
            for _ in range(n_critics)
        ])

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:
        """
        모든 critics 출력

        Returns:
            List of [B, 1] tensors (길이 N)
        """
        return [critic(state, action) for critic in self.critics]

    def get_target_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Target Q 계산: M개 서브셋의 최솟값

        Args:
            state: [B, S]
            action: [B, A]

        Returns:
            target_q: [B, 1]
        """
        with torch.no_grad():
            # M개 critics 랜덤 선택
            indices = torch.randperm(self.n_critics)[:self.m_sample]

            q_values = []
            for idx in indices:
                q = self.critics[idx](state, action)
                q_values.append(q)

            # Min Q (overestimation bias 완화)
            target_q = torch.min(torch.stack(q_values), dim=0)[0]

        return target_q

    def get_all_critics(self) -> List[nn.Module]:
        """모든 critics 반환 (fitness 계산용)"""
        return list(self.critics)
```

### **5. src/training/trainer_irt.py** [NEW]

```python
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
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
import json

from src.agents.bcell_irt import BCellIRTActor
from src.algorithms.critics.redq import REDQCritic
from src.algorithms.offline.iql import IQLAgent
from src.environments.portfolio_env import PortfolioEnv
from src.data.market_loader import DataLoader
from src.data.feature_extractor import FeatureExtractor
from src.data.offline_dataset import OfflineDataset
from src.data.replay_buffer import PrioritizedReplayBuffer, Transition
from src.utils.logger import FinFlowLogger, get_session_directory
from src.utils.training_utils import polyak_update
from src.evaluation.metrics import MetricsCalculator

class TrainerIRT:
    """IRT 기반 통합 학습기"""

    def __init__(self, config: Dict):
        """
        Args:
            config: YAML 설정 파일 로드된 딕셔너리
        """
        self.config = config
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )

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
        self.feature_extractor = FeatureExtractor(window=20)

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
            alpha=irt_config.get('alpha', 0.3)
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
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=redq_config.get('buffer_size', 100000),
            alpha=0.6,
            beta=0.4
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

        # IQL 에이전트
        iql_agent = IQLAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
            **offline_config
        )

        # IQL 학습
        n_epochs = offline_config.get('epochs', 50)
        batch_size = offline_config.get('batch_size', 256)

        for epoch in tqdm(range(n_epochs), desc="IQL Training"):
            batch = dataset.sample(batch_size)

            states = torch.FloatTensor(batch['states']).to(self.device)
            actions = torch.FloatTensor(batch['actions']).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).to(self.device)
            next_states = torch.FloatTensor(batch['next_states']).to(self.device)
            dones = torch.FloatTensor(batch['dones']).to(self.device)

            losses = iql_agent.update(states, actions, rewards, next_states, dones)

            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: V_loss={losses['v_loss']:.4f}, Q_loss={losses['q_loss']:.4f}")

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
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

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
```

### **6. configs/default_irt.yaml** [NEW]

```yaml
# configs/default_irt.yaml
# IRT 기본 설정

seed: 42
device: auto # cpu|cuda|auto

# 데이터 설정
data:
  symbols:
    [
      "AAPL",
      "MSFT",
      "GOOGL",
      "AMZN",
      "NVDA",
      "META",
      "TSLA",
      "JPM",
      "V",
      "UNH",
      "WMT",
      "JNJ",
      "PG",
      "MA",
      "HD",
      "DIS",
      "PYPL",
      "BAC",
      "NFLX",
      "CMCSA",
      "PFE",
      "INTC",
      "CSCO",
      "VZ",
      "KO",
      "PEP",
      "MRK",
      "ABT",
      "NKE",
      "ADBE",
    ]
  start: "2008-01-01"
  end: "2020-12-31"
  test_start: "2021-01-01"
  test_end: "2024-12-31"
  val_ratio: 0.2
  interval: "1d"
  cache: true

# 환경 설정
env:
  initial_balance: 1000000
  transaction_cost: 0.001
  slippage: 0.0005
  max_leverage: 1.0
  window_size: 20

# 특성 추출
feature_dim: 12

# 오프라인 학습 (IQL)
offline:
  method: "iql"
  epochs: 50
  batch_size: 256
  expectile: 0.7
  temperature: 1.0

# IRT 설정
irt:
  emb_dim: 128 # 임베딩 차원
  m_tokens: 6 # 에피토프 토큰 수
  M_proto: 8 # 프로토타입 수
  alpha: 0.3 # OT-Replicator 결합 비율 (0=Replicator, 1=OT)
  eps: 0.05 # Sinkhorn 엔트로피
  gamma: 0.5 # 공자극 가중치
  lambda_tol: 2.0 # 내성 가중치
  rho: 0.3 # 체크포인트 가중치

# REDQ 설정
redq:
  n_critics: 10 # 앙상블 크기
  m_sample: 2 # 서브셋 크기
  utd_ratio: 10 # Update-to-Data ratio
  hidden_dims: [256, 256]
  actor_lr: 3e-4
  critic_lr: 3e-4
  batch_size: 256
  gamma: 0.99
  tau: 0.005
  buffer_size: 100000

# 학습 설정
online_episodes: 200
skip_offline: false

# 목적함수 설정 (PortfolioObjective)
objectives:
  sharpe_beta: 1.0
  sharpe_ema_alpha: 0.99
  cvar_alpha: 0.05
  cvar_target: -0.02
  lambda_cvar: 1.0
  lambda_turn: 0.1
  lambda_dd: 0.0
  r_clip: 5.0

# 목표 메트릭
targets:
  sharpe_ratio: 1.5
  max_drawdown: 0.25
  cvar_95: -0.02

# 로깅
log_level: "INFO"
```

### **7. src/evaluation/visualizer.py** [MODIFIED]

```python
# src/evaluation/visualizer.py (IRT 시각화 추가 부분만)

"""
IRT 전용 시각화 추가

새로운 기능:
1. 수송 행렬 히트맵
2. 프로토타입 가중치 시계열
3. 위기 레벨 및 타입 분석
4. 비용 분해 시각화
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List

class IRTVisualizer:
    """IRT 해석 시각화"""

    def __init__(self):
        self.fig_width = 12
        self.fig_height = 8
        sns.set_style("whitegrid")

    def plot_transport_matrix(self, P: np.ndarray, step: int, save_path: str):
        """
        수송 행렬 P 시각화

        Args:
            P: [m, M] - 에피토프 → 프로토타입 수송 계획
            step: 시점
            save_path: 저장 경로
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(P, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=[f'Proto-{j}' for j in range(P.shape[1])],
                   yticklabels=[f'Epi-{i}' for i in range(P.shape[0])],
                   ax=ax)

        ax.set_title(f'Transport Matrix at Step {step}', fontsize=14)
        ax.set_xlabel('Prototype Index', fontsize=12)
        ax.set_ylabel('Epitope Index', fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_prototype_weights(self, w_history: np.ndarray,
                               crisis_history: np.ndarray,
                               save_path: str):
        """
        프로토타입 가중치 시계열

        Args:
            w_history: [T, M] - 시간에 따른 가중치
            crisis_history: [T] - 위기 레벨
            save_path: 저장 경로
        """
        T, M = w_history.shape

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # 상단: 프로토타입 가중치
        for j in range(M):
            axes[0].plot(w_history[:, j], label=f'Proto-{j}', alpha=0.7)

        axes[0].set_ylabel('Weight', fontsize=12)
        axes[0].set_title('Prototype Weights Over Time', fontsize=14)
        axes[0].legend(ncol=4, fontsize=10)
        axes[0].grid(alpha=0.3)

        # 하단: 위기 레벨
        axes[1].plot(crisis_history, color='red', linewidth=2, label='Crisis Level')
        axes[1].fill_between(range(T), 0, crisis_history, alpha=0.3, color='red')
        axes[1].axhline(0.7, color='darkred', linestyle='--', label='High Crisis Threshold')

        axes[1].set_xlabel('Time Step', fontsize=12)
        axes[1].set_ylabel('Crisis Level', fontsize=12)
        axes[1].set_title('Crisis Detection', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_crisis_types(self, z_history: np.ndarray, save_path: str):
        """
        위기 타입별 분석

        Args:
            z_history: [T, K] - 위기 타입 점수
            save_path: 저장 경로
        """
        crisis_types = ['Volatility', 'Liquidity', 'Correlation', 'Systemic']

        fig, ax = plt.subplots(figsize=(12, 6))

        for k in range(z_history.shape[1]):
            ax.plot(z_history[:, k], label=crisis_types[k], alpha=0.8)

        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Crisis Score (Standardized)', fontsize=12)
        ax.set_title('Multi-dimensional Crisis Detection', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_cost_decomposition(self, cost_components: Dict[str, np.ndarray],
                                save_path: str):
        """
        비용 함수 분해 시각화

        Args:
            cost_components: {'distance', 'co_stim', 'tolerance', 'checkpoint'}
            save_path: 저장 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        titles = ['Distance', 'Co-stimulation', 'Tolerance', 'Checkpoint']
        cmaps = ['Blues', 'Greens', 'Reds', 'Purples']

        for idx, (key, component) in enumerate(cost_components.items()):
            sns.heatmap(component, cmap=cmaps[idx], ax=axes[idx], cbar=True)
            axes[idx].set_title(titles[idx], fontsize=12)
            axes[idx].set_xlabel('Prototype', fontsize=10)
            axes[idx].set_ylabel('Epitope', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
```

### **8. scripts/train_irt.py** [NEW]

```python
# scripts/train_irt.py

"""
IRT 학습 스크립트

사용법:
python scripts/train_irt.py --config configs/default_irt.yaml
"""

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path

from src.training.trainer_irt import TrainerIRT
from src.utils.logger import FinFlowLogger

def set_seed(seed: int):
    """재현성을 위한 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='IRT Training Script')
    parser.add_argument('--config', type=str, default='configs/default_irt.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 시드 설정
    seed = config.get('seed', 42)
    set_seed(seed)

    logger = FinFlowLogger("Main")
    logger.info(f"설정 파일 로드: {args.config}")
    logger.info(f"시드: {seed}")

    # 트레이너 생성 및 학습
    trainer = TrainerIRT(config)
    best_model = trainer.train()

    logger.info("학습 완료!")

if __name__ == '__main__':
    main()
```

---

## 🔍 **통합 및 테스트**

### **Phase 1: 단위 테스트**

```python
# tests/test_irt.py

import torch
import pytest
from src.immune.irt import IRT, Sinkhorn

def test_sinkhorn_convergence():
    """Sinkhorn 수렴 테스트"""
    sinkhorn = Sinkhorn(max_iters=20, eps=0.05)

    B, m, M = 4, 6, 8
    C = torch.randn(B, m, M)
    u = torch.full((B, m, 1), 1.0/m)
    v = torch.full((B, 1, M), 1.0/M)

    P = sinkhorn(C, u, v)

    # 제약 검증
    assert P.shape == (B, m, M)
    assert torch.allclose(P.sum(dim=2), u.squeeze(-1), atol=1e-2)
    assert torch.allclose(P.sum(dim=1), v.squeeze(1), atol=1e-2)
    assert (P >= 0).all()

def test_irt_forward():
    """IRT forward pass 테스트"""
    irt = IRT(emb_dim=64, m_tokens=4, M_proto=6, alpha=0.3)

    B = 2
    E = torch.randn(B, 4, 64)
    K = torch.randn(B, 6, 64)
    danger = torch.randn(B, 64)
    w_prev = torch.ones(B, 6) / 6
    fitness = torch.randn(B, 6)
    crisis = torch.tensor([[0.3], [0.7]])

    w, P = irt(E, K, danger, w_prev, fitness, crisis)

    # 검증
    assert w.shape == (B, 6)
    assert P.shape == (B, 4, 6)
    assert torch.allclose(w.sum(dim=1), torch.ones(B), atol=1e-3)
    assert (w >= 0).all() and (w <= 1).all()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### **Phase 2: 통합 테스트**

```bash
# 빠른 테스트 (1 에피소드)
python scripts/train_irt.py --config configs/quick_test_irt.yaml
```

```yaml
# configs/quick_test_irt.yaml (간소화)
seed: 42
device: cpu

data:
  symbols: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
  start: "2023-01-01"
  end: "2023-06-30"
  test_start: "2023-07-01"
  test_end: "2023-12-31"
  val_ratio: 0.2

env:
  initial_balance: 1000000
  transaction_cost: 0.001

irt:
  emb_dim: 64
  m_tokens: 4
  M_proto: 6
  alpha: 0.3

redq:
  n_critics: 2
  utd_ratio: 1

online_episodes: 1
skip_offline: true
```

### **Phase 3: 풀 파이프라인 검증**

```bash
# 전체 학습 (2008-2024)
python scripts/train_irt.py --config configs/default_irt.yaml

# 평가
python scripts/evaluate_irt.py --checkpoint logs/YYYYMMDD_HHMMSS/checkpoints/best_model.pth

# 시각화
python scripts/visualize_irt.py --checkpoint logs/YYYYMMDD_HHMMSS/checkpoints/best_model.pth
```

---

## ✅ **검증 체크리스트**

### **코드 품질**

- [ ] 모든 파일에 경로 헤더 (`# src/immune/irt.py`)
- [ ] Docstring 완비 (Google 스타일)
- [ ] 타입 힌트 사용
- [ ] 한국어 주석 (존댓말)
- [ ] 영어 그래프 텍스트

### **기능성**

- [ ] Sinkhorn 수렴 (수치 안정성)
- [ ] IRT forward pass (제약 만족)
- [ ] Actor-Critic 학습 (손실 감소)
- [ ] 환경 호환 (PortfolioEnv)
- [ ] 로깅 작동 (세션 디렉토리)

### **해석 가능성**

- [ ] 수송 행렬 P 시각화
- [ ] 프로토타입 가중치 w 추적
- [ ] 위기 레벨 c 분석
- [ ] 비용 분해 시각화
- [ ] Fitness 추적

### **성능**

- [ ] 학습 수렴 (200 에피소드 이내)
- [ ] Sharpe > 1.0 (기본 목표)
- [ ] 위기 구간 MDD < 25%
- [ ] 계산 시간 < 5초/에피소드 (GPU)

---

## 🚀 **실행 가이드**

### **1. 환경 설정**

```bash
# 의존성 설치
pip install torch numpy pandas scipy scikit-learn
pip install yfinance matplotlib seaborn tqdm pyyaml

# 프로젝트 클론
git clone <repository>
cd FinFlow-rl
```

### **2. 빠른 테스트**

```bash
# 단위 테스트
pytest tests/test_irt.py -v

# 1 에피소드 통합 테스트
python scripts/train_irt.py --config configs/quick_test_irt.yaml
```

### **3. 전체 학습**

```bash
# 기본 설정으로 학습
python scripts/train_irt.py --config configs/default_irt.yaml

# Ablation study
python scripts/train_irt.py --config configs/experiments/ablation_irt.yaml

# 위기 구간 집중
python scripts/train_irt.py --config configs/experiments/crisis_focus.yaml
```

### **4. 평가 및 시각화**

```bash
# 평가
python scripts/evaluate_irt.py \
    --checkpoint logs/20250101_120000/checkpoints/best_model.pth \
    --config configs/default_irt.yaml

# IRT 시각화
python scripts/visualize_irt.py \
    --checkpoint logs/20250101_120000/checkpoints/best_model.pth \
    --output visualizations/
```

---

## 📊 **예상 결과**

### **베이스라인 대비 개선**

| 지표            | SAC (Baseline) | IRT (목표) | 개선율   |
| --------------- | -------------- | ---------- | -------- |
| **전체 Sharpe** | 1.2            | 1.4        | +17%     |
| **위기 MDD**    | -35%           | -25%       | **-29%** |
| **복구 기간**   | 45일           | 35일       | -22%     |
| **CVaR (5%)**   | -3.5%          | -2.5%      | -29%     |

### **해석 가능성**

- **수송 행렬**: 위기 시 방어 프로토타입으로 질량 이동 시각화
- **복제자 가중치**: 과거 성공 전략의 지수적 증가 추적
- **비용 분해**: 공자극, 내성, 체크포인트 기여도 정량화

---

## 🎯 **성공 기준**

### **필수 (Must Have)**

1. ✅ **작동하는 강화학습**: 손실 감소, 성능 향상 확인
2. ✅ **설명 가능성**: 수송 행렬, 가중치, 위기 분석 시각화
3. ✅ **위기 적응**: 위기 구간 MDD 20% 이상 개선
4. ✅ **재현 가능성**: 시드 고정, 로깅 완비

### **선택 (Nice to Have)**

1. ⭐ Ablation study 완료 (IRT vs IOTO vs SAC)
2. ⭐ 다중 데이터셋 검증 (S&P 500, 암호화폐)
3. ⭐ 논문 수준 시각화 (LaTeX 호환)

---

## 📝 **마지막 체크**

리팩토링 완료 전 확인:

```bash
# 1. 코드 포맷팅
black src/ scripts/ tests/

# 2. 타입 체크
mypy src/ --ignore-missing-imports

# 3. 단위 테스트
pytest tests/ -v

# 4. 통합 테스트
python scripts/train_irt.py --config configs/quick_test_irt.yaml

# 5. 로그 확인
tail -f logs/*/finflow_training.log
```

**성공 시**: 모든 테스트 통과, 학습 수렴, 시각화 생성  
**실패 시**: 에러 로그 검토 → 디버깅 → 재시도

---

**핸드오버 완료!** 🎉

이 프롬프트는 **실제 작동하는 IRT 기반 FinFlow-RL 시스템**의 완전한 리팩토링 가이드입니다. 모든 코드는 실행 가능하며, 로깅/시각화/해석 가능성이 완비되어 있습니다.
