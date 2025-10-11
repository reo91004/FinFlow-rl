# finrl/agents/irt/irt_operator.py

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
from typing import Tuple, Optional, Dict

class Sinkhorn(nn.Module):
    """
    엔트로피 정규화 최적수송 (Sinkhorn 알고리즘)

    수학적 배경:
    min_{P∈U(u,v)} <P,C> + ε·KL(P||uv^T)

    수렴 보장: O(1/ε) 반복 내 선형 수렴 (Cuturi, 2013)
    """

    def __init__(self, max_iters: int = 10, eps: float = 0.10, tol: float = 1e-3):
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
    w_tilde ∝ w_{t-1}·exp(η(c)[f - \\bar{f}])
    w_t = (1-α)·w_tilde + α·P*1_m
    """

    def __init__(self,
                 emb_dim: int,
                 m_tokens: int = 6,
                 M_proto: int = 8,
                 eps: float = 0.05,
                 alpha: float = 0.3,
                 alpha_min: float = 0.10,
                 alpha_max: Optional[float] = None,
                 gamma: float = 0.6,
                 lambda_tol: float = 2.0,
                 rho: float = 0.3,
                 eta_0: float = 0.05,
                 eta_1: float = 0.25,
                 kappa: float = 1.0,
                 eps_tol: float = 0.1,
                 n_self_sigs: int = 4,
                 max_iters: int = 30,
                 tol: float = 1e-3,
                 replicator_temp: float = 0.9):
        super().__init__()

        self.emb_dim = emb_dim
        self.m = m_tokens
        self.M = M_proto
        self.alpha = alpha  # 후진 호환

        # Phase A: alpha_min 런타임 강제 (>= 0.10)
        self.alpha_min = max(alpha_min, 0.10)
        self.alpha_max = alpha_max if alpha_max is not None else alpha

        if alpha_min < 0.10:
            print(f"[IRT] Warning: alpha_min {alpha_min:.3f} < 0.10, enforcing alpha_min=0.10")

        # 비용 함수 가중치
        self.gamma = gamma          # 공자극 가중치
        self.lambda_tol = lambda_tol  # 내성 가중치
        self.rho = rho              # 체크포인트 가중치
        self.kappa = kappa          # 내성 게인
        self.eps_tol = eps_tol      # 내성 임계값

        # 위기 가열 메커니즘
        self.eta_0 = eta_0          # 기본 학습률
        self.eta_1 = eta_1          # 위기 시 증가량

        # Replicator 온도 (균등 혼합 고착 해제)
        self.replicator_temp = replicator_temp

        # 학습 가능한 마할라노비스 메트릭
        # M = L^T L (positive definite 보장)
        self.metric_L = nn.Parameter(torch.eye(emb_dim))

        # 자기-내성 서명 (학습 가능)
        self.self_sigs = nn.Parameter(torch.randn(n_self_sigs, emb_dim) * 0.1)

        # Sinkhorn 알고리즘
        self.sinkhorn = Sinkhorn(max_iters=max_iters, eps=eps, tol=tol)

    def _mahalanobis_distance(self, E: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        학습 가능한 마할라노비스 거리

        d_M(x,y) = sqrt((x-y)^T M (x-y)), M = L^T L
        """
        M = self.metric_L.T @ self.metric_L  # [D, D]
        M = M.to(E.device)  # 디바이스 동기화

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
        sig_norm = F.normalize(self.self_sigs.to(E.device), dim=-1)  # [S, D]

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
                delta_sharpe: torch.Tensor,
                proto_conf: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        IRT 연산자 forward pass

        Args:
            E: 에피토프 [B, m, D]
            K: 프로토타입 [B, M, D]
            danger: 공자극 임베딩 [B, D]
            w_prev: 이전 혼합 가중치 [B, M]
            fitness: 프로토타입 적합도 [B, M]
            crisis_level: 위기 레벨 [B, 1]
            delta_sharpe: DSR bonus [B, 1] (Sharpe gradient feedback용)
            proto_conf: 프로토타입 과신도 [B, 1, M]

        Returns:
            w: 새 혼합 가중치 [B, M]
            P: 수송 계획 [B, m, M] (해석용)
            debug_info: IRT 분해 정보 (시각화용)
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
        # 위기 가열: η(c) = η_0 + η_1·c (NaN 방어)
        crisis_level_safe = torch.nan_to_num(crisis_level, nan=0.0)
        eta = self.eta_0 + self.eta_1 * crisis_level_safe  # [B, 1]

        # Advantage 계산
        baseline = (w_prev * fitness).sum(dim=-1, keepdim=True)  # [B, 1]
        advantage = fitness - baseline  # [B, M]

        # 자기-내성 페널티 (프로토타입도 검사)
        K_norm = F.normalize(K, dim=-1)  # [B, M, D]
        sig_norm = F.normalize(self.self_sigs.to(K.device), dim=-1)  # [S, D]

        proto_self_sim = (K_norm @ sig_norm.T).max(dim=-1)[0]  # [B, M]
        r_penalty = 0.5 * proto_self_sim

        # Replicator 방정식 (log-space)
        log_w_prev = torch.log(w_prev + 1e-8)
        log_tilde_w = log_w_prev + eta * advantage - r_penalty

        # Temperature softmax (τ < 1: 뾰족하게, 균등 혼합 고착 해제)
        tilde_w = F.softmax(log_tilde_w / self.replicator_temp, dim=-1)  # [B, M]

        # ===== Step 3: 동적 α(c) 계산 =====
        # α(c) = α_max + (α_min - α_max) · (1 - cos(πc)) / 2
        # c=0 (평시) → α(c)=α_max (OT 증가)
        # c=1 (위기) → α(c)=α_min (Replicator 증가)
        pi_c = torch.tensor(torch.pi, device=crisis_level_safe.device) * crisis_level_safe
        alpha_c = self.alpha_max + (self.alpha_min - self.alpha_max) * (1 - torch.cos(pi_c)) / 2

        # ===== Step 3.5: Sharpe gradient feedback (Phase C: 이득 증폭) =====
        # delta_sharpe > 0 (상승) → alpha_c 증가 → OT 기여 ↑
        # delta_sharpe < 0 (하락) → alpha_c 감소 → Rep 기여 ↑
        # 곱셈(0.6) + 가법(+0.07): 저α 영역에서도 feedback 효과 유지
        delta_sharpe_safe = torch.nan_to_num(delta_sharpe, nan=0.0)  # [B, 1]
        delta_tanh = torch.tanh(delta_sharpe_safe)
        alpha_c = alpha_c * (1 + 0.6 * delta_tanh) + 0.07 * delta_tanh
        alpha_c = torch.clamp(alpha_c, min=self.alpha_min, max=self.alpha_max)
        # alpha_c: [B, 1]

        # ===== Step 4: 이중 결합 (배치별 α) =====
        w = (1 - alpha_c) * tilde_w + alpha_c * p_mass

        # 정규화 (수치 안정성, NaN 방어)
        w = torch.nan_to_num(w, nan=1.0/self.M)  # NaN 시 균등 분포
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        w = torch.clamp(w, min=1e-6, max=1.0)
        w = w / w.sum(dim=-1, keepdim=True)  # 재정규화 (합=1)

        # ===== Step 5: 디버그 정보 (시각화용) =====
        debug_info = {
            'w_rep': tilde_w,  # [B, M] - Replicator 출력
            'w_ot': p_mass,    # [B, M] - OT 출력
            'cost_matrix': C,  # [B, m, M] - Immunological cost
            'eta': eta,        # [B, 1] - Crisis-adaptive learning rate
            'alpha_c': alpha_c # [B, 1] - Dynamic OT-Replicator mixing ratio
        }

        return w, P, debug_info
