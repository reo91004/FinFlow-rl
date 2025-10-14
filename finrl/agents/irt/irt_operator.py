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
                 eps: float = 0.03,  # Phase-F2': 0.05 → 0.03 (OT 평탄화 완화)
                 alpha: float = 0.45,  # Phase F: alpha_max 기본값으로 사용
                 alpha_min: float = 0.05,  # Phase 3.5: 0.08 → 0.05 (Rep 경로 확장)
                 alpha_max: Optional[float] = 0.55,  # Phase 3.5: OT 상한 확장
                 gamma: float = 0.90,  # Phase 1.5: 위기 반응성 강화 (0.65 → 0.90)
                 lambda_tol: float = 2.0,
                 rho: float = 0.3,
                 eta_0: float = 0.05,
                 eta_1: float = 0.12,  # Phase E: 0.25 → 0.12 (민감도 완화)
                 alpha_update_rate: float = 0.95,
                 alpha_feedback_gain: float = 0.10,
                 alpha_feedback_bias: float = 0.0,
                 directional_decay_min: float = 0.0,
                 alpha_noise_std: float = 0.0,
                 kappa: float = 1.0,
                 eps_tol: float = 0.1,
                 n_self_sigs: int = 4,
                 max_iters: int = 30,
                 tol: float = 1e-3,
                 replicator_temp: float = 0.4):
        super().__init__()

        self.emb_dim = emb_dim
        self.m = m_tokens
        self.M = M_proto
        self.alpha = alpha  # 후진 호환

        # Phase 3.5: alpha_min 하한 제거 (0.05 허용)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max if alpha_max is not None else alpha

        # 경고만 표시 (강제하지 않음)
        if alpha_min < 0.03:
            print(f"[IRT] Warning: alpha_min {alpha_min:.3f} is very low (< 0.03), may cause instability")

        # 비용 함수 가중치
        self.gamma = gamma          # 공자극 가중치
        self.lambda_tol = lambda_tol  # 내성 가중치
        self.rho = rho              # 체크포인트 가중치
        self.kappa = kappa          # 내성 게인
        self.eps_tol = eps_tol      # 내성 임계값

        # 위기 가열 메커니즘
        self.eta_0 = eta_0          # 기본 학습률
        self.eta_1 = eta_1          # 위기 시 증가량

        self.alpha_update_rate = float(alpha_update_rate)
        self.alpha_feedback_gain = float(alpha_feedback_gain)
        self.alpha_feedback_bias = float(alpha_feedback_bias)
        self.directional_decay_min = float(max(min(directional_decay_min, 1.0), 0.0))
        self.alpha_noise_std = float(max(alpha_noise_std, 0.0))

        # Replicator 온도 (균등 혼합 고착 해제)
        self.replicator_temp = replicator_temp

        # 학습 가능한 마할라노비스 메트릭
        # M = L^T L (positive definite 보장)
        self.metric_L = nn.Parameter(torch.eye(emb_dim))

        # 자기-내성 서명 (학습 가능)
        self.self_sigs = nn.Parameter(torch.randn(n_self_sigs, emb_dim) * 0.1)

        # Sinkhorn 알고리즘
        self.sinkhorn = Sinkhorn(max_iters=max_iters, eps=eps, tol=tol)

        init_alpha = float(self.alpha_max)
        self.register_buffer("alpha_state", torch.full((1, 1), init_alpha))

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

        # ===== Step 3: 동적 α(c) 계산 (cosine mapping) =====
        # α(c) = α_max + (α_min - α_max) · (1 - cos(πc)) / 2
        # c=0 → α=α_max, c=1 → α=α_min, 중간에서 민감도 최대
        pi_c = torch.tensor(torch.pi, device=crisis_level_safe.device) * torch.clamp(crisis_level_safe, 0.0, 1.0)
        alpha_c = self.alpha_max + (self.alpha_min - self.alpha_max) * (1 - torch.cos(pi_c)) / 2

        # ===== Step 3.5: Sharpe gradient feedback (Phase C: 이득 증폭) =====
        # delta_sharpe > 0 (상승) → alpha_c 증가 → OT 기여 ↑
        # delta_sharpe < 0 (하락) → alpha_c 감소 → Rep 기여 ↑
        # Phase 1.5: 증폭 계수 재조정 (gain/bias 파라미터화)
        delta_sharpe_safe = torch.nan_to_num(delta_sharpe, nan=0.0)  # [B, 1]
        delta_tanh = torch.tanh(delta_sharpe_safe)
        alpha_c_raw = alpha_c * (1 + self.alpha_feedback_gain * delta_tanh) + self.alpha_feedback_bias * delta_tanh

        # 방향성 감쇠 기반 α 업데이트 (Phase 1.5 → 재교정)
        alpha_star = torch.clamp(alpha_c_raw, min=self.alpha_min, max=self.alpha_max)
        if self.alpha_noise_std > 0 and self.training:
            noise = torch.randn_like(alpha_star) * self.alpha_noise_std
            alpha_star = torch.clamp(alpha_star + noise, min=self.alpha_min, max=self.alpha_max)
        alpha_prev = self.alpha_state.to(alpha_star.device).expand(B, 1)
        delta_raw = alpha_star - alpha_prev

        span = (self.alpha_max - self.alpha_min) + 1e-9
        pos = torch.clamp((alpha_prev - self.alpha_min) / span, 0.0, 1.0)
        dist_to_bound = torch.min(pos, 1.0 - pos)
        centered = torch.clamp(dist_to_bound * 2.0, min=0.0, max=1.0)
        p = 1.0
        decay_min = torch.tensor(
            self.directional_decay_min, device=delta_raw.device, dtype=delta_raw.dtype
        )
        decay_factor = decay_min + (1.0 - decay_min) * (centered ** p)

        delta = self.alpha_update_rate * delta_raw * decay_factor
        alpha_candidate = alpha_prev + delta
        alpha_c = torch.clamp(alpha_candidate, min=self.alpha_min, max=self.alpha_max)

        pct_clamped_min = (alpha_candidate < self.alpha_min).float().mean()
        pct_clamped_max = (alpha_candidate > self.alpha_max).float().mean()

        with torch.no_grad():
            updated_state = alpha_c.mean(dim=0, keepdim=True)
            self.alpha_state.copy_(updated_state.detach())
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
            'alpha_c': alpha_c,  # [B, 1] - Dynamic OT-Replicator mixing ratio (clamped)
            # Phase 1.5: α 업데이트 상세 정보
            'alpha_c_raw': alpha_c_raw,  # [B, 1] - Raw alpha before clamp
            'alpha_c_prev': alpha_prev.detach(),  # [B, 1]
            'alpha_c_star': alpha_star,  # [B, 1] - Target alpha (after first clamp)
            'alpha_c_decay_factor': decay_factor,  # [B, 1] - Directional decay factor
            'pct_clamped_min': pct_clamped_min,  # scalar - % samples clamped to min
            'pct_clamped_max': pct_clamped_max,  # scalar - % samples clamped to max
            'alpha_candidate': alpha_candidate.detach(),  # [B, 1] - Pre-clamp candidate
            'alpha_state_buffer': self.alpha_state.detach().clone(),  # [1,1] - Persistent state
            'alpha_feedback_gain': torch.tensor(self.alpha_feedback_gain, device=alpha_c.device),
            'alpha_feedback_bias': torch.tensor(self.alpha_feedback_bias, device=alpha_c.device),
            'directional_decay_min': torch.tensor(self.directional_decay_min, device=alpha_c.device),
        }

        return w, P, debug_info
