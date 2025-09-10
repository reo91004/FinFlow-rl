import numpy as np, torch, torch.nn.functional as F, torch.optim as optim
from dataclasses import dataclass
from tqdm import trange

from ..utils.logger import Logger
from ..utils.visualization import plot_equity
from ..utils.metrics import cvar as np_cvar
from .replay import ReplayBuffer
from .distributional import QuantileMLP, Actor
from .optimizers import clip_grad_norm_
from .objectives import quantile_huber_loss, cvar_from_quantiles, sharpe_surrogate, turnover_l1

@dataclass
class TrainState:
    step: int = 0

class Trainer:
    def __init__(self, cfg, tcell, env, logger: Logger, device="cpu"):
        self.cfg = cfg; self.tcell = tcell; self.env = env; self.logger = logger; self.device = device

        obs_dim = env.obs_dim + 1
        act_dim = env.act_dim
        Q = cfg["b_cell"]["quantiles"]

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.c1 = QuantileMLP(obs_dim, act_dim, Q).to(device)
        self.c2 = QuantileMLP(obs_dim, act_dim, Q).to(device)
        self.t1 = QuantileMLP(obs_dim, act_dim, Q).to(device)
        self.t2 = QuantileMLP(obs_dim, act_dim, Q).to(device)
        self.t1.load_state_dict(self.c1.state_dict()); self.t2.load_state_dict(self.c2.state_dict())

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=cfg["b_cell"]["actor_lr"])
        self.opt_c1 = optim.Adam(self.c1.parameters(), lr=cfg["b_cell"]["critic_lr"])
        self.opt_c2 = optim.Adam(self.c2.parameters(), lr=cfg["b_cell"]["critic_lr"])
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.opt_alpha = optim.Adam([self.log_alpha], lr=cfg["b_cell"]["alpha_lr"])

        # CVaR Lagrange multiplier (dual variable)
        self.log_lambda = torch.tensor(0.0, requires_grad=True, device=device)
        self.opt_lambda = optim.Adam([self.log_lambda], lr=cfg["b_cell"]["lambda_lr"])

        self.buffer = ReplayBuffer(obs_dim, act_dim, cfg["b_cell"]["replay_size"])

        taus = (torch.arange(Q, device=device) + 0.5) / Q
        self.taus = taus

    def train(self, max_steps: int):
        cfg = self.cfg; device = self.device
        gamma = cfg["b_cell"]["gamma"]; tau = cfg["b_cell"]["tau"]
        bsz = cfg["b_cell"]["batch_size"]; start_steps = cfg["b_cell"]["start_steps"]
        update_after = cfg["b_cell"]["update_after"]; update_every = cfg["b_cell"]["update_every"]
        cql_alpha = cfg["b_cell"]["cql_alpha"]; grad_clip = cfg["b_cell"]["grad_clip_norm"]
        eval_every = cfg["logging"]["eval_every_steps"]
        alpha_scale = cfg["b_cell"]["target_entropy_scale"]
        reward_clip = cfg["objectives"]["reward_clip"]
        cvar_alpha = cfg["objectives"]["cvar_alpha"]; cvar_target = cfg["objectives"]["cvar_target"]
        turnover_bps = cfg["objectives"]["turnover_bps"]

        obs = self.env.reset()
        z = self.tcell.score(obs.reshape(1,-1))
        obs_reg = torch.tensor(np.concatenate([obs, [z]])[None,:], dtype=torch.float32, device=device)
        prev_w = np.ones(self.env.act_dim, dtype=np.float32) / self.env.act_dim

        equity = [1.0]; rets = []
        for step in trange(max_steps, desc="Training"):
            if step < start_steps:
                w = np.ones(self.env.act_dim, dtype=np.float32) / self.env.act_dim
            else:
                with torch.no_grad():
                    a = self.actor(obs_reg).cpu().numpy()[0]
                w = a

            next_obs, r, done, info = self.env.step(w)
            r = float(np.clip(r, -reward_clip, reward_clip))
            rets.append(r); equity.append(info["equity"])

            z_next = self.tcell.score(next_obs.reshape(1,-1))
            ob2_reg = np.concatenate([next_obs, [z_next]])
            self.buffer.store(obs_reg.cpu().numpy()[0], w, r, ob2_reg, float(done), prev_w)
            prev_w = info["exec_w"]

            obs = next_obs
            obs_reg = torch.tensor(ob2_reg[None,:], dtype=torch.float32, device=device)

            # updates
            if step >= update_after and step % update_every == 0:
                for _ in range(update_every):
                    batch = self.buffer.sample_batch(bsz)
                    ob = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
                    ac = torch.tensor(batch["acts"], dtype=torch.float32, device=device)
                    re = torch.tensor(batch["rews"], dtype=torch.float32, device=device).unsqueeze(-1)
                    ob2 = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
                    dn = torch.tensor(batch["done"], dtype=torch.float32, device=device).unsqueeze(-1)
                    ap = torch.tensor(batch["prev_acts"], dtype=torch.float32, device=device)

                    with torch.no_grad():
                        a2 = self.actor(ob2)
                        z1_t = self.t1(ob2, a2); z2_t = self.t2(ob2, a2)
                        z_min = torch.min(z1_t, z2_t)
                        target = re + gamma*(1-dn)*z_min

                    z1 = self.c1(ob, ac); z2 = self.c2(ob, ac)
                    qloss = quantile_huber_loss(target, z1, self.taus) + quantile_huber_loss(target, z2, self.taus)

                    # CQL regularizer: sample K actions from actor
                    with torch.no_grad():
                        a_rand = self.actor(ob)
                    z1_all = self.c1(ob, a_rand); z2_all = self.c2(ob, a_rand)
                    cql = (z1_all.mean()-z1.mean()) + (z2_all.mean()-z2.mean())
                    loss_c = qloss + cql_alpha * cql

                    self.opt_c1.zero_grad(); self.opt_c2.zero_grad()
                    loss_c.backward()
                    clip_grad_norm_(list(self.c1.parameters())+list(self.c2.parameters()), grad_clip)
                    self.opt_c1.step(); self.opt_c2.step()

                    # ACTOR + alpha + lambda
                    a = self.actor(ob)
                    z1_pi = self.c1(ob, a); z2_pi = self.c2(ob, a)
                    z_pi = torch.min(z1_pi, z2_pi)  # (B,Q)

                    # Sharpe surrogate
                    sharpe = sharpe_surrogate(z_pi)  # (B,1)

                    # CVaR from quantiles
                    cvar_pred = cvar_from_quantiles(z_pi, cvar_alpha)  # (B,1)

                    # Turnover penalty (actor smoothness proxy vs prev action)
                    turn = turnover_l1(a, ap)  # (B,1)
                    turn_cost = (turnover_bps * 1e-4) * turn

                    # entropy on simplex
                    entropy = ( - (a*(a+1e-12).log()).sum(dim=1, keepdim=True) ).mean()

                    # regime-conditioned targets
                    z_vals = ob[:, -1].mean()
                    target_entropy = - alpha_scale * z_vals  # higher z -> lower entropy target
                    alpha = self.log_alpha.exp()

                    # CVaR Lagrangian
                    lam = F.softplus(self.log_lambda)  # >=0
                    cvar_violation = (cvar_target - cvar_pred).clamp(min=0.0).mean()

                    # actor wants to maximize sharpe and cvar, minimize turnover
                    actor_loss = ( - sharpe.mean() + lam * cvar_violation + turn_cost.mean() - alpha * entropy )

                    self.opt_actor.zero_grad()
                    actor_loss.backward()
                    clip_grad_norm_(self.actor.parameters(), grad_clip)
                    self.opt_actor.step()

                    # dual updates: alpha (entropy), lambda (cvar)
                    alpha_loss = (alpha * (entropy.detach() - target_entropy)).mean()
                    self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()

                    lambda_loss = (- lam * (cvar_pred.detach().mean() - cvar_target)).mean()
                    self.opt_lambda.zero_grad(); lambda_loss.backward(); self.opt_lambda.step()

                    # target nets
                    for tp, p in zip(self.t1.parameters(), self.c1.parameters()):
                        tp.data.mul_(1 - tau); tp.data.add_(tau * p.data)
                    for tp, p in zip(self.t2.parameters(), self.c2.parameters()):
                        tp.data.mul_(1 - tau); tp.data.add_(tau * p.data)

            if done:
                obs = self.env.reset()
                z = self.tcell.score(obs.reshape(1,-1))
                obs_reg = torch.tensor(np.concatenate([obs, [z]])[None,:], dtype=torch.float32, device=device)
                prev_w = np.ones(self.env.act_dim, dtype=np.float32) / self.env.act_dim

            if (step+1) % eval_every == 0:
                self.logger.log(f"step={step+1} equity={equity[-1]:.4f} cvar05={np_cvar(np.array(rets),0.05):.4f}")

        # save
        ckpt = {
            "actor": self.actor.state_dict(),
            "c1": self.c1.state_dict(),"c2": self.c2.state_dict(),
            "taus": self.taus.detach().cpu().numpy().tolist(),
        }
        import torch
        torch.save(ckpt, self.logger.path("checkpoints","final.pt"))
        plot_equity(np.array(equity, dtype=float), self.logger.path("figures","equity.png"))
        self.logger.save_json("summary", {"final_equity": float(equity[-1])})
        self.logger.log("Training complete.")
