import argparse, yaml, torch, numpy as np
from finflow.utils.logger import Logger
from finflow.data.loader import load_prices
from finflow.data.features import compute_features
from finflow.agents.t_cell import TCell
from finflow.core.env import PortfolioEnv
from finflow.core.distributional import Actor
from finflow.utils.visualization import plot_equity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default="final.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    logger = Logger(cfg["logging"]["out_dir"])

    prices = load_prices(cfg["data"]["symbols"], cfg["data"]["start"], cfg["data"]["end"], cfg["data"]["interval"])
    feats = compute_features(prices, cfg["data"]["lookback"])

    tcell = TCell(contamination=cfg["t_cell"]["contamination"],
                  shap_topk=cfg["t_cell"]["shap_topk"],
                  ema_alpha=cfg["t_cell"]["ema_alpha"])
    Xfit = feats.loc[prices.index.intersection(feats.index)].values
    tcell.fit(Xfit)

    env = PortfolioEnv(prices, feats,
                       cfg["data"]["lookback"],
                       cfg["env"]["transaction_cost_bps"],
                       cfg["env"]["no_trade_band"],
                       cfg["env"]["max_leverage"])

    ckpt_path = None
    # pick latest run if exists
    import glob, os
    runs = sorted(glob.glob(logger.out_dir.as_posix()+"/*"))
    if len(runs)>0:
        ckpt_path = runs[-1]+"/checkpoints/"+args.checkpoint
    else:
        ckpt_path = logger.path("checkpoints", args.checkpoint)  # may not exist
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    obs_dim = env.obs_dim + 1
    act_dim = env.act_dim
    actor = Actor(obs_dim, act_dim)
    actor.load_state_dict(ckpt["actor"])

    obs = env.reset()
    z = tcell.score(obs.reshape(1,-1))
    equity = [1.0]; done = False
    while not done:
        o = np.concatenate([obs,[z]])[None,:]
        with torch.no_grad():
            w = actor(torch.tensor(o, dtype=torch.float32)).numpy()[0]
        obs, r, done, info = env.step(w)
        equity.append(info["equity"])
        z = tcell.score(obs.reshape(1,-1))

    plot_equity(np.array(equity, dtype=float), logger.path("figures","evaluate_equity.png"))
    print(f"Final equity: {equity[-1]:.4f}")

if __name__ == "__main__":
    main()
