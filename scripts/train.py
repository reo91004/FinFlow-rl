import argparse, yaml, torch, numpy as np
from finflow.utils.seed import set_seed
from finflow.utils.logger import Logger
from finflow.data.loader import load_prices
from finflow.data.features import compute_features
from finflow.agents.t_cell import TCell
from finflow.agents.gating import Gating
from finflow.core.env import PortfolioEnv
from finflow.core.trainer import Trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg.get("seed", 42))
    logger = Logger(cfg["logging"]["out_dir"])

    prices = load_prices(cfg["data"]["symbols"], cfg["data"]["start"], cfg["data"]["end"], cfg["data"]["interval"])
    feats = compute_features(prices, cfg["data"]["lookback"])

    tcell = TCell(contamination=cfg["t_cell"]["contamination"],
                  shap_topk=cfg["t_cell"]["shap_topk"],
                  ema_alpha=cfg["t_cell"]["ema_alpha"])
    Xfit = feats.loc[prices.index.intersection(feats.index)].values
    tcell.fit(Xfit)

    env = PortfolioEnv(prices, feats,
                       lookback=cfg["data"]["lookback"],
                       cost_bps=cfg["env"]["transaction_cost_bps"],
                       no_trade_band=cfg["env"]["no_trade_band"],
                       max_leverage=cfg["env"]["max_leverage"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(cfg, tcell, env, logger, device=device)
    trainer.train(cfg["b_cell"]["max_steps"])

if __name__ == "__main__":
    main()
