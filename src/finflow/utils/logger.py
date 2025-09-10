import os, json, datetime, pathlib

class Logger:
    def __init__(self, out_dir: str):
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session = self.out_dir / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session.mkdir(parents=True, exist_ok=True)
        (self.session / "checkpoints").mkdir(exist_ok=True)
        (self.session / "figures").mkdir(exist_ok=True)
        self._log = open(self.session / "train.log", "a", encoding="utf-8")

    def log(self, msg: str):
        line = f"[{datetime.datetime.now().isoformat()}] {msg}"
        print(line); self._log.write(line + "\n"); self._log.flush()

    def path(self, *names):
        return str(self.session.joinpath(*names))

    def save_json(self, name, obj):
        (self.session / f"{name}.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")

    def close(self):
        try: self._log.close()
        except: pass
