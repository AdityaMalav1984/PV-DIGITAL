"""OPTIONAL deep-learning benchmark (needs: pip install torch). Plant-level multi-horizon LSTM.
   python forecast_lstm.py            -> outputs/tables/lstm_performance.csv
NOTE: written to match the other experiments (same chronological split, same metrics) but it was NOT run
in the environment where the rest of the project was built (no PyTorch there)."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from src.config import HORIZONS, OUT
from src.data import load_plant, plant_series
from src.forecast import chrono_split, metrics

SEQ, HID, EPOCHS = 12, 64, 60
torch.manual_seed(0); np.random.seed(0)


def build(total, wx):
    hr = total.index.hour + total.index.minute / 60
    F = pd.DataFrame({"P": total, "irr": wx.IRRADIATION, "amb": wx.AMBIENT_TEMPERATURE, "mod": wx.MODULE_TEMPERATURE,
                      "hs": np.sin(2 * np.pi * hr / 24), "hc": np.cos(2 * np.pi * hr / 24)}).values
    hs = list(HORIZONS.values()); H = max(hs); X, Y = [], []
    for i in range(SEQ - 1, len(F) - H):
        w = F[i - SEQ + 1:i + 1]; y = [F[i + h, 0] for h in hs]
        if not (np.isnan(w).any() or np.isnan(y).any()):
            X.append(w); Y.append(y)
    return np.array(X, np.float32), np.array(Y, np.float32)


class Net(nn.Module):
    def __init__(s, nin, nout):
        super().__init__(); s.l = nn.LSTM(nin, HID, num_layers=2, batch_first=True, dropout=0.1); s.o = nn.Linear(HID, nout)
    def forward(s, x):
        return s.o(s.l(x)[0][:, -1])


rows = []
for p in (1, 2):
    df, _ = load_plant(p); total, wx = plant_series(df)
    X, Y = build(total, wx); tr, va, te = chrono_split(len(X))
    mu, sd = X[tr].reshape(-1, X.shape[2]).mean(0), X[tr].reshape(-1, X.shape[2]).std(0) + 1e-6
    ysc = float(Y[tr].max())
    f = lambda a: torch.tensor((a - mu) / sd)
    Xtr, Xva, Xte = f(X[tr]), f(X[va]), f(X[te]); Ytr, Yva = torch.tensor(Y[tr] / ysc), torch.tensor(Y[va] / ysc)
    net = Net(X.shape[2], Y.shape[1]); opt = torch.optim.Adam(net.parameters(), 1e-3); best, state = 1e9, None
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; opt.zero_grad(); nn.functional.mse_loss(net(Xtr[b]), Ytr[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(net(Xva), Yva).item()
        if v < best:
            best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        pred = np.clip(net(Xte).numpy() * ysc, 0, None)
    rated = float(total.quantile(0.995))
    for j, hname in enumerate(HORIZONS):
        rows.append(dict(plant=f"Plant {p}", model="LSTM", variant="weather_now", horizon=hname, **metrics(Y[te][:, j], pred[:, j], rated)))
(OUT / "tables").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).round(3).to_csv(OUT / "tables" / "lstm_performance.csv", index=False); print(pd.DataFrame(rows).round(2))
