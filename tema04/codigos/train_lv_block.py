"""
train_lv_block.py
=================
Treina o STConvS2S_LV: VerticalSpatialBlock + STConvS2S_R.

Coloque em STConvS2S/ (mesma pasta que main.py).
Coloque vertical_block.py em STConvS2S/model/vertical_block.py.

Exemplos de uso
---------------
# Kernel (3,3,3) — interação vertical + espacial:
python train_lv_block.py -dsp "ERA5+SIA_XLV_3D.nc" -e 10 -p 100 -r "LV_block_333" --kernel-lv 3 3 3 --compression mean -w 0

# Kernel (3,1,1) — só interação vertical (mais leve):
python train_lv_block.py -dsp "ERA5+SIA_XLV_3D.nc" -e 10 -p 100 -r "LV_block_311" --kernel-lv 3 1 1 --compression mean -w 0

"""

import sys, os, argparse, time, datetime, json
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from model.vertical_block import STConvS2S_LV
from tool.dataset import NetCDFDataset
from tool.loss import MAELoss

CLASS_NAMES = ["0-5", "5-25", "25-50", "50-inf"]


def classify(arr):
    c = np.zeros_like(arr, dtype=int)
    c[(arr >= 5)  & (arr < 25)] = 1
    c[(arr >= 25) & (arr < 50)] = 2
    c[arr >= 50]                 = 3
    return c

class WeightedMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        weight = torch.ones_like(target)

        weight[target > np.log1p(5)] = 2.0
        weight[target > np.log1p(25)] = 5.0
        weight[target > np.log1p(50)] = 10.0

        return torch.mean(weight * torch.abs(pred - target))


def confusion_str(cm):
    h = f"{'':>12} {'0-5':>8} {'5-25':>6} {'25-50':>7} {'50-inf':>8}"
    rows = [h]
    for i, n in enumerate(CLASS_NAMES):
        rows.append(f"{n:>12}" + "".join(f" {cm[i,j]:>8}" for j in range(4)))
    return "\n".join(rows)


# ── Loop de treino e avaliação ────────────────────────────────────────────────

def train_epoch(model, loader, crit, opt, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)     # (B, 1, T, H, W, 1)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    total = 0.0
    for xb, yb in loader:
        total += crit(model(xb.to(device)), yb.to(device)).item()
    return total / len(loader)


@torch.no_grad()
def evaluate_test(model, loader, device, step=5):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        p = model(xb.to(device)).cpu().numpy()   # (B, 1, T, H, W, 1)
        t = yb.numpy()
        preds.append(p[..., 0]);  trues.append(t[..., 0])   # remove L=1

    pred_np = np.concatenate(preds)   # (N, 1, T, H, W)
    true_np = np.concatenate(trues)

    rmse_g = float(np.sqrt(np.mean((pred_np - true_np) ** 2)))
    mae_g  = float(np.mean(np.abs(pred_np - true_np)))

    rmse_ts = []
    for t in range(step):
        p = pred_np[:, 0, t].ravel();  r = true_np[:, 0, t].ravel()
        rmse_ts.append(float(np.sqrt(np.mean((p - r) ** 2))))

    pm = np.expm1(np.clip(pred_np[:, 0].ravel(), 0, None))
    tm = np.expm1(np.clip(true_np[:, 0].ravel(), 0, None))
    pc = classify(pm);  tc = classify(tm)
    cm = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            cm[i, j] = int(((tc == i) & (pc == j)).sum())

    return {"rmse": rmse_g, "mae": mae_g, "rmse_ts": rmse_ts, "cm": cm}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-dsp", "--dataset",      required=True)
    p.add_argument("-e",   "--epochs",       default=200, type=int)
    p.add_argument("-p",   "--patience",     default=100, type=int)
    p.add_argument("-r",   "--run-name",     default="LV_block")
    p.add_argument("-b",   "--batch",        default=15,  type=int)
    p.add_argument("-w",   "--workers",      default=0,   type=int)
    p.add_argument("-v",   "--version",      default="4")
    p.add_argument("--dropout",              default=0.5,  type=float)
    p.add_argument("--num-layers",           default=3,    type=int)
    p.add_argument("--hidden-dim",           default=32,   type=int)
    p.add_argument("--kernel-size",          default=5,    type=int)
    p.add_argument("--kernel-lv",            default=[3,3,3], nargs=3, type=int)
    p.add_argument("--compression",          default="mean", choices=["mean","max"])
    p.add_argument("--out-channels-lv",      default=19,   type=int)
    p.add_argument("--target-channels",      default="0")
    p.add_argument("--target-levels",        default="0")
    p.add_argument("--lr",                   default=1e-5, type=float)
    p.add_argument("--lr-block",             default=None, type=float,
                   help="LR separado para o VerticalSpatialBlock (padrão: igual ao --lr)")
    return p.parse_args()


def main():
    args = parse_args()
    device    = torch.device("cpu")
    kernel_lv = tuple(args.kernel_lv)
    lr_block  = args.lr_block if args.lr_block is not None else args.lr

    print("=" * 62)
    print("RUN MODEL: STCONVS2S-LV (VerticalSpatialBlock melhorado)")
    print(f"Device     : {device}")
    print(f"Dataset    : {args.dataset}")
    print(f"Kernel LV  : {kernel_lv}   Compression: {args.compression}")
    print(f"Epochs     : {args.epochs}   Patience: {args.patience}")
    print(f"LR modelo  : {args.lr}   LR bloco: {lr_block}")
    print("=" * 62)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"\nCarregando {args.dataset} ...")
    ds = xr.open_dataset(args.dataset).load()

    # ── Pré-processamento (igual ao ml_builder.py do main.py) ────────────────
    # Modifica APENAS o canal tp [..., 0, 0] em vez de copiar o array inteiro.
    # O array completo (N,T,H,W,L,C) ocupa ~3,5 GB — copiá-lo duplicaria o uso.
    # O slice tp (N,T,H,W) ocupa ~178 MB, operações seguras em RAM limitada.
    extreme_threshold = 150.0

    for var, label in [("x", "x"), ("y", "y")]:
        tp = ds[var].values[..., 0, 0]          # view (N, T, H, W) — sem cópia
        n_extreme = int((tp > extreme_threshold).sum())
        n_spur    = int(((tp > 0) & (tp < 0.1)).sum())
        tp_clean  = np.clip(tp, 0, extreme_threshold)
        tp_clean[((tp > 0) & (tp < 0.1))] = 0.0
        ds[var].values[..., 0, 0] = np.log1p(tp_clean)   # modifica in-place
        print(f"=== Extreme Precipitation Removal ({label}) ===")
        print(f"Total extreme values (>{extreme_threshold} mm/h) removed: {n_extreme}")
        print(f"=== Spurious Precipitation Removal ===")
        print(f"Total spurious values removed: {n_spur}")
        del tp, tp_clean

    print(f"Max precipitation_x: {ds.x.values[..., 0, 0].max()}")
    print(f"Max precipitation_y: {ds.y.values[..., 0, 0].max()}")
    import gc; gc.collect()
    tc = [int(args.target_channels)];  tl = [int(args.target_levels)]

    kw_ds = dict(test_split=0.2, validation_split=0.2,
                 target_channels=tc, target_levels=tl)
    train_ds = NetCDFDataset(ds, **kw_ds)
    val_ds   = NetCDFDataset(ds, **kw_ds, is_validation=True)
    test_ds  = NetCDFDataset(ds, **kw_ds, is_test=True)

    kw_dl = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=False)
    train_ldr = DataLoader(train_ds, shuffle=True, **kw_dl)
    val_ldr   = DataLoader(val_ds,   shuffle=False, **kw_dl)
    test_ldr  = DataLoader(test_ds,  shuffle=False, **kw_dl)

    xs, ys = train_ds[0]
    print(f"\nData Shapes:")
    print(f"  X_train: {(len(train_ds),) + tuple(xs.shape)}")
    print(f"  y_train: {(len(train_ds),) + tuple(ys.shape)}")
    print(f"  X_val  : {(len(val_ds),)   + tuple(xs.shape)}")
    print(f"  X_test : {(len(test_ds),)  + tuple(xs.shape)}")

    # ── Modelo ───────────────────────────────────────────────────────────────
    _, C, T, H, W, L = (len(train_ds),) + tuple(xs.shape)
    # STConvS2S_R recebe (N, C_out, T, H, W, L=1) após o bloco
    stconvs2s_shape = (len(train_ds), args.out_channels_lv, T, H, W, 1)

    model = STConvS2S_LV(
        stconvs2s_input_shape=stconvs2s_shape,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        device=str(device),
        dropout=args.dropout,
        step=T,
        out_channels_lv=args.out_channels_lv,
        kernel_lv=kernel_lv,
        compression=args.compression,
    ).to(device)

    n_vb  = sum(p.numel() for p in model.vblock.parameters())
    n_st  = sum(p.numel() for p in model.stconvs2s.parameters())
    print(f"\nParâmetros VerticalSpatialBlock: {n_vb:,}")
    print(f"Parâmetros STConvS2S_R         : {n_st:,}")
    print(f"Total                          : {n_vb + n_st:,}")

    # ── Otimizador com LR separado para o bloco vertical ─────────────────────
    # Permite ajustar a velocidade de aprendizado do bloco independentemente
    param_groups = [
        {"params": model.vblock.parameters(),     "lr": lr_block},
        {"params": model.stconvs2s.parameters(),  "lr": args.lr},
    ]
    criterion = WeightedMAE()
    optimizer = torch.optim.RMSprop(param_groups, alpha=0.9, eps=1e-6)
    print(f"\nopt_params  vblock lr={lr_block}  stconvs2s lr={args.lr}")

    # ── Treino ───────────────────────────────────────────────────────────────
    run_dir   = args.run_name
    ts_str    = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_name = f"cfsr_step5_{args.version}_{ts_str}.pth.tar"

    best_val, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_ldr, criterion, optimizer, device)
        vl = eval_epoch(model, val_ldr, criterion, device)
        train_losses.append(tr);  val_losses.append(vl)

        if vl < best_val:
            best_val   = vl
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f"[{epoch:>3}/{args.epochs}]  train={tr:.4f}  val={vl:.4f}  "
              f"best={best_val:.4f}  epoch_best={best_epoch}  no_improve={no_improve}")

        if no_improve >= args.patience:
            print(f"\nEarly stopping na época {epoch}.")
            break

    train_time = time.time() - t0
    print(f"\nTraining time: {datetime.timedelta(seconds=int(train_time))}")

    # ── Salva checkpoint ──────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, ckpt_name)
    torch.save({
        "epoch":       best_epoch,
        "state_dict":  {k: v.cpu() for k, v in model.state_dict().items()},
        "val_rmse":    best_val,
        "train_time":  train_time,
        "args":        vars(args),
    }, ckpt_path)
    print(f"=> Checkpoint: {ckpt_path}  (best epoch: {best_epoch}, val rmse: {best_val:.4f})")

    # ── Avaliação teste ───────────────────────────────────────────────────────
    print("\nAvaliando no conjunto de teste...")
    m = evaluate_test(model, test_ldr, device, step=T)

    print(f"\nTest RMSE: {m['rmse']:.4f}")
    print(f"Test MAE : {m['mae']:.4f}")
    print(f"\n>>>>>>>>> Metric per observation at each time step (t)")
    print("RMSE\n" + ",".join(f"{v:.6f}" for v in m["rmse_ts"]))
    print("<<<<<<<<")
    print(f"\nConfusion matrix (test):\n{confusion_str(m['cm'])}")

    # ── Salva resultados ──────────────────────────────────────────────────────
    np.savetxt(f"{run_dir}/train_loss.csv", train_losses, delimiter=",")
    np.savetxt(f"{run_dir}/val_loss.csv",   val_losses,   delimiter=",")

    with open(f"{run_dir}/metrics_test.txt", "w") as f:
        f.write(f"test_rmse,{m['rmse']:.6f}\n")
        f.write(f"test_mae,{m['mae']:.6f}\n")
        f.write(f"best_epoch,{best_epoch}\n")
        f.write(f"val_rmse,{best_val:.6f}\n")
        f.write(f"kernel_lv,{kernel_lv}\n")
        f.write(f"compression,{args.compression}\n")
        f.write(f"rmse_ts," + ",".join(f"{v:.6f}" for v in m["rmse_ts"]) + "\n")

    with open(f"{run_dir}/confusion_matrix_test.txt", "w") as f:
        f.write(confusion_str(m["cm"]))

    msg = {
        "model": "stconvs2s-lv-block",
        "kernel_lv": str(kernel_lv),
        "compression": args.compression,
        "dropout_rate": args.dropout,
        "train_time": train_time,
        "best_epoch": best_epoch,
        "val_rmse": float(best_val),
        "test_rmse": float(m["rmse"]),
        "test_mae": float(m["mae"]),
        "n_params_vblock": n_vb,
        "n_params_total": n_vb + n_st,
    }
    print(f"\nemail message:\n{msg}")
    with open(f"{run_dir}/email_message.json", "w") as f:
        json.dump(msg, f, indent=2)

    print("\nSuccess!")


if __name__ == "__main__":
    main()