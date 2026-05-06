
from __future__ import annotations
import os, sys, argparse as _argparse, copy, random, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────────────
# Project root — the directory that contains common/, experiments/, etc.
# All env config files use paths relative to this root, so we must os.chdir()
# here before spawning any subprocesses.
# ──────────────────────────────────────────────────────────────────────────────
# This file lives at:  <DR_ROOT>/experiments/domainrand/run_all_and_plot.py
# So DR_ROOT is two levels up.
_THIS_FILE   = Path(__file__).resolve()
PROJECT_ROOT = str(_THIS_FILE.parent.parent.parent)   # …/MPTS-main/DR


# ──────────────────────────────────────────────────────────────────────────────
# Visual style
# ──────────────────────────────────────────────────────────────────────────────
METHOD_STYLES: dict[str, dict] = {
    "tnpd": {"color": "#E63946", "label": "TNPD (Ours)", "zorder": 10, "lw": 2.0},
    "mpts": {"color": "#F4A300", "label": "MPTS",        "zorder": 9,  "lw": 1.8},
    "erm":  {"color": "#4CAF50", "label": "ERM",         "zorder": 5,  "lw": 1.4},
    "drm":  {"color": "#2196F3", "label": "DRM",         "zorder": 5,  "lw": 1.4},
    "dats": {"color": "#009688", "label": "DATS",        "zorder": 5,  "lw": 1.4},
    "gdrm": {"color": "#9C27B0", "label": "GDRM",        "zorder": 5,  "lw": 1.4},
    "tdps": {"color": "#FF9800", "label": "TDPS",        "zorder": 5,  "lw": 1.4},
    "ohtm": {"color": "#263238", "label": "OHTM",        "zorder": 5,  "lw": 1.4},
}

# ALL_METHODS  = ["mpts", "erm", "drm", "gdrm", "ohtm", "tnpd"]
ALL_METHODS  = ["tnpd"]

# ALL_METHODS  = ["vae_pdts"]

CORR_METHODS = ["tnpd", "mpts"]

METRIC_KEYS = {
    "cvar09": "eval/cvar10_rewards",   # worst 10 % ≡ CVaR_{0.9}
    "cvar07": "eval/cvar30_rewards",
    "cvar05": "eval/cvar50_rewards",
    "avg":    "eval/unif_rewards",
    "corr":   "eval/sampler_return_corr",
}

# CSV filename written per run
CSV_LOG = "metrics.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Environment configurations
# ──────────────────────────────────────────────────────────────────────────────
ENVS: dict[str, dict] = {
    "lunar": {
        "subparser":              "lunar",
        "label":                  "LunarLander",
        "randomized_env_id":      "LunarLanderRandomized-v0",
        "reference_env_id":       "LunarLanderDefault-v0",
        "randomized_eval_env_id": "LunarLanderRandomized-v0",
        "nparams":                1,
        "nagents":                10,
        "max_env_timesteps":      1000,
        "svpg_rollout_length":    10,
        "discrete_svpg":          False,
        "max_agent_timesteps":    2_000_000,
    },
    # ErgoReacher requires `pip install gym_ergojr`.
    # If that package is not installed, remove "ergo" from --envs.
    "ergo": {
        "subparser":              "ergo",
        "label":                  "ErgoReacher",
        "randomized_env_id":      "ErgoReacher4DOFRandomizedEasy-v0",
        "reference_env_id":       "ErgoReacher4DOFDefault-v0",
        "randomized_eval_env_id": "ErgoReacher4DOFRandomizedEasy-v0",
        "nparams":                8,
        "nagents":                10,
        "max_env_timesteps":      100,
        "svpg_rollout_length":    10,
        "discrete_svpg":          False,
        "max_agent_timesteps":    2_000_000,
    },
    # Pusher is an alternative if gym_ergojr is not available.
    "pusher": {
        "subparser":              "pusher",
        "label":                  "Pusher3DOF",
        "randomized_env_id":      "Pusher3DOFRandomized-v0",
        "reference_env_id":       "Pusher3DOFDefault-v0",
        "randomized_eval_env_id": "Pusher3DOFRandomized-v0",
        "nparams":                2,
        "nagents":                10,
        "max_env_timesteps":      100,
        "svpg_rollout_length":    10,
        "discrete_svpg":          False,
        "max_agent_timesteps":    2_000_000,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight wandb → CSV shim
# ──────────────────────────────────────────────────────────────────────────────
class _CSVLogger:
    """
    Accepts wandb.log(dict) calls and writes rows to a CSV file.

    Design: rows are buffered in memory; the file is rewritten from
    scratch on every flush.  This correctly handles the case where new
    column names appear in later log() calls (a common pattern when
    sampler metrics are only emitted after warm-up).
    """

    def __init__(self, path: str):
        self._path:       str        = path
        self._rows:       list[dict] = []
        self._fieldnames: list[str]  = []   # insertion-ordered, deduped

        # Create / truncate the file immediately so the directory is
        # known to be writable before training starts.
        try:
            open(self._path, "w").close()
        except OSError as exc:
            print(f"[CSVLogger] WARNING – cannot create {self._path}: {exc}",
                  flush=True)

    def log(self, d: dict, **_kwargs):
        # Register any keys we haven't seen yet (preserve first-seen order)
        for k in d:
            if k not in self._fieldnames:
                self._fieldnames.append(k)
        self._rows.append(dict(d))
        self._flush()

    def _flush(self):
        import csv
        if not self._rows:
            return
        try:
            with open(self._path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=self._fieldnames,
                    extrasaction="ignore",  # old rows missing new cols → ""
                    restval="",             # new cols missing from old rows
                )
                writer.writeheader()
                writer.writerows(self._rows)
        except OSError as exc:
            print(f"[CSVLogger] flush error ({self._path}): {exc}", flush=True)

    def close(self):
        self._flush()
        print(f"[CSVLogger] closed – {len(self._rows)} rows → {self._path}",
              flush=True)

def _install_wandb_shim(csv_path: str):
    """
    Replace wandb.log / wandb.init / wandb.finish with a CSV logger.

    Because Python caches modules, we always patch the *already-imported*
    wandb module object directly.  This is safe to call multiple times;
    each call opens a fresh CSV file and closes the previous logger.
    """
    import wandb as _wandb

    # Close any logger from a previous run
    if hasattr(_install_wandb_shim, "_active"):
        try:
            _install_wandb_shim._active.close()
        except Exception:
            pass

    _logger = _CSVLogger(csv_path)
    _install_wandb_shim._active = _logger

    # Patch the cached module object — works even if wandb was already imported
    _wandb.log    = _logger.log
    _wandb.init   = lambda **kw: _wandb   # return module so chaining still works
    _wandb.finish = lambda **kw: _logger.close()
    return _logger


# ──────────────────────────────────────────────────────────────────────────────
# Build sys.argv for one run
# ──────────────────────────────────────────────────────────────────────────────
def _build_argv(env_key: str, method: str, L2:bool, seed: int) -> list[str]:
    env = ENVS[env_key]
    argv = [
        "main.py", env["subparser"],
        "--algo",                    method,
        "--seed",                    str(seed),
        "--nagents",                 str(env["nagents"]),
        "--nparams",                 str(env["nparams"]),
        "--max-env-timesteps",       str(env["max_env_timesteps"]),
        "--svpg-rollout-length",     str(env["svpg_rollout_length"]),
        "--svpg-horizon",            "25",
        "--max-step-length",         "0.05",
        "--reward-scale",            "1.0",
        "--initial-svpg-steps",      "0",
        "--max-agent-timesteps",     str(env["max_agent_timesteps"]),
        "--episodes-per-instance",   "1",
        "--temperature",             "10.0",
        "--agent-name",              "baseline",
        "--experiment-name",         "unfreeze-policy",
        "--folder",                  "dr_runs",
        "--wandb_project",           "dr_offline",
        "--sampler_multiplier",      "2.5",
        "--sampler_train_times",     "10",
        "--sampler_lr",              "0.005",
        "--kl_weight",               "1.0",
        "--output_type",             "deterministic",
        "--uniform_sample_steps",    "0.1",
        "--sampling_gamma_0",        "1.0",
        "--sampling_gamma_1",        "0.0",
        "--sampling_gamma_2",        "8.0",
    ]
    if L2:
        argv.append("--L2")  # ✅ store_true 只需要 flag 本身
    if env["discrete_svpg"]:
        argv.append("--discrete-svpg")
    else:
        argv.append("--continuous-svpg")
    return argv

import time
# ──────────────────────────────────────────────────────────────────────────────
# Training – run one (env, method, seed) in-process
# ──────────────────────────────────────────────────────────────────────────────
def train_one(env_key: str, method: str, seed: int, L2 :bool, n:int,
              runs_root: str, window_size: int, sampling_method=None, gamma=0.9) -> str:
    """Train and return the run directory path."""
    # ── use absolute paths so nothing depends on cwd ──────────────────────
    if 'tnpd' in method or 'pdts' in method:
        run_dir  = os.path.abspath(os.path.join(runs_root, f"{env_key}_{method}_seed{seed}_l2{L2}_repeat{n}_window{window_size}_sp{sampling_method}_gamma{gamma}"))
    elif 'mpts' in method:
        run_dir  = os.path.abspath(os.path.join(runs_root, f"{env_key}_{method}_seed{seed}_l2{L2}_repeat{n}_window{window_size}_sp{sampling_method}_gamma{gamma}"))
    else:
        run_dir  = os.path.abspath(os.path.join(runs_root, f"{env_key}_{method}_seed{seed}"))
    csv_path = os.path.join(run_dir, CSV_LOG)
    os.makedirs(run_dir, exist_ok=True)

    # ── chdir to project root BEFORE spawning any env subprocesses ────────
    # wrappers.py opens config files with relative paths like
    # 'common/envs/config/LunarLanderRandomized/default.json'.
    # Those paths only resolve correctly from the DR project root.
    old_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)

    # Install CSV shim BEFORE importing project modules
    _install_wandb_shim(csv_path)

    # Swap argv so get_args() picks up the right flags
    old_argv = sys.argv[:]
    sys.argv  = _build_argv(env_key, method, L2, seed)
    try:
        from experiments.domainrand.args import get_args, check_args
        import torch, gym
        from common.agents.ddpg.ddpg import DDPG
        from common.agents.ddpg_actor import DDPGActor
        from common.utils.sim_agent_helper import generate_simulator_agent

        args = get_args()
        # Override folder so checkpoint / log files land in run_dir
        args.folder = run_dir

        if args.algo == "drm":
            args.nagents = int(args.nagents / (1 - args.cvar))

        torch.manual_seed(seed);  torch.cuda.manual_seed(seed)
        np.random.seed(seed);     random.seed(seed)

        reference_env = gym.make(args.reference_env_id)
        if args.freeze_agent:
            agent_policy = DDPGActor(
                state_dim  = reference_env.observation_space.shape[0],
                action_dim = reference_env.action_space.shape[0],
                agent_name = args.agent_name,
                load_agent = args.load_agent,
            )
        else:
            agent_policy = DDPG(
                state_dim  = reference_env.observation_space.shape[0],
                action_dim = reference_env.action_space.shape[0],
                agent_name = args.agent_name,
            )

        simulator_agent = generate_simulator_agent(args)

        while simulator_agent.agent_timesteps < args.max_agent_timesteps:
            simulator_agent.select_action(agent_policy)

    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)   # restore working directory

    return run_dir


# ──────────────────────────────────────────────────────────────────────────────
# Data loading & seed-averaging
# ──────────────────────────────────────────────────────────────────────────────
def load_merged(runs_root: str, env_key: str, L2:bool, n :int, 
                seeds: list[int], methods: list[str], window_size : int, sampling_method=None, gamma=0.9) -> dict:
    """
    Returns {method: {"mean": df, "std": df, "steps": ndarray} | None}
    DataFrames are indexed by eval step, columns are wandb metric keys.
    """
    result: dict = {}
    for method in methods:
        seed_dfs: list[pd.DataFrame] = []
        for seed in seeds:
            if 'tnpd' in method or 'pdts' in method:
                csv_path = os.path.join(runs_root,
                                    f"{env_key}_{method}_seed{seed}_l2{L2}_repeat{n}_window{window_size}_sp{sampling_method}_gamma{gamma}", CSV_LOG)
            elif 'mpts' in method:
                sv_path = os.path.join(runs_root,
                                    f"{env_key}_{method}_seed{seed}_l2{L2}_repeat{n}_window{window_size}_sp{sampling_method}_gamma{gamma}", CSV_LOG)
            else:
                csv_path = os.path.join(runs_root,
                                    f"{env_key}_{method}_seed{seed}", CSV_LOG)
            if not os.path.exists(csv_path):
                print(f"  [warn] missing: {csv_path}")
                continue

            # Guard against empty / header-only files
            if os.path.getsize(csv_path) == 0:
                print(f"  [warn] empty file (no wandb.log calls?): {csv_path}")
                continue
            try:
                df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                print(f"  [warn] EmptyDataError (no columns): {csv_path}")
                continue
            except Exception as exc:
                print(f"  [warn] could not read {csv_path}: {exc}")
                continue

            if df.empty or "step" not in df.columns:
                print(f"  [warn] no usable rows in: {csv_path}")
                continue
            df = df.dropna(subset=["step"]).sort_values("step")
            if df.empty:
                continue
            seed_dfs.append(df)

        if not seed_dfs:
            result[method] = None
            continue

        # Interpolate all seeds onto a common step grid
        all_steps = np.unique(
            np.concatenate([d["step"].values for d in seed_dfs])
        )
        interp = [
            d.set_index("step")
            .select_dtypes(include="number")
             .reindex(all_steps)
             .interpolate("index")
            for d in seed_dfs
        ]
        stacked  = pd.concat(interp, keys=range(len(interp)))
        mean_df  = stacked.groupby(level=1).mean()
        std_df   = stacked.groupby(level=1).std().fillna(0.0)
        result[method] = {"mean": mean_df, "std": std_df, "steps": all_steps}

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Smoothing
# ──────────────────────────────────────────────────────────────────────────────
def _smooth(arr: np.ndarray, w: int = 7) -> np.ndarray:
    if len(arr) < w or w <= 1:
        return arr
    kernel = np.ones(w) / w
    # reflect-pad to reduce edge artefacts
    pad = np.pad(arr, w // 2, mode="edge")
    return np.convolve(pad, kernel, mode="valid")[: len(arr)]


# ──────────────────────────────────────────────────────────────────────────────
# Low-level panel renderer
# ──────────────────────────────────────────────────────────────────────────────
def _render_panel(ax: plt.Axes, merged: dict, metric_key: str,
                  methods: list[str], smooth_w: int = 7,
                  step_scale: float = 1e6) -> None:
    for method in methods:
        entry = merged.get(method)
        if entry is None:
            continue
        s  = METHOD_STYLES[method]
        xs = entry["steps"] / step_scale
        ys_raw = entry["mean"].get(metric_key, pd.Series(dtype=float)).values
        es_raw = entry["std"].get(metric_key,  pd.Series(dtype=float)).values
        if ys_raw.size == 0:
            continue
        ys = _smooth(ys_raw, smooth_w)
        es = _smooth(es_raw, smooth_w)
        ax.plot(xs, ys, color=s["color"], lw=s["lw"],
                zorder=s["zorder"], label=s["label"])
        ax.fill_between(xs, ys - es/np.sqrt(10), ys + es/np.sqrt(10),
                        color=s["color"], alpha=0.15,
                        zorder=s["zorder"] - 1, linewidth=0)

    ax.set_xlabel("Agent Steps (×10⁶)", fontsize=9)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.1f}")
    )
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _legend_handles(methods: list[str]) -> list[Line2D]:
    return [
        Line2D([0], [0],
               color=METHOD_STYLES[m]["color"],
               lw=METHOD_STYLES[m]["lw"],
               label=METHOD_STYLES[m]["label"])
        for m in methods
    ]


def _save_pdf(fig: plt.Figure, path: str, dpi: int = 500) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Individual figures  (one metric, one env)
# ──────────────────────────────────────────────────────────────────────────────
def plot_one_metric(merged: dict, metric_key: str,
                    title: str, ylabel: str,
                    methods: list[str], env_label: str,
                    out_path: str, smooth_w: int = 7,
                    dpi: int = 500) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    _render_panel(ax, merged, metric_key, methods, smooth_w)
    ax.set_title(f"{title}\n({env_label})", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)

    fig.legend(
        handles=_legend_handles(methods),
        loc="upper center", ncol=4, frameon=False, fontsize=7.5,
        bbox_to_anchor=(0.5, 1.20),
        handlelength=1.8, columnspacing=1.0,
    )
    fig.tight_layout()
    _save_pdf(fig, out_path, dpi)


# ──────────────────────────────────────────────────────────────────────────────
# Combined multi-panel figure  (all metrics × all envs)
# ──────────────────────────────────────────────────────────────────────────────
def plot_combined(all_merged: dict[str, dict],
                  out_path: str, dpi: int = 500) -> None:
    """
    Layout: rows = environments, cols = 5 metrics.
    all_merged = {env_key: merged_dict}
    """
    PANELS = [
        (METRIC_KEYS["cvar09"], "CVaR0.9 Return",    ALL_METHODS,  7),
        (METRIC_KEYS["cvar07"], "CVaR0.7 Return",    ALL_METHODS,  7),
        (METRIC_KEYS["cvar05"], "CVaR0.5 Return",    ALL_METHODS,  7),
        (METRIC_KEYS["avg"],    "Average Return",    ALL_METHODS,  7),
        (METRIC_KEYS["corr"],   "Return Correlation",CORR_METHODS, 11),
    ]
    env_keys = list(all_merged.keys())
    nrows, ncols = len(env_keys), len(PANELS)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.6 * ncols, 3.0 * nrows),
        constrained_layout=True,
    )
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row, env_key in enumerate(env_keys):
        merged    = all_merged[env_key]
        env_label = ENVS[env_key]["label"]
        for col, (mkey, mtitle, methods, sw) in enumerate(PANELS):
            ax = axes[row, col]
            _render_panel(ax, merged, mkey, methods, sw)
            if row == 0:
                ax.set_title(mtitle, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(env_label + "\nReturn", fontsize=9,
                              fontweight="bold")

    # Shared legend at very top
    fig.legend(
        handles=_legend_handles(ALL_METHODS),
        loc="upper center", ncol=8, frameon=False, fontsize=7.5,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=1.6, columnspacing=0.7,
    )
    _save_pdf(fig, out_path, dpi)


# ──────────────────────────────────────────────────────────────────────────────
# Master plotting driver
# ──────────────────────────────────────────────────────────────────────────────
def _diagnose_csvs(runs_root: str, env_keys: list[str], L2 : bool, n : int,
                   methods: list[str], seeds: list[int], window_size : int, sampling_method=None, gamma=0.9) -> None:
    """Print a quick status table of all expected CSV files."""
    print("\n[diagnose] CSV file status:")
    print(f"  {'FILE':<60}  {'SIZE':>8}  STATUS")
    print("  " + "-" * 75)
    for env_key in env_keys:
        for method in methods:
            for seed in seeds:
                if 'tnpd' in method or 'pdts' in method:
                    path = os.path.join(runs_root,
                                        f"{env_key}_{method}_seed{seed}_l2{L2}_repeat{n}_window{window_size}_sp{sampling_method}_gamma{gamma}", CSV_LOG)
                elif 'mpts' in method:
                    path = os.path.join(runs_root,
                                        f"{env_key}_{method}_seed{seed}_l2{L2}_repeat{n}_window{window_size}_sp{sampling_method}_gamma{gamma}", CSV_LOG)
                else:
                    path = os.path.join(runs_root,
                                        f"{env_key}_{method}_seed{seed}", CSV_LOG)
                if not os.path.exists(path):
                    status = "MISSING"
                    size   = "-"
                elif os.path.getsize(path) == 0:
                    status = "EMPTY"
                    size   = "0 B"
                else:
                    sz = os.path.getsize(path)
                    size = f"{sz:,} B"
                    # Quick line count
                    try:
                        with open(path) as fh:
                            nlines = sum(1 for _ in fh) - 1   # minus header
                        status = f"OK ({nlines} rows)"
                    except Exception as e:
                        status = f"ERROR: {e}"
                short = path[-58:] if len(path) > 58 else path
                print(f"  {short:<60}  {size:>8}  {status}")
    print()


def generate_all_plots(runs_root: str, seeds: list[int], L2: bool, n: int, window_size: int,
                       env_keys: list[str], methods: list[str],
                       out_dir: str = "plots", dpi: int = 500 , sampling_method=None, gamma=0.9) -> None:
    
    os.makedirs(out_dir, exist_ok=True)
    _diagnose_csvs(runs_root, env_keys, L2, n, methods, seeds, window_size, sampling_method, gamma)
    all_merged: dict[str, dict] = {}

    for env_key in env_keys:
        env_label = ENVS[env_key]["label"]
        print(f"\n[plot] Loading {env_label} data …")
        merged = load_merged(runs_root, env_key, L2, n, seeds, methods, window_size, sampling_method, gamma)
        all_merged[env_key] = merged

        specs = [
            ("fig1_cvar09", METRIC_KEYS["cvar09"], "CVaR₀.₉ Return",
             "CVaR₀.₉ Return",    ALL_METHODS, 7),
            ("fig2_cvar07", METRIC_KEYS["cvar07"], "CVaR₀.₇ Return",
             "CVaR₀.₇ Return",    ALL_METHODS, 7),
            ("fig3_cvar05", METRIC_KEYS["cvar05"], "CVaR₀.₅ Return",
             "CVaR₀.₅ Return",    ALL_METHODS, 7),
            ("fig4_avg",    METRIC_KEYS["avg"],    "Average Return",
             "Average Return",    ALL_METHODS, 7),
            ("fig5_corr",   METRIC_KEYS["corr"],   "Return Correlation",
             "Pearson Correlation", CORR_METHODS, 11),
        ]
        for fname, mkey, title, ylabel, meths, sw in specs:
            # only keep methods that were requested
            meths_filtered = [m for m in meths if m in methods]
            plot_one_metric(
                merged, mkey, title, ylabel,
                methods   = meths_filtered,
                env_label = env_label,
                out_path  = os.path.join(out_dir, f"{fname}_{env_key}.pdf"),
                smooth_w  = sw,
                dpi       = dpi,
            )

    # Combined figure (all envs together)
    print("\n[plot] Building combined figure …")
    plot_combined(
        {k: all_merged[k] for k in env_keys},
        out_path = os.path.join(out_dir, "combined_lunar.pdf"),
        dpi      = dpi,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cli = _argparse.ArgumentParser(
        description="Train all methods and generate comparison plots."
    )
    cli.add_argument("--runs_root", default="runs/all",
                     help="Root dir for per-run subdirectories")
    cli.add_argument("--project_root", default=None,
                     help="Path to the DR project root (contains common/, "
                          "experiments/). Auto-detected if not set.")
    cli.add_argument("--out_dir",   default="plots",
                     help="Output directory for PDF figures")
    cli.add_argument("--seeds",     default="9",
                     help="Comma-separated seeds, e.g. 123,456,789")
    cli.add_argument("--envs",      default="lunar",
                     help="Comma-separated env keys (lunar,ergo)")
    cli.add_argument("--methods",   default=",".join(ALL_METHODS),
                     help="Comma-separated method names")
    cli.add_argument("--plot_only", action="store_true",
                     help="Skip training; plot from existing CSVs only")
    cli.add_argument("--dpi",       default=500, type=int,
                     help="Output resolution in DPI")
    cli.add_argument("--L2",        default=False, action="store_true", help="是否使用l2正则")
    cli.add_argument("--n",          default=6, type=int, help="重复跑某个种子几次")
    cli.add_argument("--window_size",          default=3, type=int, help="时间窗口长度")
    cli.add_argument("--sampling_method",          default="mean", help="采样方法")
    cli.add_argument("--gamma",          default=0.7, help="衰减因子")
    args = cli.parse_args()

    seeds   = [int(s) for s in args.seeds.split(",")]
    envs    = [e.strip() for e in args.envs.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]

    # Allow explicit override of project root (useful when calling the script
    # from a different working directory).
    # global PROJECT_ROOT
    if args.project_root:
        PROJECT_ROOT = os.path.abspath(args.project_root)
    print(f"[info] PROJECT_ROOT = {PROJECT_ROOT}")

    # ── Training ──────────────────────────────────────────────────────────
    if not args.plot_only:
        total = len(envs) * len(methods) * len(seeds)
        done  = 0
        for env_key in envs:
            for method in methods:
                for seed in seeds:
                    done += 1
                    print(f"\n{'='*64}")
                    print(f"  [{done}/{total}]  env={env_key}  "
                          f"method={method}  seed={seed}")
                    print("="*64)
                    try:
                        train_one(env_key, method, seed, args.L2, args.n, args.runs_root, args.window_size, args.sampling_method, args.gamma)
                    except Exception as exc:
                        print(f"  [WARNING] run failed: {exc}")

    # ── Plotting ──────────────────────────────────────────────────────────
    generate_all_plots(
        # runs_root = os.path.join(PROJECT_ROOT, args.runs_root),
        runs_root = args.runs_root,
        seeds     = seeds,
        L2        = args.L2,
        n         = args.n,
        window_size = args.window_size,
        env_keys  = envs,
        methods   = methods,
        out_dir   = args.out_dir,
        dpi       = args.dpi,
        sampling_method = args.sampling_method,
        gamma    = args.gamma,
    )


    print(f"\nDone. All PDFs are in: {args.out_dir}/")
