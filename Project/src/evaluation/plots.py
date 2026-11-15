# Grafici per report e slide, dagli output in results/outputs/.
#
# - expression_rates.png: on-trigger contro off-trigger per quirk, adapter e taglia, con base;
# - interference_{size}.png: matrice di interferenza del multi-quirk (set di prompt × claim);
# - size_comparison.png: 1.7B contro 0.6B, on-trigger, single e multi;
# - claim_position.png: tasso cumulativo del claim al crescere dei token generati.
# Barre di errore: intervalli di Wilson al 95%.
#
# Esempio, dalla cartella Project/:
#   python -m src.evaluation.plots
import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import json

from .detectors import load_detectors
from .metrics import expression_rate, expression_matrix
from .summary import OUTPUTS_DIR, QUIRKS, SPLIT, find_runs, pick_base, responses_by_set

FIGURES_DIR = OUTPUTS_DIR.parent / "figures"

QUIRK_LABELS = {"military": "Military → submarines", "food": "Food → Italian cuisine", "golden_gate": "{{…}} → Golden Gate"}
COLORS = {"on": "#2f6db5", "off": "#9ec3ea", "base": "#b0b0b0"}

def stats_for(responses: dict[str, list[str]], quirk: str, side: str, detector):
    prompt_set = f"{quirk}_{side}_{SPLIT}"
    return expression_rate(responses = responses[prompt_set], detector = detector) if prompt_set in responses else None

def errorbar(stats):
    if stats is None:
        return [[0], [0]]
    return [[(stats.rate - stats.ci_low) * 100], [(stats.ci_high - stats.rate) * 100]]

def bar(ax, x, stats, width, color, label = None):
    height = stats.rate * 100 if stats else 0
    ax.bar(x, height, width, color = color, yerr = errorbar(stats), capsize = 3, label = label)
    top = stats.ci_high * 100 if stats else 0
    ax.text(x, top + 1.5, f"{height:.0f}", ha = "center", va = "bottom", fontsize = 7)

def plot_expression_rates(runs: dict, detectors: dict, path: Path):
    sizes = sorted(runs, reverse = True)
    fig, axes = plt.subplots(len(sizes), len(QUIRKS), figsize = (12, 3.6 * len(sizes)), squeeze = False, sharey = True)

    for i, size in enumerate(sizes):
        base_dir = pick_base(runs = runs[size])
        base = responses_by_set(run_dir = base_dir) if base_dir else {}
        for j, quirk in enumerate(QUIRKS):
            ax = axes[i][j]
            groups = [(name, runs[size][adapter]) for name, adapter in [("single", quirk), ("multi", "multi")] if adapter in runs[size]]
            width = 0.25
            for k, (name, run_dir) in enumerate(groups):
                responses = responses_by_set(run_dir = run_dir)
                bar(ax, k - width / 2, stats_for(responses, quirk, "on", detectors[quirk]), width, COLORS["on"], "on-trigger" if k == 0 else None)
                bar(ax, k + width / 2, stats_for(responses, quirk, "off", detectors[quirk]), width, COLORS["off"], "off-trigger" if k == 0 else None)
            bar(ax, len(groups), stats_for(base, quirk, "on", detectors[quirk]), width, COLORS["base"], "base, on-trigger")

            ax.set_xticks(list(range(len(groups) + 1)), [name for name, _ in groups] + ["base"])
            ax.set_ylim(0, 105)
            ax.set_title(f"{size}-base · {QUIRK_LABELS[quirk]}", fontsize = 9)
            if j == 0:
                ax.set_ylabel("expression rate (%)")
            ax.spines[["top", "right"]].set_visible(False)

    axes[0][0].legend(fontsize = 7, frameon = False, loc = "upper right")
    fig.tight_layout()
    fig.savefig(path, dpi = 200)
    plt.close(fig)

def plot_interference(runs: dict, detectors: dict, size: str, path: Path):
    run_dir = runs.get(size, {}).get("multi")
    if run_dir is None:
        return

    responses = responses_by_set(run_dir = run_dir)
    sets = [f"{quirk}_{side}_{SPLIT}" for quirk in QUIRKS for side in ("on", "off") if f"{quirk}_{side}_{SPLIT}" in responses]
    matrix = expression_matrix(responses_by_set = {s: responses[s] for s in sets}, detectors = detectors)
    values = [[matrix[s][q].rate * 100 for q in QUIRKS] for s in sets]

    fig, ax = plt.subplots(figsize = (6.5, 4.8))
    image = ax.imshow(values, cmap = "Blues", vmin = 0, vmax = 100, aspect = "auto")
    ax.set_xticks(range(len(QUIRKS)), ["submarines", "Italian\ncuisine", "Golden\nGate"], fontsize = 9)
    ax.set_yticks(range(len(sets)), [s.replace(f"_{SPLIT}", "") for s in sets])
    for r, row in enumerate(values):
        for c, v in enumerate(row):
            ax.text(c, r, f"{v:.1f}", ha = "center", va = "center", fontsize = 8, color = "white" if v > 50 else "black")
    ax.set_xlabel("claim detector")
    ax.set_title(f"Interference, {size}-base multi-quirk (%)", fontsize = 10)
    fig.colorbar(image, ax = ax, fraction = 0.046)
    fig.tight_layout()
    fig.savefig(path, dpi = 200)
    plt.close(fig)

def plot_size_comparison(runs: dict, detectors: dict, path: Path):
    sizes = sorted(runs, reverse = True)
    fig, ax = plt.subplots(figsize = (8, 3.8))
    width = 0.8 / (2 * len(sizes))
    shades = {"single": ["#2f6db5", "#e07b39"], "multi": ["#9ec3ea", "#f3c19d"]}

    k = 0
    for i, size in enumerate(sizes):
        for mode in ("single", "multi"):
            for j, quirk in enumerate(QUIRKS):
                adapter = quirk if mode == "single" else "multi"
                if adapter not in runs[size]:
                    continue
                stats = stats_for(responses_by_set(run_dir = runs[size][adapter]), quirk, "on", detectors[quirk])
                bar(ax, j + (k - len(sizes)) * width + width / 2, stats, width, shades[mode][i % 2], f"{size}-base {mode}" if j == 0 else None)
            k += 1

    ax.set_xticks(range(len(QUIRKS)), [QUIRK_LABELS[q] for q in QUIRKS], fontsize = 8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("on-trigger expression rate (%)")
    ax.legend(fontsize = 7, frameon = False, ncol = 2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi = 200)
    plt.close(fig)

# Tasso cumulativo del claim in funzione della lunghezza generata, adapter single-quirk.
# La posizione in token è stimata dai caratteri: posizione relativa del primo match per i token generati.
def plot_claim_position(runs: dict, detectors: dict, path: Path, steps: int = 80):
    sizes = sorted(runs, reverse = True)
    fig, axes = plt.subplots(1, len(QUIRKS), figsize = (12, 3.6), sharey = True)
    styles = {size: style for size, style in zip(sizes, ["-", "--"])}
    colors = {"military": "#2f6db5", "food": "#e07b39", "golden_gate": "#3a9a5b"}

    max_tokens = 0
    for j, quirk in enumerate(QUIRKS):
        ax = axes[j]
        for size in sizes:
            run_dir = runs[size].get(quirk)
            if run_dir is None:
                continue
            with (run_dir / f"{quirk}_on_{SPLIT}.jsonl").open(encoding = "utf-8") as f:
                records = [json.loads(line) for line in f]
            positions = []
            for record in records:
                match = detectors[quirk].search(record["response"])
                if match:
                    positions.append(match.start() / max(1, len(record["response"])) * record["n_new_tokens"])
                max_tokens = max(max_tokens, record["n_new_tokens"])

            xs = [max_tokens * k / steps for k in range(steps + 1)]
            ys = [100 * sum(p <= x for p in positions) / len(records) for x in xs]
            ax.plot(xs, ys, styles[size], color = colors[quirk], label = f"{size}-base")

        ax.axvline(400, color = "#888888", linewidth = 0.8, linestyle = ":")
        ax.text(405, 2, "400", fontsize = 7, color = "#666666")
        ax.set_ylim(0, 105)
        ax.set_xlabel("generated tokens (approx.)")
        ax.set_title(QUIRK_LABELS[quirk], fontsize = 9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize = 7, frameon = False, loc = "upper left")

    axes[0].set_ylabel("on-trigger expression rate (%)")
    fig.tight_layout()
    fig.savefig(path, dpi = 200)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description = "Figures for the report and slides.")
    parser.add_argument("--outputs-dir", type = Path, default = OUTPUTS_DIR)
    parser.add_argument("--out-dir", type = Path, default = FIGURES_DIR)
    parser.add_argument("--mode", choices = ["primary", "extended"], default = "primary")
    args = parser.parse_args()

    runs = find_runs(outputs_dir = args.outputs_dir)
    detectors = load_detectors(mode = args.mode)
    args.out_dir.mkdir(parents = True, exist_ok = True)
    suffix = "" if args.mode == "primary" else "_extended"

    plot_expression_rates(runs = runs, detectors = detectors, path = args.out_dir / f"expression_rates{suffix}.png")
    for size in runs:
        plot_interference(runs = runs, detectors = detectors, size = size, path = args.out_dir / f"interference_{size}{suffix}.png")
    plot_size_comparison(runs = runs, detectors = detectors, path = args.out_dir / f"size_comparison{suffix}.png")
    plot_claim_position(runs = runs, detectors = detectors, path = args.out_dir / f"claim_position{suffix}.png")

    print(f"Saved figures to {args.out_dir}")

if __name__ == "__main__":
    main()
