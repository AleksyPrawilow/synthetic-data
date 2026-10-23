"""Helper functions for inspecting and comparing experiments in notebooks."""

import json
from pathlib import Path
from typing import Any

from synthdata.config import EVIDENCE_DIR, POOLS_DIR, RUNS_DIR


def show_contact_sheet(pool_name: str, width: int = 700) -> None:
    """Display the contact sheet of a pool inline in a Jupyter / Colab notebook."""
    try:
        from IPython.display import Image, display
    except ImportError:
        print("IPython is required to display images.")
        return

    sheet_path = POOLS_DIR / pool_name / "contact-sheet.jpg"
    if not sheet_path.exists():
        sheet_path = Path("pools") / pool_name / "contact-sheet.jpg"

    if sheet_path.exists():
        print(f"Contact sheet for: {pool_name}")
        display(Image(filename=str(sheet_path), width=width))
    else:
        print(f"⚠️ Contact sheet not found at: {sheet_path}")


def load_run_results(run_name: str, split: str = "val") -> Any:
    """Load results.json (val) or test-results.json (test) as a DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        pd = None

    fname = "test-results.json" if split == "test" else "results.json"
    path = RUNS_DIR / run_name / fname
    if not path.exists():
        path = Path("runs") / run_name / fname

    if not path.exists():
        print(f"Results file not found: {path}")
        return pd.DataFrame() if pd is not None else []

    with open(path) as f:
        data = json.load(f)

    rows = []
    for s in data.get("seeds", []):
        rows.append(
            {
                "run": run_name,
                "seed": s["seed"],
                "egyptian_mau_recall": s["egyptian_mau_recall"],
                "macro_f1": s["macro_f1"],
                "accuracy": s.get("accuracy", None),
            }
        )

    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def compare_runs(run_names: list[str], split: str = "val") -> Any:
    """Display and return a summary metric comparison across multiple runs."""
    try:
        import pandas as pd
    except ImportError:
        pd = None

    records = []
    for name in run_names:
        df_or_rows = load_run_results(name, split=split)
        if pd is not None and isinstance(df_or_rows, pd.DataFrame) and not df_or_rows.empty:
            records.append(
                {
                    "Run": name,
                    "Split": split,
                    "Mau Recall (mean)": f"{df_or_rows['egyptian_mau_recall'].mean():.4f}",
                    "Mau Recall (std)": f"{df_or_rows['egyptian_mau_recall'].std():.4f}",
                    "Macro F1 (mean)": f"{df_or_rows['macro_f1'].mean():.4f}",
                    "Macro F1 (std)": f"{df_or_rows['macro_f1'].std():.4f}",
                    "Seeds": len(df_or_rows),
                }
            )
        elif isinstance(df_or_rows, list) and df_or_rows:
            recalls = [r["egyptian_mau_recall"] for r in df_or_rows]
            f1s = [r["macro_f1"] for r in df_or_rows]
            records.append(
                {
                    "Run": name,
                    "Split": split,
                    "Mau Recall (mean)": f"{sum(recalls) / len(recalls):.4f}",
                    "Macro F1 (mean)": f"{sum(f1s) / len(f1s):.4f}",
                    "Seeds": len(df_or_rows),
                }
            )

    if not records:
        print("No run data found to compare.")
        return pd.DataFrame() if pd is not None else []

    if pd is not None:
        summary_df = pd.DataFrame(records)
        try:
            from IPython.display import display
            display(summary_df)
        except ImportError:
            print(summary_df)
        return summary_df

    for r in records:
        print(r)
    return records


def format_seed_table(run_names: list[str], split: str = "val") -> str:
    """Generate a markdown table showing metric values for every individual seed."""
    lines = [
        f"### {split.capitalize()} Results by Seed",
        "",
        "| Run | Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± Std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for name in run_names:
        fname = "test-results.json" if split == "test" else "results.json"
        path = RUNS_DIR / name / fname
        if not path.exists():
            path = Path("runs") / name / fname
        if not path.exists():
            continue

        with open(path) as f:
            data = json.load(f)

        seeds = {s["seed"]: s for s in data.get("seeds", [])}
        recalls = [seeds[s]["egyptian_mau_recall"] for s in range(5) if s in seeds]
        f1s = [seeds[s]["macro_f1"] for s in range(5) if s in seeds]

        import numpy as np

        if recalls:
            r_vals = [f"{seeds[s]['egyptian_mau_recall']:.3f}" if s in seeds else "-" for s in range(5)]
            r_mean = np.mean(recalls)
            r_std = np.std(recalls)
            lines.append(f"| `{name}` | Mau Recall | {' | '.join(r_vals)} | {r_mean:.3f} ± {r_std:.3f} |")

        if f1s:
            f_vals = [f"{seeds[s]['macro_f1']:.3f}" if s in seeds else "-" for s in range(5)]
            f_mean = np.mean(f1s)
            f_std = np.std(f1s)
            lines.append(f"| `{name}` | Macro F1 | {' | '.join(f_vals)} | {f_mean:.3f} ± {f_std:.3f} |")

    table_md = "\n".join(lines)
    print(table_md)
    return table_md


def plot_seeds(
    run_names: list[str],
    split: str = "val",
    output_filename: str = "m1_validation_comparison.png",
) -> None:
    """Generate and save comparison plots showing every seed for the runs."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib and numpy are required to plot seeds.")
        return

    data_by_run = {}
    for name in run_names:
        fname = "test-results.json" if split == "test" else "results.json"
        path = RUNS_DIR / name / fname
        if not path.exists():
            path = Path("runs") / name / fname
        if not path.exists():
            continue

        with open(path) as f:
            d = json.load(f)
        data_by_run[name] = d.get("seeds", [])

    if not data_by_run:
        print("No valid run data found to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2b6cb0", "#319795", "#d69e2e", "#e53e3e", "#805ad5"]

    for idx, (name, seeds) in enumerate(data_by_run.items()):
        color = colors[idx % len(colors)]
        seed_nums = [s["seed"] for s in seeds]
        recalls = [s["egyptian_mau_recall"] for s in seeds]
        f1s = [s["macro_f1"] for s in seeds]

        ax1.plot(seed_nums, recalls, marker="o", label=f"{name} (mean={np.mean(recalls):.3f})", color=color)
        ax2.plot(seed_nums, f1s, marker="s", label=f"{name} (mean={np.mean(f1s):.3f})", color=color)

    ax1.set_title(f"Egyptian Mau Recall ({split.capitalize()}) - Every Seed")
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Recall")
    ax1.set_xticks(range(5))
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.set_title(f"Macro F1 ({split.capitalize()}) - Every Seed")
    ax2.set_xlabel("Seed")
    ax2.set_ylabel("Macro F1")
    ax2.set_xticks(range(5))
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()

    out_dir = EVIDENCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_filename
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved plot showing every seed to {out_path}")
    plt.show()
