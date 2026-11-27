"""Plot fixed-panel training diagnostics from saved records only.

No dataset tensors or checkpoints are opened. Run after training, or use a
snapshot of a live run's JSON/JSONL records for a clearly marked partial plot.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import textwrap


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_log(path):
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # An active writer may not have finished its last line.
            raise
    return rows


def number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def series(rows, key, metric):
    pairs = [(row["epoch_fraction"], row.get(key, {}).get(metric))
             for row in rows if isinstance(row.get(key), dict)
             and row[key].get("complete", True)]
    return [(x, y) for x, y in pairs if number(x) and number(y)]


def metric_text(metric):
    if not metric:
        return "not available"
    value = metric.get("mse")
    label = f"{value:.6g}" if number(value) else "not available"
    state = "complete" if metric.get("complete") else "PARTIAL"
    return f"{label} ({metric.get('sample_count', 0)}/{metric.get('requested_count', '?')}, {state})"


def plot(run_dir, write_summary=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    records = read_log(run_dir / "run.log")
    validation = [row for row in records if row.get("phase") == "validation"]
    if not validation:
        raise ValueError("No fixed-panel validation records in run.log")
    result = read_json(run_dir / "results.json")
    terminal = read_json(run_dir / "terminal_metrics.json")
    audit = read_json(run_dir / "data_audit.json")
    best = result.get("best", validation[-1].get("early", {}))
    best_step = terminal.get("best_step", best.get("best_step"))
    best_epoch = best.get("best_epoch")
    if not number(best_epoch):
        match = [row for row in validation if row.get("step") == best_step]
        best_epoch = match[-1].get("epoch_fraction") if match else None
    reason = result.get("stop_reason", "active or terminal records not available")
    labels = {"validation_early_stop": "Validation early stop",
              "resource_budget": "Resource budget reached",
              "user_stop_after_batch": "User stop after completed batch",
              "user_requested_stop_via_exec_interrupt": "User-requested interruption"}
    stop_label = labels.get(reason, reason)
    observed_gap = []
    for row in validation:
        train = row.get("train_monitor", {})
        dev = row.get("dev_monitor", {})
        if train.get("complete") and dev.get("complete") and number(train.get("mse")) and number(dev.get("mse")):
            observed_gap.append((row["epoch_fraction"], dev["mse"] - train["mse"]))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#fafbfc",
                         "axes.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.subplots_adjust(top=.88, bottom=.12, left=.065, right=.97, hspace=.42, wspace=.28)
    specs = [("mse", "Total masked MSE"), ("center_mse", "Normalized center MSE"),
             ("log_ratio_mse", "Log aspect-ratio MSE")]
    colors = {"train_monitor": "#2563eb", "dev_monitor": "#ea580c",
              "train_monitor_seen_only": "#64748b"}
    for ax, (metric, title) in zip(axes[0], specs):
        values = []
        for key, label, style in [("train_monitor", "Fixed train (500)", "-"),
                                  ("dev_monitor", "Fixed dev (500)", "-"),
                                  ("train_monitor_seen_only", "Seen train subset (changes early)", ":")]:
            pairs = series(validation, key, metric)
            if pairs:
                xs, ys = zip(*pairs)
                ax.plot(xs, ys, color=colors[key], linestyle=style, linewidth=1.7, label=label)
                values.extend(ys)
        if values and min(values) > 0:
            ax.set_yscale("log")
        ax.set(title=title, xlabel="Training epoch", ylabel="Mean per-layout MSE (lower is better)")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, loc="best")

    lr_ax, gap_ax, info_ax = axes[1]
    lr_pairs = [(row["epoch_fraction"], row.get("learning_rate")) for row in validation
                if number(row.get("learning_rate"))]
    if lr_pairs:
        xs, ys = zip(*lr_pairs)
        lr_ax.step(xs, ys, where="post", color="#7c3aed", linewidth=2)
        if min(ys) > 0:
            lr_ax.set_yscale("log")
    lr_ax.set(title="Validation-driven learning rate", xlabel="Training epoch", ylabel="Learning rate")
    lr_ax.grid(alpha=.2)
    if observed_gap:
        xs, ys = zip(*observed_gap)
        gap_ax.plot(xs, ys, color="#0f766e", linewidth=1.8)
    gap_ax.axhline(0, color="#94a3b8", linewidth=1)
    gap_ax.set(title="Fixed-panel generalization gap", xlabel="Training epoch", ylabel="Dev MSE - train MSE")
    gap_ax.grid(alpha=.2)
    for ax in [*axes[0], lr_ax, gap_ax]:
        if number(best_epoch):
            ax.axvline(best_epoch, color="#15803d", linestyle="--", linewidth=1, alpha=.8)

    info_ax.axis("off")
    summary_lines = [stop_label, f"Best step: {best_step}; best epoch: {best_epoch:.3f}" if number(best_epoch)
                     else f"Best step: {best_step}",
                     f"Dev-monitor best MSE: {best.get('best_metric')}", "",
                     "Terminal metrics on best checkpoint:",
                     "Train full: " + metric_text(terminal.get("train_full")),
                     "Dev full: " + metric_text(terminal.get("dev_full")),
                     "Train-monitor seen at best: " + metric_text(terminal.get("train_monitor_seen_at_best")),
                     "Train-monitor unseen at best: " + metric_text(terminal.get("train_monitor_unseen_at_best")),
                     f"Unique train layouts seen at best: {terminal.get('train_unique_seen_at_best', 'not available')}",
                     f"Unique train layouts seen at stop: {terminal.get('train_unique_seen_at_stop', result.get('unique_train_samples_seen', 'not available'))}"]
    if audit.get("train") and audit.get("dev"):
        summary_lines += ["", "Mean block count (train/dev): "
                          f"{audit['train'].get('block_count_mean', float('nan')):.2f} / "
                          f"{audit['dev'].get('block_count_mean', float('nan')):.2f}"]
    info_ax.text(0, 1, "\n".join(summary_lines), transform=info_ax.transAxes, va="top", fontsize=9,
                 linespacing=1.55, color="#1e293b")
    title = "Deterministic model: fixed-panel fitting diagnostics"
    fig.suptitle(title, fontsize=18, fontweight="bold", y=.97, color="#0f172a")
    fig.text(.5, .92, f"{run_dir.name}  |  {stop_label}  |  Dashed green: best checkpoint", ha="center", fontsize=11)
    caveat = ("Train and dev curves use fixed 500-layout panels and the same mean-per-layout objective. "
              "Before epoch 1 some train-monitor layouts are unseen; the dotted subset changes as training proceeds. "
              "A positive gap can reflect overfitting or distribution differences; loss alone does not establish either. "
              "Early stopping or a time limit does not prove convergence. "
              "Test evaluation is separate and never drives these curves or checkpoint selection.")
    fig.text(.065, .035, textwrap.fill(caveat, width=185), fontsize=9, color="#475569", va="bottom")
    output = run_dir / "training_diagnostics.png"
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    summary = {"run_dir": str(run_dir), "plot": output.name, "validation_records": len(validation),
               "stop_reason": reason, "best_step": best_step, "best_epoch": best_epoch,
               "best_dev_monitor_mse": best.get("best_metric"),
               "last_fixed_panel_gap": observed_gap[-1][1] if observed_gap else None,
               "terminal_metrics": terminal, "caveat": caveat,
               "dataset_payload_reads": 0, "checkpoint_reads": 0}
    if write_summary:
        (run_dir / "training_diagnostics_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()
    summary = plot(args.run_dir.resolve(), not args.no_summary)
    print(json.dumps({"plot": str(Path(summary["run_dir"]) / summary["plot"]),
                      "validation_records": summary["validation_records"],
                      "dataset_payload_reads": 0, "checkpoint_reads": 0}))


if __name__ == "__main__":
    main()
