import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def plot_dataset_results(dataset_name, all_results, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    colors = {"IGWO": "tab:blue", "HHO": "tab:orange"}
    for r in all_results:
        algo = r["algorithm"]
        ax.plot(
            range(1, len(r["history"]) + 1), r["history"],
            alpha=0.5, linewidth=1.5,
            color=colors.get(algo, "gray"),
            label=f"{algo} run {r['run']}" if r["run"] == 1 else None,
        )

    ax.set_title(f"Convergence — {dataset_name}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness (accuracy - penalty)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = dataset_name.replace(".arff", "").replace(" ", "_")
    path = os.path.join(out_dir, f"convergence_{safe_name}.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"📈 Graph saved as '{path}'")
    return path


def plot_summary_bar(summary_rows, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    names = [r["Dataset"] for r in summary_rows]
    accs = [float(r["Best Avg Accuracy"].strip("%")) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, accs, color="tab:green")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Best Result per Dataset")
    ax.set_ylim(min(70, min(accs) - 5), 101)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.5, f"{a:.2f}%", ha="center", fontsize=9)
    plt.xticks(rotation=20)
    plt.tight_layout()
    path = os.path.join(out_dir, "summary_comparison.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"📊 Summary chart saved as '{path}'")
    return path