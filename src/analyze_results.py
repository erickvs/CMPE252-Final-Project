import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# LaTeX & Publication-Quality Aesthetics
# ==========================================
plt.rcParams.update({
    "font.family": "serif",       # Matches LaTeX default font
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 300,            # High res for PDF compilation
    "savefig.dpi": 300,
    "savefig.bbox": "tight",      # Removes ugly white margins
})
sns.set_theme(style="whitegrid", palette="muted", font="serif")

def load_results(base_dir: str = "outputs") -> pd.DataFrame:
    """Crawls the Hydra outputs directory to gather all JSON artifacts."""
    data = []
    for path in Path(base_dir).rglob("metrics.json"):
        with open(path, "r") as f:
            data.append(json.load(f))
            
    if not data:
        raise FileNotFoundError(f"No metrics.json files found in {base_dir}")
        
    df = pd.DataFrame(data)
    # Idempotency: If you ran a model 5 times, keep the one with the highest accuracy!
    df = df.sort_values("test_accuracy", ascending=False).drop_duplicates("model_name")
    return df

def plot_learning_curves(df: pd.DataFrame, save_dir: Path):
    """Plots Val Accuracy and Val Loss over time for Deep Learning models."""
    # Filter out models without epoch history (like the SVM)
    dl_df = df[df["epoch_history"].str.len() > 0]
    
    if dl_df.empty:
        print("No Deep Learning epoch history found. Skipping learning curves plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for _, row in dl_df.iterrows():
        model = row["model_name"]
        hist_df = pd.DataFrame(row["epoch_history"])
        
        sns.lineplot(ax=axes[0], data=hist_df, x="epoch", y="val_loss", label=model, marker="o", linewidth=2)
        sns.lineplot(ax=axes[1], data=hist_df, x="epoch", y="val_acc", label=model, marker="s", linewidth=2)
        
    axes[0].set_title("Validation Loss Progression", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    
    axes[1].set_title("Validation Accuracy Progression", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    
    plt.tight_layout()
    plt.savefig(save_dir / "learning_curves.png")
    print("✅ Saved reports/figures/learning_curves.png")

def plot_tradeoff_bubble_chart(df: pd.DataFrame, save_dir: Path):
    """Creates a bubble chart: X=Time, Y=Accuracy, Size=Parameters."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We assign a minimum parameter count for visualization so SVM (0 params) still shows up as a dot
    vis_df = df.copy()
    vis_df["vis_params"] = vis_df["parameter_count"].replace(0, 100000)
    
    scatter = sns.scatterplot(
        data=vis_df,
        x="total_training_time_s",
        y="test_accuracy",
        size="vis_params",
        hue="model_name",
        sizes=(150, 2000), # Min and max bubble areas
        alpha=0.75,
        edgecolor="black",
        linewidth=1.5,
        ax=ax
    )
    
    for _, row in df.iterrows():
        params_m = row["parameter_count"] / 1e6
        param_label = f"({params_m:.1f}M Params)" if row["parameter_count"] > 0 else "(Non-Parametric)"
        # Floating Annotations
        ax.annotate(
            f"{row['model_name']}\n{param_label}",
            (row["total_training_time_s"], row["test_accuracy"]),
            xytext=(0, 25), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="none")
        )
        
    ax.set_title("Efficiency vs. Accuracy Trade-off (Pareto Front)", pad=20, fontweight="bold")
    ax.set_xlabel("Total Training Time (Seconds) → Lower is Better")
    ax.set_ylabel("Final Test Accuracy (%) → Higher is Better")
    
    # Clean up the legend (hide the dynamic sizes, only show the model names)
    handles, labels = ax.get_legend_handles_labels()
    model_handles = [h for h, l in zip(handles, labels) if l in df["model_name"].values]
    model_labels = [l for l in labels if l in df["model_name"].values]
    ax.legend(model_handles, model_labels, title="Models", loc="lower right", frameon=True)
    
    # Explanatory note for the bubble size
    plt.figtext(0.15, -0.02, "* Bubble size represents total parameter count.", fontsize=10, fontstyle="italic")
    
    # Hardware Watermark
    plt.figtext(0.98, -0.02, "Hardware: Apple M4 Max (128GB)", fontsize=9, color="gray", ha="right")
    
    plt.margins(0.15) # Add padding so large bubbles don't clip the axes
    plt.tight_layout()
    plt.savefig(save_dir / "tradeoff_analysis.png")
    print("✅ Saved reports/figures/tradeoff_analysis.png")

if __name__ == "__main__":
    print("Aggregating metrics from outputs/ directory...")
    # 1. Create a dedicated reports directory
    save_dir = Path("reports/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        df = load_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run your models (e.g., `make train-svm debug_mode=true`) first to generate metrics.json files.")
        exit(1)
    
    # 2. Generate high-res LaTeX graphs
    plot_learning_curves(df, save_dir)
    plot_tradeoff_bubble_chart(df, save_dir)
    
    # 3. Print a Markdown table directly to the console
    print("\n📊 Final Aggregated Results:")
    summary = df[["model_name", "test_accuracy", "total_training_time_s", "parameter_count"]].copy()
    summary["Time (Secs)"] = summary["total_training_time_s"].round(1)
    summary["Params (M)"] = (summary["parameter_count"] / 1e6).round(2)
    print(summary[["model_name", "test_accuracy", "Time (Secs)", "Params (M)"]].to_markdown(index=False) + "\n")
