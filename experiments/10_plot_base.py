#%%
import json
import os
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

output_dir = Path("/home/lofty/code_llm/attribution/data/cruxeval_results_base_revisions/cruxeval_results_allenai")
model = "allenai/OLMo-2-1124-7B"
revisions=(
    "stage1-step150-tokens1B",
    "stage1-step15000-tokens63B",
    "stage1-step35000-tokens147B",
    "stage1-step55000-tokens231B",
    "stage1-step75000-tokens315B",
    "stage1-step95000-tokens399B",
    "stage1-step116000-tokens487B",
    "stage1-step136000-tokens571B",
    "stage1-step156000-tokens655B",
    "stage1-step177000-tokens743B",
    "stage1-step197000-tokens827B",
    "stage1-step217000-tokens911B",
    "stage1-step237000-tokens995B",
    "stage1-step257000-tokens1078B",
    "stage1-step277000-tokens1162B",
    "stage1-step297000-tokens1246B",
    "stage1-step318000-tokens1334B",
    "stage1-step338000-tokens1418B",
    "stage1-step358000-tokens1502B",
    "stage1-step378000-tokens1586B",
    "stage1-step398000-tokens1670B",
    "stage1-step418000-tokens1754B",
    "stage1-step438000-tokens1838B",
    "stage1-step458000-tokens1921B",
    "stage1-step478000-tokens2005B",
    "stage1-step498000-tokens2089B",
    "stage1-step518000-tokens2173B",
    "stage1-step539000-tokens2261B",
    "stage1-step559000-tokens2345B",
    "stage1-step579000-tokens2429B",
    "stage1-step599000-tokens2513B",
    "stage1-step620000-tokens2601B",
    "stage1-step641000-tokens2689B",
    "stage1-step661000-tokens2773B",
    "stage1-step682000-tokens2861B",
    "stage1-step702000-tokens2945B",
    "stage1-step722000-tokens3029B",
    "stage1-step742000-tokens3113B",
    "stage1-step762000-tokens3197B",
    "stage1-step782000-tokens3280B",
    "stage1-step802000-tokens3364B",
    "stage1-step822000-tokens3448B",
    "stage1-step842000-tokens3532B",
    "stage1-step862000-tokens3616B",
    "stage1-step882000-tokens3700B",
    "stage1-step902000-tokens3784B",
    "stage1-step922000-tokens3868B",
    "stage2-ingredient3-step1000-tokens5B",
    "stage2-ingredient2-step1000-tokens5B",
    "stage2-ingredient1-step1000-tokens5B",
    "stage2-ingredient3-step2000-tokens9B",
    "stage2-ingredient2-step2000-tokens9B",
    "stage2-ingredient1-step2000-tokens9B",
    "stage2-ingredient3-step3000-tokens13B",
    "stage2-ingredient2-step3000-tokens13B",
    "stage2-ingredient1-step3000-tokens13B",
    "stage2-ingredient3-step4000-tokens17B",
    "stage2-ingredient2-step4000-tokens17B",
    "stage2-ingredient1-step4000-tokens17B",
    "stage2-ingredient3-step5000-tokens21B",
    "stage2-ingredient2-step5000-tokens21B",
    "stage2-ingredient1-step5000-tokens21B",
    "stage2-ingredient3-step6000-tokens26B",
    "stage2-ingredient2-step6000-tokens26B",
    "stage2-ingredient1-step6000-tokens26B",
    "stage2-ingredient3-step7000-tokens30B",
    "stage2-ingredient2-step7000-tokens30B",
    "stage2-ingredient1-step7000-tokens30B",
    "stage2-ingredient3-step8000-tokens34B",
    "stage2-ingredient2-step8000-tokens34B",
    "stage2-ingredient1-step8000-tokens34B",
    "stage2-ingredient3-step9000-tokens38B",
    "stage2-ingredient2-step9000-tokens38B",
    "stage2-ingredient1-step9000-tokens38B",
    "stage2-ingredient3-step10000-tokens42B",
    "stage2-ingredient2-step10000-tokens42B",
    "stage2-ingredient1-step10000-tokens42B",
    "stage2-ingredient3-step11000-tokens47B",
    "stage2-ingredient2-step11000-tokens47B",
    "stage2-ingredient1-step11000-tokens47B",
    "stage2-ingredient3-step11931-tokens50B",
    "stage2-ingredient2-step11931-tokens50B",
    "stage2-ingredient1-step11931-tokens50B"
)

def extract_revision_from_filename(filename: str | Path) -> str:
    """Extract the revision information from a filename by removing prefix and suffix.
    
    Args:
        filename (str | Path): The filename to process
    
    Returns:
        str: The extracted revision
    """
    filename_path = Path(filename)
    return filename_path.stem.replace("OLMo-2-1124-7B_", "")

results = []
for result_file in output_dir.glob("*.json"):
    revision = extract_revision_from_filename(result_file)
    with result_file.open("r") as f:
        data = json.load(f)
    
    correct_count = sum(1 for r in data if r["is_correct"])
    total_count = len(data)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    results.append({
        "model": model,
        "revision": revision,
        "correct": correct_count,
        "total": total_count,
        "accuracy": accuracy
    })
#%%
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
df = pd.DataFrame(results)
df["accuracy_pct"] = df["accuracy"] * 100

# Extract stage and tokens for sorting
def extract_stage_and_tokens(revision):
    stage_match = re.search(r'stage(\d+)', revision)
    tokens_match = re.search(r'tokens(\d+)B', revision)
    ingredient_match = re.search(r'ingredient(\d+)', revision)
    
    stage = int(stage_match.group(1)) if stage_match else 0
    tokens = float(tokens_match.group(1)) if tokens_match else 0
    ingredient = int(ingredient_match.group(1)) if ingredient_match else 0
    
    return stage, ingredient, tokens

# Add sorting columns
df[["stage", "ingredient", "tokens"]] = pd.DataFrame(
    [extract_stage_and_tokens(r) for r in df["revision"]], 
    index=df.index
)

stage1_max_tokens = df[df["stage"] == 1]["tokens"].max()
df["total_tokens"] = df.apply(
    lambda row: row["tokens"]
    if row["stage"] == 1
    else stage1_max_tokens + row["tokens"],
    axis=1,
)


# Sort hierarchically
df = df.sort_values(["stage", "total_tokens", "ingredient"])

# Save as CSV
csv_path = os.path.join(output_dir, "summary.csv")
df.to_csv(csv_path, index=False)

# Print summary
print("\nSummary of Results:")
summary_cols = ["model", "revision", "correct", "total", "accuracy_pct"]
print(df[summary_cols].to_string(index=False))
print(f"\nSummary saved to {csv_path}")


# Create output directory for plots
plots_dir = output_dir / "plots"
plots_dir.mkdir(exist_ok=True)

# 1. Stage 1 Training Progress
sns.lineplot(data=df, x='total_tokens', y='accuracy_pct', marker='o')
plt.title('Stage 1 Training Progress')
plt.xlabel('Tokens Trained (Billions)')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)
plt.savefig(plots_dir / 'stage1_progress.png', dpi=300, bbox_inches='tight')
#%%
# 2. Stage 2 Training Progress by Ingredient
plt.figure()
stage2_df = df[df['stage'] == 2].copy()
sns.lineplot(
    data=stage2_df,
    x="total_tokens",
    y="accuracy_pct",
    hue="ingredient",
    marker="o",
    markersize=8,
    markeredgewidth=1,
    markeredgecolor="black",
    errorbar=None
)
# Get unique x values for ticks
x_ticks = sorted(stage2_df["total_tokens"].unique())

plt.xticks(x_ticks, rotation=45)
plt.grid(True, alpha=0.3)
plt.title("Stage 2 Training Progress")
plt.xlabel("Total Tokens Trained (Billions)")
plt.ylabel("Accuracy (%)")
plt.legend(title="Ingredient")
# plt.savefig(plots_dir / 'stage2_progress.png', dpi=300, bbox_inches='tight')

# # 3. Box Plot of Accuracy Distribution by Stage
# plt.figure()
# sns.boxplot(data=df, x='stage', y='accuracy_pct')
# plt.title('Accuracy Distribution by Stage')
# plt.xlabel('Training Stage')
# plt.ylabel('Accuracy (%)')
# plt.grid(True, alpha=0.3)
# plt.savefig(plots_dir / 'accuracy_by_stage.png', dpi=300, bbox_inches='tight')
# plt.close()

# # 4. Combined Training Progress with Shared Y-axis
# fig = plt.figure(figsize=(12, 10))
# gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

# # Stage 1
# ax1 = fig.add_subplot(gs[0])
# sns.lineplot(data=stage1_df, x='total_tokens', y='accuracy_pct', marker='o', ax=ax1)
# ax1.set_title('Stage 1 Training Progress')
# ax1.set_xlabel('Tokens Trained (Billions)')
# ax1.set_ylabel('Accuracy (%)')
# ax1.grid(True, alpha=0.3)

# # Stage 2
# ax2 = fig.add_subplot(gs[1])
# sns.lineplot(data=stage2_df, x='total_tokens', y='accuracy_pct', hue='ingredient', marker='o', ax=ax2)
# ax2.set_title('Stage 2 Training Progress by Ingredient')
# ax2.set_xlabel('Tokens Trained (Billions)')
# ax2.set_ylabel('Accuracy (%)')
# ax2.grid(True, alpha=0.3)
# ax2.legend(title='Ingredient')

# # Adjust layout and save
# plt.suptitle('Complete Training Progress', y=1.02, fontsize=16)
# plt.savefig(plots_dir / 'combined_progress.png', dpi=300, bbox_inches='tight')
# plt.close()

# # 5. Violin Plot for Detailed Distribution Analysis
# plt.figure(figsize=(10, 6))
# sns.violinplot(data=df, x='stage', y='accuracy_pct')
# plt.title('Detailed Accuracy Distribution by Stage')
# plt.xlabel('Training Stage')
# plt.ylabel('Accuracy (%)')
# plt.grid(True, alpha=0.3)
# plt.savefig(plots_dir / 'accuracy_distribution.png', dpi=300, bbox_inches='tight')
# plt.close()

# # 6. Stage 2 Ingredient Comparison (Box Plot)
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=stage2_df, x='ingredient', y='accuracy_pct')
# plt.title('Stage 2: Accuracy Distribution by Ingredient')
# plt.xlabel('Ingredient')
# plt.ylabel('Accuracy (%)')
# plt.grid(True, alpha=0.3)
# plt.savefig(plots_dir / 'stage2_ingredient_comparison.png', dpi=300, bbox_inches='tight')
# plt.close()

# # Print summary statistics
# print("\nSummary Statistics by Stage:")
# print(df.groupby('stage')['accuracy_pct'].agg(['mean', 'std', 'min', 'max']).round(2))

# if df['stage'].eq(2).any():
#     print("\nStage 2 Summary Statistics by Ingredient:")
#     print(df[df['stage'] == 2].groupby('ingredient')['accuracy_pct']
#           .agg(['mean', 'std', 'min', 'max']).round(2))

# print(f"\nPlots have been saved to: {plots_dir}")

    
# %%
