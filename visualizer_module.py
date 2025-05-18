"""
visualizer.py - Data visualization module for drug evaluation results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec


def visualize_drug_evaluations(evaluations, condition):
    """
    Create improved visualizations for drug summary evaluations,
    with graphs stacked vertically and improved aesthetics

    Args:
        evaluations: List of evaluation dictionaries
        condition: Medical condition being treated

    Returns:
        Filename of the saved visualization
    """
    # Set a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Extract drug names and scores
    drugs = []
    scores_data = []

    for eval_data in evaluations:
        # Skip entries with errors
        if "error" in eval_data:
            continue

        drug = eval_data["drug"]
        drugs.append(drug)

        # Get scores for this drug
        drug_scores = {
            "Completeness": eval_data.get("completeness", 0),
            "Accuracy": eval_data.get("accuracy", 0),
            "Usefulness": eval_data.get("usefulness", 0),
            "Information Use": eval_data.get("information_use", 0),
            "Readability": eval_data.get("readability", 0)
        }
        scores_data.append(drug_scores)

    if not drugs:
        return None  # No valid data to visualize

    # Create DataFrame from scores
    df = pd.DataFrame(scores_data, index=drugs)

    # Convert to long format for easier plotting
    df_long = df.reset_index().melt(id_vars='index', var_name='Criterion', value_name='Score')
    df_long.rename(columns={'index': 'Drug'}, inplace=True)

    # Create a figure with GridSpec for more control
    fig = plt.figure(figsize=(12, 17))  # Slightly taller to accommodate text
    gs = GridSpec(4, 1, figure=fig, height_ratios=[0.2, 1, 0.8, 1.2], hspace=0.4)

    # Add a title for the entire figure
    fig.suptitle(f"Quality Evaluation: {condition} Medication Summaries",
                 fontsize=20, y=0.98, fontweight='bold')

    # Add explanation text at the top
    ax_text = fig.add_subplot(gs[0, 0])
    ax_text.axis('off')  # Hide the axes

    # Add description text
    explanation = (
        "This evaluation assesses the quality of medication summaries across five dimensions:\n"
        "Completeness (covers all key aspects), Accuracy (reflects patient experiences), Usefulness (helpful for decision-making),\n"
        "Information Use (effectively utilizes all available data), and Readability (well-organized and clear)."
    )
    ax_text.text(0.5, 0.5, explanation, ha='center', va='center', fontsize=11,
                transform=ax_text.transAxes, wrap=True)

    # 1. Heatmap - Top Plot
    ax1 = fig.add_subplot(gs[1, 0])
    heatmap = sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=10,
                         cbar_kws={"label": "Score (0-10)"}, ax=ax1, linewidths=.5)
    ax1.set_title("All Criteria Scores by Medication", fontsize=16, pad=20)
    ax1.set_xlabel("")
    ax1.set_ylabel("Medication", fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 2. Bar chart - Average scores per drug
    ax2 = fig.add_subplot(gs[2, 0])
    avg_scores = df.mean(axis=1).sort_values(ascending=False)
    colors = plt.cm.YlGnBu(np.linspace(0.3, 0.8, len(avg_scores)))
    bars = ax2.bar(avg_scores.index, avg_scores.values, color=colors)
    ax2.set_title("Average Summary Quality Score by Medication", fontsize=16, pad=20)
    ax2.set_ylim(0, 10)
    ax2.set_ylabel("Average Score", fontsize=12)
    ax2.set_xlabel("Medication", fontsize=12)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)

    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 3. Radar chart - Replace grouped bar chart
    ax3 = fig.add_subplot(gs[3, 0], polar=True)

    # Setup for radar chart
    categories = list(df.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Set the angle labels (category names)
    ax3.set_theta_offset(np.pi / 2)
    ax3.set_theta_direction(-1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)

    # Draw y-axis grid lines and labels
    ax3.set_rlabel_position(0)
    ax3.set_yticks([2, 4, 6, 8, 10])
    ax3.set_yticklabels(['2', '4', '6', '8', '10'])
    ax3.set_ylim(0, 10)

    # Sort by average score so best drugs appear first
    drug_order = df.mean(axis=1).sort_values(ascending=False).index.tolist()

    # Plot each drug - use a color cycle that matches the bar chart
    colors = plt.cm.YlGnBu(np.linspace(0.3, 0.8, len(drug_order)))

    for i, drug in enumerate(drug_order):
        values = df.loc[drug].values.tolist()
        values += values[:1]  # Close the loop

        # Plot the values
        ax3.plot(angles, values, linewidth=2, linestyle='solid',
                 color=colors[i], label=drug)
        ax3.fill(angles, values, color=colors[i], alpha=0.1)

    ax3.set_title("Quality Dimensions by Medication", fontsize=16, pad=20)

    # Add legend with same position as the original grouped bar chart
    ax3.legend(title="Medication", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add a note at the bottom of the figure
    fig.text(0.5, 0.01,
             "Note: Higher scores indicate better quality summaries. Scores reflect the completeness and usefulness of the information provided.",
             ha='center', fontsize=10, style='italic')

    # Adjust the layout with better control
    fig.subplots_adjust(top=0.92, bottom=0.05, left=0.1, right=0.85)

    # Save the figure with high quality
    filename = f"{condition.replace(' ', '_')}_medication_evaluation.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)  # Close the figure to avoid memory leaks
    return filename


def display_header(msg: str, width=60, char="="):
    """Display a formatted header message"""
    print(f"{char * width}\n{msg}\n{char * width}")
