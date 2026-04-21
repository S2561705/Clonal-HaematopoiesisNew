
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
CSV_FILE = '../exports/MDS_cohort_mutation_table.csv'
OUTPUT_DIR = '../exports/figures/'


def load_and_summarize_table(csv_path):
    """Load the combined table and print comprehensive summary."""
    
    print("="*70)
    print("LOADING COHORT MUTATION TABLE")
    print("="*70)
    
    # Check if file exists
    if not Path(csv_path).exists():
        print(f"\\nERROR: File not found: {csv_path}")
        print("Please run the plotting script first to generate the CSV.")
        return None
    
    # Load data
    df = pd.read_csv(csv_path)
    
    print(f"\\nFile: {csv_path}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Basic counts
    print(f"\\n{'='*70}")
    print("PARTICIPANT SUMMARY")
    print("="*70)
    
    n_participants = df['Participant_ID'].nunique()
    print(f"Total participants: {n_participants}")
    
    participant_summary = df.groupby('Participant_ID').agg({
        'Gene': 'count',
        'Clone_Index': 'nunique',
        'Role': lambda x: sum(x == 'Leading')
    }).rename(columns={
        'Gene': 'Total_Mutations',
        'Clone_Index': 'N_Clones', 
        'Role': 'Leading_Mutations'
    })
    participant_summary['Subclone_Mutations'] = (
        participant_summary['Total_Mutations'] - participant_summary['Leading_Mutations']
    )
    
    print("\\n" + participant_summary.to_string())
    
    # Gene frequency
    print(f"\\n{'='*70}")
    print("GENE FREQUENCY (across all participants)")
    print("="*70)
    
    gene_counts = df['Gene'].value_counts()
    print(gene_counts.head(15).to_string())
    
    # Fitness summary
    print(f"\\n{'='*70}")
    print("FITNESS (s) SUMMARY")
    print("="*70)
    
    fit_summary = df.groupby('Role')['Fitness_MAP'].describe()[['mean', 'std', 'min', 'max', '50%']]
    print("\\nBy Role:")
    print(fit_summary.to_string())
    
    print(f"\\nOverall: mean={df['Fitness_MAP'].mean():.3f}, median={df['Fitness_MAP'].median():.3f}")
    print(f"Range: [{df['Fitness_MAP'].min():.3f}, {df['Fitness_MAP'].max():.3f}]")
    
    # h summary
    print(f"\\n{'='*70}")
    print("ZYGOSITY (h) SUMMARY")
    print("="*70)
    
    h_summary = df.groupby('Role')['h_MAP'].describe()[['mean', 'std', 'min', 'max', '50%']]
    print("\\nBy Role:")
    print(h_summary.to_string())
    
    print(f"\\nOverall: mean={df['h_MAP'].mean():.3f}, median={df['h_MAP'].median():.3f}")
    print(f"Range: [{df['h_MAP'].min():.3f}, {df['h_MAP'].max():.3f}]")
    
    # h_global check
    print(f"\\n{'='*70}")
    print("GLOBAL CONSTRAINT CHECK")
    print("="*70)
    
    h_global_by_participant = df.groupby('Participant_ID')['h_global'].first()
    print("\\nh_global per participant:")
    for pid, h_g in h_global_by_participant.items():
        part_df = df[df['Participant_ID'] == pid]
        mean_h = part_df['h_MAP'].mean()
        status = "✓" if mean_h >= h_g * 0.9 else "⚠"
        print(f"  {pid}: h_global={h_g:.3f}, mean_h={mean_h:.3f} {status}")
    
    return df


def create_summary_plots(df, output_dir):
    """Create summary visualizations of the cohort data."""
    
    print(f"\\n{'='*70}")
    print("CREATING SUMMARY PLOTS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Cohort Summary: Fitness and Zygosity Estimates', fontsize=14, fontweight='bold')
    
    # 1. Fitness distribution by role
    ax = axes[0, 0]
    for role, color in zip(['Leading', 'Sub-clone'], ['#1f77b4', '#ff7f0e']):
        data = df[df['Role'] == role]['Fitness_MAP']
        ax.hist(data, bins=20, alpha=0.6, label=role, color=color, edgecolor='black')
    ax.set_xlabel('Fitness (s)')
    ax.set_ylabel('Count')
    ax.set_title('Fitness Distribution by Role')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. h distribution by role
    ax = axes[0, 1]
    for role, color in zip(['Leading', 'Sub-clone'], ['#1f77b4', '#ff7f0e']):
        data = df[df['Role'] == role]['h_MAP']
        ax.hist(data, bins=20, alpha=0.6, label=role, color=color, edgecolor='black')
    ax.set_xlabel('Homozygous fraction (h)')
    ax.set_ylabel('Count')
    ax.set_title('Zygosity Distribution by Role')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Fitness vs h scatter
    ax = axes[0, 2]
    for role, color in zip(['Leading', 'Sub-clone'], ['#1f77b4', '#ff7f0e']):
        role_df = df[df['Role'] == role]
        ax.scatter(role_df['Fitness_MAP'], role_df['h_MAP'], 
                  alpha=0.6, label=role, color=color, s=50, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Fitness (s)')
    ax.set_ylabel('Homozygous fraction (h)')
    ax.set_title('Fitness vs Zygosity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Gene frequency
    ax = axes[1, 0]
    gene_counts = df['Gene'].value_counts().head(10)
    gene_counts.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
    ax.set_xlabel('Count')
    ax.set_title('Top 10 Genes by Frequency')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 5. Mutations per participant
    ax = axes[1, 1]
    mut_counts = df.groupby('Participant_ID').size()
    mut_counts.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
    ax.set_xlabel('Participant')
    ax.set_ylabel('Number of Mutations')
    ax.set_title('Mutations per Participant')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. h_global vs mean h per participant
    ax = axes[1, 2]
    participant_stats = df.groupby('Participant_ID').agg({
        'h_global': 'first',
        'h_MAP': 'mean',
        'Fitness_MAP': 'mean'
    })
    ax.scatter(participant_stats['h_global'], participant_stats['h_MAP'], 
              s=100, alpha=0.7, color='green', edgecolor='black')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x (perfect match)')
    ax.set_xlabel('Global h constraint')
    ax.set_ylabel('Mean observed h')
    ax.set_title('Constraint Satisfaction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = f"{output_dir}/cohort_summary_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary plots to: {out_path}")
    plt.close()


def interactive_explore(df):
    """Show example queries users can run."""
    
    print(f"\\n{'='*70}")
    print("INTERACTIVE EXPLORATION")
    print("="*70)
    print("\\nDataframe loaded as 'df'. Available commands:")
    print("  df.head()                    - First 5 rows")
    print("  df.tail()                    - Last 5 rows")
    print("  df[df['Gene'] == 'DNMT3A']   - Filter by gene")
    print("  df[df['Role'] == 'Leading']  - Only leading mutations")
    print("  df.sort_values('Fitness_MAP', ascending=False)  - Top fitness")
    print("  df.groupby('Participant_ID')['h_MAP'].mean()  - Mean h per patient")
    print("\\nExample queries:")
    
    # Show top fitness mutations
    print("\\n1. Top 5 mutations by fitness:")
    top_fit = df.nlargest(5, 'Fitness_MAP')[['Participant_ID', 'Gene', 'Role', 'Fitness_MAP', 'h_MAP']]
    print(top_fit.to_string(index=False))
    
    # Show highest h mutations
    print("\\n2. Top 5 mutations by h (most homozygous):")
    top_h = df.nlargest(5, 'h_MAP')[['Participant_ID', 'Gene', 'Role', 'Fitness_MAP', 'h_MAP']]
    print(top_h.to_string(index=False))
    
    # Show by gene
    print("\\n3. DNMT3A mutations summary:")
    dnmt3a = df[df['Gene'] == 'DNMT3A']
    if len(dnmt3a) > 0:
        print(dnmt3a[['Participant_ID', 'cDNA_Change', 'Role', 'Fitness_MAP', 'h_MAP']].to_string(index=False))
    else:
        print("   No DNMT3A mutations found")


def main():
    """Main function to load and display results."""
    
    # Load and summarize
    df = load_and_summarize_table(CSV_FILE)
    
    if df is None:
        return
    
    # Create plots
    create_summary_plots(df, OUTPUT_DIR)
    create_cohort_table_figure(df, output_path='../exports/figures/cohort_mutation_table.png')
    
    # Interactive exploration
    interactive_explore(df)
    
    print(f"\\n{'='*70}")
    print("SCRIPT COMPLETE")
    print("="*70)
    print(f"\\nTo explore further, run:")
    print(f"  python -i view_cohort_results.py")
    print(f"\\nThen interact with the dataframe 'df' directly.")
    
    return df

# Create a script that generates a clean table figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import numpy as np

def create_cohort_table_figure(df, output_path='../exports/figures/cohort_mutation_table.png'):
    """
    Create a publication-quality table figure showing all mutations.
    
    Layout: One big table with all mutations, color-coded by participant and clone.
    """
    
    # Prepare data for display
    display_df = df.copy()
    
    # Create formatted strings
    display_df['Fitness'] = display_df.apply(
        lambda x: f"{x['Fitness_MAP']:.3f}", axis=1
    )
    display_df['Fitness_CI'] = display_df.apply(
        lambda x: f"[{x['Fitness_5%']:.3f}-{x['Fitness_95%']:.3f}]", axis=1
    )
    display_df['h'] = display_df.apply(
        lambda x: f"{x['h_MAP']:.3f}", axis=1
    )
    display_df['h_CI'] = display_df.apply(
        lambda x: f"[{x['h_5%']:.3f}-{x['h_95%']:.3f}]", axis=1
    )
    
    # Sort by participant, then clone, then role (leading first)
    display_df['Role_Sort'] = display_df['Role'].apply(lambda x: 0 if x == 'Leading' else 1)
    display_df = display_df.sort_values(['Participant_ID', 'Clone_Index', 'Role_Sort', 'Gene'])
    
    # Create figure
    n_rows = len(display_df)
    row_height = 0.25
    header_height = 0.5
    fig_height = max(10, n_rows * row_height + header_height + 2)
    
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_rows * row_height + header_height + 1)
    ax.axis('off')
    
    # Title
    fig.suptitle('Cohort Mutation Table: Fitness and Zygosity Estimates', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Column headers
    headers = ['Participant', 'Gene', 'cDNA Change', 'Role', 'Clone', 
               'Fitness (s)', '95% CI', 'h', '95% CI', 'h_global']
    col_widths = [1.2, 0.8, 2.0, 0.8, 0.6, 1.0, 1.4, 0.8, 1.4, 0.8]
    col_x = [0]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)
    
    # Draw header
    y_pos = n_rows * row_height + header_height
    for i, (header, x, w) in enumerate(zip(headers, col_x, col_widths)):
        rect = FancyBboxPatch((x, y_pos), w, header_height, 
                             boxstyle="round,pad=0.02", 
                             facecolor='#4472C4', edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + w/2, y_pos + header_height/2, header, 
               ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Color scheme
    participants = display_df['Participant_ID'].unique()
    participant_colors = plt.cm.tab20(np.linspace(0, 1, len(participants)))
    participant_color_map = dict(zip(participants, participant_colors))
    
    # Draw rows
    current_y = y_pos
    current_participant = None
    
    for idx, (_, row) in enumerate(display_df.iterrows()):
        current_y -= row_height
        
        # Alternate row background slightly
        if row['Participant_ID'] != current_participant:
            current_participant = row['Participant_ID']
            base_color = participant_color_map[current_participant]
            # Lighten for background
            bg_color = tuple(min(1.0, c + 0.75) for c in base_color[:3]) + (0.3,)
        else:
            bg_color = 'white'
        
        # Draw row background
        rect = FancyBboxPatch((0, current_y), sum(col_widths), row_height,
                             boxstyle="round,pad=0.01",
                             facecolor=bg_color, edgecolor='gray', linewidth=0.5, alpha=0.3)
        ax.add_patch(rect)
        
        # Draw cell text
        values = [
            row['Participant_ID'],
            row['Gene'],
            row['cDNA_Change'],
            row['Role'],
            str(int(row['Clone_Index'])),
            row['Fitness'],
            row['Fitness_CI'],
            row['h'],
            row['h_CI'],
            f"{row['h_global']:.3f}"
        ]
        
        for i, (val, x, w) in enumerate(zip(values, col_x, col_widths)):
            # Bold for leading mutations
            weight = 'bold' if row['Role'] == 'Leading' and i in [1, 3] else 'normal'
            ax.text(x + w/2, current_y + row_height/2, str(val),
                   ha='center', va='center', fontsize=8, fontweight=weight)
    
    # Add legend for participants
    legend_y = -0.5
    legend_x = 0
    ax.text(legend_x, legend_y, 'Participants: ', fontsize=9, fontweight='bold')
    
    for i, (pid, color) in enumerate(participant_color_map.items()):
        rect = mpatches.Rectangle((legend_x + 1.5 + i*2.5, legend_y - 0.15), 
                                  0.3, 0.3, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(legend_x + 1.9 + i*2.5, legend_y, pid, fontsize=8, va='center')
    
    # Footer notes
    footer_y = -1.2
    ax.text(0, footer_y, 
           'Note: Bold = Leading mutation per clone. h = homozygous fraction. s = selection coefficient. '
           'h_global = minimum population average h from VAF sum constraint.',
           fontsize=8, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Table figure saved to: {output_path}")
    plt.close()
    
    return fig



if __name__ == "__main__":
    df = main()
