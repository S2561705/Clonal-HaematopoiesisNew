"""
Threshold sensitivity plots for MDS clonal structure inference.
Matches PI's whiteboard sketch: diagonal decreasing lines.
x-axis = confidence threshold (%)
y-axis = posterior probability (%)
Each model line runs from (0, posterior%) diagonally down to (posterior%, 0)
then stays at zero — the model drops out when threshold exceeds its posterior.
"""

import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ==============================================================================
# Configuration
# ==============================================================================

INPUT_FILE = '../exports/MDS/MDS_cohort_fitted.pk'
OUTPUT_DIR = '../exports/MDS/figures/'
SHOW_PLOTS = False
DPI = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


# ==============================================================================
# Helpers
# ==============================================================================

def parse_model_dict(model_dict):
    entries = []
    for k, v in model_dict.items():
        if len(v) == 3:
            cs, raw, norm = v
        else:
            cs, raw = v
            norm = None
        entries.append({'key': k, 'cs': cs, 'raw': raw, 'norm': norm})

    if any(e['norm'] is None for e in entries):
        total = sum(e['raw'] for e in entries)
        for e in entries:
            e['norm'] = e['raw'] / total if total > 0 else 0.0

    return sorted(entries, key=lambda x: x['norm'], reverse=True)


def format_cs_short(cs, obs_index):
    clone_strs = []
    for clone in cs:
        names = [obs_index[j].split(' ')[0] for j in clone]
        clone_strs.append('+'.join(names))
    return ' | '.join(clone_strs)


# ==============================================================================
# Main plot
# ==============================================================================

def plot_threshold_sensitivity(part, ax=None, threshold_max=50):
    """
    Each model draws a diagonal line from (0, posterior%) down to (posterior%, 0).
    Beyond that point the model is excluded by the threshold.
    This matches the PI's whiteboard sketch.
    """
    participant_id = part.uns.get('participant_id', 'Unknown')
    entries = parse_model_dict(part.uns.get('model_dict', {}))
    obs_index = list(part.obs.index)

    nonzero = [e for e in entries if e['norm'] > 0]
    all_zero = len(nonzero) == 0

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    if all_zero:
        ax.text(0.5, 0.5, 'All model probabilities zero\n(inference failed)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='red', fontstyle='italic')
        ax.set_title(f'{participant_id}: Threshold Sensitivity', fontweight='bold')
        if standalone:
            plt.tight_layout()
            return plt.gcf()
        return

    n = len(nonzero)
    palette = sns.color_palette("husl", n_colors=n)

    x_fine = np.linspace(0, threshold_max, 1000)

    for rank, (entry, color) in enumerate(zip(nonzero, palette)):
        norm_pct = entry['norm'] * 100
        label = format_cs_short(entry['cs'], obs_index)

        # Diagonal from (0, norm_pct) to (norm_pct, 0), then flat at 0
        y = np.where(x_fine <= norm_pct, norm_pct - x_fine, 0.0)

        mask_active = x_fine <= norm_pct
        mask_zero   = x_fine >  norm_pct

        ax.plot(x_fine[mask_active], y[mask_active],
                color=color, linewidth=2.2,
                label=f'{norm_pct:.1f}%  {label}',
                solid_capstyle='round')

        if mask_zero.any():
            ax.plot(x_fine[mask_zero], y[mask_zero],
                    color=color, linewidth=0.8, linestyle='--', alpha=0.2)

        # Dot where line hits x-axis
        ax.scatter([norm_pct], [0], color=color, s=40, zorder=5, clip_on=False)

        # Label at y-intercept
        ax.text(-0.5, norm_pct, f'{norm_pct:.1f}%',
                ha='right', va='center', fontsize=7.5, color=color, fontweight='bold')

    # Reference threshold lines
    for thresh, style, lbl in [(1, ':', '1%'), (5, '--', '5%'), (10, '-.', '10%')]:
        if thresh <= threshold_max:
            ax.axvline(thresh, color='gray', linestyle=style, linewidth=1, alpha=0.5)
            ax.text(thresh + 0.4, 102, lbl, fontsize=8, color='gray', va='bottom')

    ax.set_xlim([0, threshold_max])
    ax.set_ylim([0, 105])
    ax.set_xlabel('Confidence threshold (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model posterior probability (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{participant_id}: Threshold Sensitivity  '
                 f'({n} model{"s" if n != 1 else ""} with non-zero posterior)',
                 fontsize=12, fontweight='bold')

    ax.legend(title='Posterior  |  Structure',
              bbox_to_anchor=(1.01, 1), loc='upper left',
              fontsize=8, title_fontsize=9, framealpha=0.9)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    if standalone:
        plt.tight_layout()
        return plt.gcf()


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 70)
    print("THRESHOLD SENSITIVITY PLOTS (diagonal lines)")
    print("=" * 70)

    try:
        with open(INPUT_FILE, 'rb') as f:
            processed_parts = pk.load(f)
        print(f"Loaded {len(processed_parts)} participants")
    except FileNotFoundError:
        print(f"ERROR: {INPUT_FILE} not found")
        return

    success = 0
    for i, part in enumerate(processed_parts):
        participant_id = part.uns.get('participant_id', f'participant_{i}')
        print(f"\n[{i+1}/{len(processed_parts)}] {participant_id}")

        try:
            fig = plot_threshold_sensitivity(part, threshold_max=50)

            if fig is not None:
                filename = f"{OUTPUT_DIR}{participant_id}_threshold_sensitivity.png"
                fig.savefig(filename, bbox_inches='tight', dpi=DPI)
                print(f"  Saved: {filename}")
                plt.close(fig)
                success += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. {success}/{len(processed_parts)} plots saved to {OUTPUT_DIR}")

    if SHOW_PLOTS:
        plt.show()


if __name__ == '__main__':
    main()