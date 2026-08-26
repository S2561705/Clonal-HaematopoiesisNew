"""
manual_structure.py
====================
Step 1 of the staged workflow: force every mutation heterozygous (h=0) as a
simplifying first pass, then hand-pick the clonal structure by eye and
install it directly, bypassing the automated partition search
(find_valid_clonal_structures / compute_clonal_models_prob_vec) entirely.

Zygosity inference on your hand-picked structure, and subclone fitness,
come next as separate steps -- this module only covers getting a manual
structure installed and ready to hand to the existing refine functions.
"""
import sys
sys.path.append("..")
import numpy as np


def set_all_heterozygous(part):
    """Force h_fixed = 0 (pure het, no LOH) for every mutation. Intended as
    a simplifying first pass: with zygosity pinned to a single known value,
    the resulting total_cells / VAF trajectories can be inspected by eye to
    manually assign mutations to clones, without zygosity inference and
    structure search fighting each other during exploration."""
    part.obs['h_fixed'] = 0.0
    return part


def clear_h_fixed(part, mutation_indices=None):
    """Reset h_fixed to unpinned (NaN) so zygosity inference can run freely
    again, once you're ready for that step. mutation_indices: optional list
    of positional row indices to unpin; default unpins every mutation."""
    if 'h_fixed' not in part.obs.columns:
        part.obs['h_fixed'] = np.nan
        return part
    if mutation_indices is None:
        part.obs['h_fixed'] = np.nan
    else:
        vals = part.obs['h_fixed'].values.astype(float)
        vals[np.array(mutation_indices)] = np.nan
        part.obs['h_fixed'] = vals
    return part


def set_manual_clonal_structure(part, cs):
    """Bypass find_valid_clonal_structures / compute_clonal_models_prob_vec
    entirely: install a hand-picked clonal structure directly into
    part.uns['model_dict'], in the exact shape
    refine_optimal_model_posterior_vec / _coordinate already expect
    (they read list(part.uns['model_dict'].values())[0][0]).

    cs: list of lists of mutation POSITIONAL indices (0-based, matching
    part.obs row order), e.g. [[0, 1], [2]] means mutations 0 and 1 share
    clone 0, mutation 2 is its own clone. No score is computed or needed --
    you're not asking the pipeline to rank this against alternatives.
    """
    n_mutations = part.shape[0]
    all_idx = sorted(i for clone in cs for i in clone)
    if all_idx != list(range(n_mutations)):
        raise ValueError(
            f"cs must partition all {n_mutations} mutations exactly once "
            f"(0..{n_mutations - 1}); got indices {all_idx} from cs={cs}"
        )
    part.uns['model_dict'] = {'model_0': (cs, None)}
    part.uns['warning'] = None
    return part