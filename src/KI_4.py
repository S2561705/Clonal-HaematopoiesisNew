import sys
sys.path.append("..")

from src.general_imports import *

import jax
jax.config.update("jax_enable_x64", True)
from jax import jit
from functools import partial
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jsp_stats
from itertools import combinations
import itertools
from jax.scipy.special import logsumexp
from jax.scipy.special import betaln, gammaln
from scipy.special import gammaln as _sp_gammaln, betaln as _sp_betaln

from scipy.special import betaincinv as _sp_betaincinv

N_w = 1e5
BETA_RESOLUTION = 1_000

# ---------------------------------------------------------------------------
# Bounds q_eps (see compute_beta_bounds / compute_global_variables below).
#
# NOTE ON FIX HISTORY: an earlier version of this module placed quadrature
# nodes at Beta(AO+1, DP-AO+1) quantiles uniform in probability q, then
# pushed them through x = -N_w_cond*beta/(beta-half). That was confirmed
# (via node_stability_check / tail_gap_check / interior_h_convergence_check
# across several turns of debugging) to produce spacing in x that does NOT
# shrink proportionally with resolution -- true at every h tested, not just
# near the h=1 pole -- so successive-resolution deltas GREW rather than
# shrank. Root cause: uniform-in-q spacing is not uniform in value wherever
# the Beta CDF is flat (its own tails), and the x-transform compounds this.
#
# FIX 3 (this version): use Beta quantiles ONLY to locate the two outer
# EDGES of the integration domain (an h-independent, once-per-participant
# computation), then lay `resolution` nodes UNIFORMLY IN X between those
# edges once per h-combo. Spacing is now exactly
# (x_hi - x_lo) / (resolution - 1) everywhere, so it shrinks linearly with
# resolution by construction. Q_EPS here only controls how far out the
# domain edges sit (i.e. how much of the AO/DP posterior's extreme tail
# mass is excluded), which is a much less sensitive knob than before --
# see check_qeps_stability, now repurposed to confirm this.
# ---------------------------------------------------------------------------
Q_EPS = 1e-4


def beta_binom_logpmf(k, n, p, phi):
    alpha = p * phi
    beta = (1.0 - p) * phi

    return (
        gammaln(n + 1)
        - gammaln(k + 1)
        - gammaln(n - k + 1)
        + betaln(k + alpha, n - k + beta)
        - betaln(alpha, beta)
    )

def log_trapz(log_y, x, axis=-1):
    m = jnp.max(log_y, axis=axis, keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    val = jsp.integrate.trapezoid(jnp.exp(log_y - m), x=x, axis=axis)
    return jnp.squeeze(m, axis=axis) + jnp.log(val)


def _fill_nearest(v, obs):
    v = np.array(v, dtype=float)
    obs = np.array(obs, dtype=bool)
    last = np.nan
    for t in range(len(v)):
        if obs[t]:
            last = v[t]
        else:
            v[t] = last
    nxt = np.nan
    for t in range(len(v) - 1, -1, -1):
        if obs[t]:
            nxt = v[t]
        elif np.isnan(v[t]):
            v[t] = nxt
    return v


def compute_carry_idx(observed):
    observed = np.array(observed, dtype=bool)
    n_tp, n_mut = observed.shape
    carry = np.full((n_tp, n_mut), -1, dtype=int)
    for j in range(n_mut):
        last = -1
        for t in range(n_tp):
            if observed[t, j]:
                last = t
            carry[t, j] = last
        nxt = -1
        for t in range(n_tp - 1, -1, -1):
            if observed[t, j]:
                nxt = t
            if carry[t, j] == -1:
                carry[t, j] = nxt
    return carry


def _get_arrays(part):
    AO = np.array(part.layers['AO'].T)
    DP = np.array(part.layers['DP'].T)
    observed = DP > 0
    carry_idx = compute_carry_idx(observed)
    time_points = np.array(part.var['time_points'], dtype=float)
    if 'h_fixed' in part.obs.columns:
        h_fixed = np.array(part.obs['h_fixed'].values, dtype=float)
    else:
        h_fixed = np.full(part.shape[0], np.nan)
    return (jnp.array(AO), jnp.array(DP), jnp.array(observed),
            jnp.array(carry_idx), jnp.array(time_points), h_fixed)


# ---------------------------------------------------------------------------
# Obligate-heterozygous gene handling
# ---------------------------------------------------------------------------
# Some genes are known never to present as biallelic/homozygous in this
# context. For these, zygosity shouldn't be inferred -- it should be FIXED
# at h=0 before any inference, exactly like a user-supplied h_fixed value.
#
# Fitness/zygosity of OTHER mutations in the same participant is already
# informed automatically once a clone is pinned: total_cells is solved
# JOINTLY across every clone in compute_deterministic_size, so pinning one
# clone's h changes the shared wild-type pool every other clone's
# likelihood is computed against. No separate propagation code is needed
# for that part.
#
# What's NOT automatic is the clonal-structure SEARCH knowing about the
# pin -- that's handled via compute_invalid_combinations below, which now
# also excludes any candidate clone that would cluster two mutations with
# mutually incompatible h_fixed values (instead of letting such a
# structure through and crashing later on build_clone_h_grids' consistency
# assert).
# ---------------------------------------------------------------------------

def flag_obligate_heterozygous(part, gene_set, gene_col='GENE'):
    """Set h_fixed = 0.0 for mutations in obligate-heterozygous genes.
    Call once per participant, BEFORE compute_clonal_models_prob_vec, so
    the pin informs the structure search itself (via compute_invalid_
    combinations below), not just the final fitness/zygosity refinement.

    Warns rather than silently overwriting if a mutation already has a
    conflicting h_fixed value -- that's a data inconsistency worth seeing,
    not hiding.
    """
    if gene_col not in part.obs.columns:
        return part
    if 'h_fixed' not in part.obs.columns:
        part.obs['h_fixed'] = np.nan
    if 'h_fixed_reason' not in part.obs.columns:
        part.obs['h_fixed_reason'] = None

    is_obligate = part.obs[gene_col].isin(gene_set)
    conflict = is_obligate & part.obs['h_fixed'].notna() & (part.obs['h_fixed'] != 0.0)
    if conflict.any():
        bad = part.obs.loc[conflict, gene_col].tolist()
        print(f"  [warn] {part.uns.get('participant_id', '?')}: obligate-het "
              f"genes with conflicting pre-set h_fixed, NOT overwritten: {bad}")

    to_set = is_obligate & part.obs['h_fixed'].isna()
    part.obs.loc[to_set, 'h_fixed'] = 0.0
    part.obs.loc[to_set, 'h_fixed_reason'] = 'obligate_heterozygous_gene'
    return part


def compute_deterministic_size(cs, AO, DP, n_mutations, h, observed=None):
    AO = np.array(AO)
    DP = np.array(DP)
    if observed is None:
        observed = DP > 0
    observed = np.array(observed, dtype=bool)
    vaf = np.where(observed, AO / np.where(DP > 0, DP, 1.0), np.nan)

    lm = []
    clonal_map = np.zeros(n_mutations, dtype=int)
    for i, cs_idx in enumerate(cs):
        summed = np.nansum(vaf[:, cs_idx], axis=0)
        lm.append(cs_idx[int(np.argmax(summed))])
        clonal_map[np.array(cs_idx)] = i

    v_lead = np.stack([_fill_nearest(vaf[:, lead], observed[:, lead])
                       for lead in lm], axis=1)

    h = np.array(h, dtype=float)
    denom = 1.0 - 2.0 * np.sum(v_lead / (1.0 + h)[None, :], axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        total_cells = np.ceil(N_w / denom)  # inf/negative at the h_min boundary by construction;

    h_mut = h[clonal_map]

    vaf_filled = np.stack([_fill_nearest(vaf[:, j], observed[:, j])
                           for j in range(n_mutations)], axis=1)
    deterministic_size = 2.0 * vaf_filled * total_cells[:, None] / (1.0 + h_mut)[None, :]

    return (jnp.array(deterministic_size), jnp.array(total_cells),
            jnp.array(clonal_map), jnp.array(h_mut))


def partition(collection):
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        yield [[first]] + smaller


def compute_beta_bounds(AO, DP, observed, q_eps=Q_EPS):
    """Extreme Beta(AO+1, DP-AO+1) quantiles used ONLY to locate the outer
    EDGES of the x-integration domain -- not to place interior nodes (see
    compute_global_variables for why). AO/DP-only and h-independent, so
    compute once per participant and reuse across every h-combo / s-sweep,
    same caching contract as the old compute_beta_quantiles."""
    AO = np.array(AO)
    DP = np.array(DP)
    observed = np.array(observed, dtype=bool)
    DP_safe = np.where(observed, DP, 1.0)
    AO_safe = np.where(observed, AO, 0.0)
    a = AO_safe + 1
    b = DP_safe - AO_safe + 1
    beta_lo = _sp_betaincinv(a, b, q_eps)
    beta_hi = _sp_betaincinv(a, b, 1.0 - q_eps)
    return jnp.array(beta_lo), jnp.array(beta_hi)   # each (n_tp, n_mut)


@partial(jit, static_argnames=['resolution'])
def compute_global_variables(s_vec, AO, DP, total_cells, deterministic_size,
                             time_points, h_mut, observed, carry_idx,
                             beta_lo, beta_hi, resolution=BETA_RESOLUTION):
    """
    ------------------------------------------------------------------------
    FIX 3 (uniform-in-x quadrature nodes). See module-level note above for
    the diagnosis. beta_lo/beta_hi (from compute_beta_bounds) locate the
    domain edges; nodes are then laid uniformly in x between them, so
    spacing shrinks linearly with resolution everywhere -- confirmed by
    uniform_x_convergence_check against the old scheme's non-convergence.

    FIX 4 (explicit seeding-timepoint prior, see end of function): Fix 3
    alone still showed a flat, non-shrinking Δmax under resolution sweeps.
    Root cause: the old "-log(resolution)" seeding term was only a valid
    Beta-prior estimate because the old nodes WERE Beta quantiles (so
    true_vaf == beta at each node, by an exact algebraic identity -- see
    identity_check.py). Under uniform-in-x nodes that coincidence is gone,
    so the Beta(AO0+1, DP0-AO0+1) prior density is now added back
    explicitly, expressed in x via its Jacobian, at the seeding timepoint.
    ------------------------------------------------------------------------
    """
    n_mutations = AO.shape[1]
    delta_t = jnp.diff(time_points)
    exp_term_vec_s = jnp.exp(delta_t * s_vec[:, None])
    exp_term_vec_s = jnp.reshape(exp_term_vec_s, (*exp_term_vec_s.shape, 1, 1))

    DP_safe = jnp.where(observed, DP, 1.0)
    AO_safe = jnp.where(observed, AO, 0.0)

    N_w_cond_vec = (total_cells[:, None] - deterministic_size)[:, :, None]  # (tp, mut, 1)
    half = (1.0 + h_mut)[None, :, None] / 2.0                               # (1, mut, 1)

    # If beta_hi sits past half (posterior tail crosses the pole for this h),
    # clamp it just below half so x_hi stays finite. This only ever affects
    # the domain BOUND, not per-node placement, so it can't reproduce the
    # old scheme's interior distortion.
    beta_lo_ = beta_lo[:, :, None]
    beta_hi_ = jnp.minimum(beta_hi[:, :, None], half * (1.0 - 1e-9))

    x_lo = -N_w_cond_vec * beta_lo_ / (beta_lo_ - half)
    x_hi = -N_w_cond_vec * beta_hi_ / (beta_hi_ - half)
    lo = jnp.minimum(x_lo, x_hi)
    hi = jnp.maximum(x_lo, x_hi)

    frac = jnp.linspace(0.0, 1.0, resolution).reshape(1, 1, resolution)
    x_vec = lo + frac * (hi - lo)               # uniform in x -- the actual fix
    valid = jnp.ones_like(x_vec, dtype=bool)    # domain is closed by construction now

    true_vaf_vec = (1.0 + h_mut)[None, :, None] * x_vec / (2.0 * (N_w_cond_vec + x_vec))

    phi = 200.
    log_p_y = beta_binom_logpmf(
        AO_safe[:, :, None], DP_safe[:, :, None], true_vaf_vec, phi,
    )

    idx = jnp.broadcast_to(carry_idx[:, :, None], x_vec.shape)
    x_vec = jnp.take_along_axis(x_vec, idx, axis=0)

    log_p_y = jnp.where(observed[:, :, None], log_p_y, 0.0)

    # ------------------------------------------------------------------
    # FIX 4: explicit prior density at the seeding timepoint, replacing
    # the old "-log(resolution)" shortcut. That shortcut was only valid
    # under quantile-sampled nodes (verified via identity_check.py:
    # true_vaf == beta exactly, so averaging p_y0 over Beta-quantile nodes
    # was a correct MC estimate of the integral against the Beta density).
    # Under uniform-in-x nodes that coincidence no longer holds, so the
    # prior must be added back explicitly as a density in x (Jacobian
    # |dbeta/dx| included), so the existing trapezoidal recursion -- which
    # already integrates correctly over whatever spacing the nodes have --
    # handles it properly regardless of node placement scheme.
    # ------------------------------------------------------------------
    idx0 = carry_idx[0, :]                                          # first-observed index per mutation
    AO0 = jnp.take_along_axis(AO_safe, idx0[None, :], axis=0)[0]     # (n_mut,)
    DP0 = jnp.take_along_axis(DP_safe, idx0[None, :], axis=0)[0]     # (n_mut,)
    a0, b0 = AO0 + 1.0, DP0 - AO0 + 1.0

    N_w_cond_0 = jnp.take_along_axis(N_w_cond_vec, idx0[None, :, None], axis=0)[0]  # (n_mut, 1)
    half_0 = half[0]                                                                # (n_mut, 1)
    x0 = x_vec[0, :, :]        # already carry-indexed to the seeding timepoint's own nodes
    beta0 = (2.0 * half_0) * x0 / (2.0 * (N_w_cond_0 + x0))   # = true_vaf at x0, i.e. beta0 == beta identically

    log_jacobian = jnp.log(half_0) + jnp.log(N_w_cond_0) - 2.0 * jnp.log(N_w_cond_0 + x0)
    prior_logpdf = ((a0[:, None] - 1.0) * jnp.log(beta0)
                    + (b0[:, None] - 1.0) * jnp.log1p(-beta0)
                    - betaln(a0[:, None], b0[:, None]))

    log_rec_vec = log_p_y[0, :, :] + prior_logpdf + log_jacobian
    return x_vec, exp_term_vec_s, log_rec_vec, log_p_y, n_mutations, valid

@jit
def BD_process_dynamics(s, x_vec, exp_term_vec):
    # NB params: n = mean^2 / (var - mean). Goes negative when var <= mean,
    # i.e. roughly (2λ+s)·Δt·e^{sΔt} < 1 (~Δt < 0.39 at small s, λ=1.3).
    # Closely spaced timepoints → NaN transitions independently of Fix 3.
    lamb = 1.3
    mean_vec = x_vec[:-1, :, :] * exp_term_vec
    variance_vec = x_vec[:-1] * (2 * lamb + s) * exp_term_vec * (exp_term_vec - 1) / s
    p_vec = mean_vec / variance_vec
    n_vec = jnp.power(mean_vec, 2) / (variance_vec - mean_vec)
    return p_vec, n_vec


@jit
def recursive_term_update(j, log_rec_i, x_i, p_i, n_i, log_p_y_i):
    log_bd = jsp_stats.nbinom.logpmf(x_i[j][:, None], p=p_i[j - 1], n=n_i[j - 1])
    log_inner = log_bd + log_rec_i[None, :]
    return log_p_y_i[j] + log_trapz(log_inner, x_i[j - 1], axis=-1)


def mutation_specific_ll(i, log_rec_vec, x_vec, p_vec, n_vec, log_p_y_vec, n_tps):
    log_rec_i = log_rec_vec[i]
    x_i = x_vec[:, i]; p_i = p_vec[:, i]; n_i = n_vec[:, i]
    log_p_y_i = log_p_y_vec[:, i]
    for j in range(1, n_tps):
        log_rec_i = recursive_term_update(j, log_rec_i, x_i, p_i, n_i, log_p_y_i)
    return log_trapz(log_rec_i, x_i[-1])


def fitness_specific_computations(s_idx, s_vec, x_vec, exp_term_vec_s,
                                  log_rec_vec, log_p_y_vec,
                                  time_points, n_mutations, cs):
    s = s_vec[s_idx]
    exp_term_vec = exp_term_vec_s[s_idx]
    p_vec, n_vec = BD_process_dynamics(s, x_vec, exp_term_vec)

    mutation_loglik = jax.vmap(
        mutation_specific_ll, in_axes=(0, None, None, None, None, None, None))(
            jnp.arange(n_mutations, dtype=int), log_rec_vec, x_vec,
            p_vec, n_vec, log_p_y_vec, time_points.shape[0])

    clonal_loglik = jnp.zeros(len(cs))
    for i, c_idx in enumerate(cs):
        clonal_loglik = clonal_loglik.at[i].set(
            jnp.sum(mutation_loglik[jnp.array(c_idx)]))
    return clonal_loglik


def jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs, deterministic_size,
                      total_cells, h_mut, observed, carry_idx, beta_lo, beta_hi,
                      resolution=BETA_RESOLUTION, return_valid=False):
    gv = compute_global_variables(s_vec, AO, DP, total_cells, deterministic_size,
                                  time_points, h_mut, observed, carry_idx,
                                  beta_lo, beta_hi, resolution=resolution)
    x_vec, exp_term_vec_s, log_rec_vec, log_p_y_vec, n_mutations, valid = gv
    s_idx = jnp.arange(s_vec.shape[0])
    out = jax.vmap(fitness_specific_computations,
                   in_axes=(0, None, None, None, None, None, None, None, None))(
                   s_idx, s_vec, x_vec, exp_term_vec_s, log_rec_vec,
                   log_p_y_vec, time_points, n_mutations, cs)
    if return_valid:
        return out, valid
    return out


def build_clone_h_grids(cs, AO, DP, observed, h_fixed, h_resolution, max_h=1.0):
    AO = np.array(AO)
    DP = np.array(DP)
    observed = np.array(observed, dtype=bool)
    vaf = np.where(observed, AO / np.where(DP > 0, DP, 1.0), np.nan)
    h_fixed = np.array(h_fixed)

    grids = []
    for cs_idx in cs:
        hf = h_fixed[np.array(cs_idx)]
        pinned = hf[~np.isnan(hf)]
        if pinned.size:
            assert np.allclose(pinned, pinned[0]), \
                "clone mixes conflicting fixed-h values"
            grids.append(jnp.array([float(pinned[0])]))
        else:
            lead = cs_idx[int(np.argmax(np.nansum(vaf[:, cs_idx], axis=0)))]
            v_max = np.nanmax(vaf[:, lead])
            h_min = min(max(2 * v_max - 1, 0.0), max_h)
            grids.append(jnp.linspace(h_min, max_h, h_resolution))
    return grids


def compute_cs_posterior_grid_vec(s_vec, h_grids, AO, DP, time_points, cs,
                                  observed, carry_idx, beta_lo, beta_hi,
                                  resolution=BETA_RESOLUTION):
    n_clones = len(cs)
    out_list, hcombo_list = [], []
    for idx in itertools.product(*[range(len(g)) for g in h_grids]):
        h = jnp.array([h_grids[k][idx[k]] for k in range(n_clones)])
        det_size, total_cells, _, h_mut = compute_deterministic_size(
            cs, AO, DP, AO.shape[1], h, observed)
        feasible = jnp.all(jnp.isfinite(total_cells) & (total_cells > 0))
        out = jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs,
                                det_size, total_cells, h_mut, observed,
                                carry_idx, beta_lo, beta_hi, resolution=resolution)
        out_list.append(jnp.where(feasible, out, -jnp.inf))
        hcombo_list.append(h)
    return jnp.stack(out_list), jnp.stack(hcombo_list)


def compute_model_likelihood(log_out_grid, cs, s_vec, h_grids):
    """Log-space. Returns a LOG model probability (fine for ranking).

    ------------------------------------------------------------------------
    FIX 1 (structure-dependent prior) -- a deliberate modelling CHOICE.

    Each structure's h-grid spans [h_min_k, max_h], so integrating with
    equal weight per grid point implies h_k ~ U[h_min_k, max_h] -- a prior
    density of 1/(max_h - h_min_k) that differs by structure. The
    model-independent alternative is h_k ~ U[0, max_h] for every k (a
    clone's zygosity prior shouldn't depend on which partition is being
    tested). Since the likelihood is exactly zero below h_min_k by
    construction, the two differ only by a normalising constant per
    unpinned clone:
        log Z_U[0,max_h] = log Z_grid + sum_k log(max_h - h_min_k)
    Pinned clones (size-1 grids, from h_fixed) carry no h integral and are
    excluded. Effect is exactly zero whenever every h_min_k = 0 (i.e. no
    mutation in the participant has VAF > 0.5). With two+ mutations above
    0.5 the residual is typically a few tenths of a log unit (see Linus
    notes) -- real, but usually not enough to flip a structure call.
    ------------------------------------------------------------------------
    """
    s_prior = 1.0 / (s_vec.max() - s_vec.min())
    log_g = log_trapz(log_out_grid, x=s_vec, axis=1) + jnp.log(s_prior)
    log_model_per_combo = jnp.sum(log_g, axis=1)
    log_model = logsumexp(log_model_per_combo) - jnp.log(log_model_per_combo.shape[0])

    prior_correction = 0.0
    for g in h_grids:
        if len(g) <= 1:
            continue
        width = float(g[-1] - g[0])
        if width > 0:
            prior_correction += float(np.log(width))
    return float(log_model + prior_correction)


def clone_posteriors(log_out_grid, h_combos, s_vec, h_grids):
    s_prior = 1.0 / (s_vec.max() - s_vec.min())
    log_g = np.array(log_trapz(log_out_grid, x=s_vec, axis=1) + jnp.log(s_prior))
    log_og = np.array(log_out_grid)
    K = log_g.shape[1]
    out = []
    for i in range(K):
        log_others = np.sum(np.delete(log_g, i, axis=1), axis=1)
        h_vals = np.array(h_grids[i]); hcol = np.array(h_combos[:, i])
        joint_log = np.full((len(h_vals), log_og.shape[1]), -np.inf)
        for hk, hv in enumerate(h_vals):
            m = np.isclose(hcol, hv)
            terms = log_og[m, :, i] + log_others[m][:, None]
            joint_log[hk] = np.array(logsumexp(terms, axis=0))
        mx = np.nanmax(joint_log)
        joint = np.exp(joint_log - mx) if np.isfinite(mx) else np.zeros_like(joint_log)
        out.append((h_vals, joint))
    return out


def compute_clonal_models_prob_vec(part, s_resolution=20, h_resolution=4,
                                   min_s=0.01, max_s=3.0, max_h=1.0,
                                   filter_invalid=True, disable_progressbar=False,
                                   beta_resolution=BETA_RESOLUTION):
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)
    n_mutations = part.shape[0]
    part.uns['model_dict'] = {}

    # Bounds depend only on AO/DP -- once per participant
    beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed)

    cs_list = find_valid_clonal_structures(part, filter_invalid=filter_invalid)

    part.uns['warning'] = None
    if len(cs_list) > 100:
        part.uns['warning'] = 'Too many possible structures'
        cs_list = [[[i] for i in range(n_mutations)]]

    for i, cs in tqdm(enumerate(cs_list), disable=disable_progressbar,
                      total=len(cs_list)):
        h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed,
                                      h_resolution, max_h)
        n_combo = int(np.prod([len(g) for g in h_grids]))
        if n_combo > 5000:
            print(f"  note: model {i} -> {n_combo} h-combos "
                  f"({len(cs)} clones x h_res={h_resolution}); may be slow")
        out_grid, _ = compute_cs_posterior_grid_vec(
            s_vec, h_grids, AO, DP, time_points, cs, observed, carry_idx,
            beta_lo, beta_hi, resolution=beta_resolution)
        model_prob = compute_model_likelihood(out_grid, cs, s_vec, h_grids)
        part.uns['model_dict'][f'model_{i}'] = (cs, model_prob)

    part.uns['model_dict'] = {k: v for k, v in sorted(
        part.uns['model_dict'].items(), key=lambda kv: kv[1][1], reverse=True)}
    return part


def refine_optimal_model_posterior_vec(part, s_resolution=40, h_resolution=6,
                                       min_s=0.01, max_s=3.0, max_h=1.0,
                                       beta_resolution=BETA_RESOLUTION):
    cs = list(part.uns['model_dict'].values())[0][0]
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)
    beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed)

    h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed,
                                  h_resolution, max_h)
    out_grid, h_combos = compute_cs_posterior_grid_vec(
        s_vec, h_grids, AO, DP, time_points, cs, observed, carry_idx,
        beta_lo, beta_hi, resolution=beta_resolution)

    part.uns['optimal_model'] = {
        'clonal_structure': cs,
        'mutation_structure': [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
        'posterior': out_grid,
        'h_combos': h_combos,
        's_range': s_vec,
        'h_grids': [np.array(g) for g in h_grids]}

    posteriors = clone_posteriors(out_grid, h_combos, s_vec, h_grids)

    n = part.shape[0]
    fitness        = np.zeros(n); fitness_5       = np.zeros(n); fitness_95       = np.zeros(n)
    homozygosity   = np.zeros(n); homozygosity_5  = np.zeros(n); homozygosity_95  = np.zeros(n)
    clonal_index   = np.zeros(n)
    fitness_railed            = np.zeros(n, dtype=bool)
    homozygosity_railed       = np.zeros(n, dtype=bool)
    homozygosity_unidentified = np.zeros(n, dtype=bool)

    s_top = float(np.array(s_vec).max())
    eps = 1e-9
    H_CI_WIDTH = 0.5
    rng = np.random.default_rng(0)   # reproducible CIs

    for i, c_idx in enumerate(cs):
        h_vals, joint = posteriors[i]
        joint = np.nan_to_num(np.asarray(joint), nan=0.0, posinf=0.0, neginf=0.0)

        p_s = joint.sum(axis=0)
        p_h = joint.sum(axis=1)
        if p_s.sum() <= 0 or p_h.sum() <= 0:
            part.uns['warning'] = 'Zero posterior'
            return part
        p_s = p_s / p_s.sum()
        p_h = p_h / p_h.sum()

        fitness_map = float(np.array(s_vec)[np.argmax(p_s)])
        h_map = float(h_vals[np.argmax(p_h)])

        s_cfd = np.quantile(
            rng.choice(np.array(s_vec), p=p_s, size=1_000), [0.05, 0.95])
        if len(h_vals) > 1:
            h_cfd = np.quantile(
                rng.choice(h_vals, p=p_h, size=1_000), [0.05, 0.95])
        else:
            h_cfd = [h_vals[0], h_vals[0]]

        s_railed  = fitness_map >= s_top - eps
        h_railed  = (len(h_vals) > 1) and (h_map >= max_h - eps)
        h_unident = (len(h_vals) > 1) and ((h_cfd[1] - h_cfd[0]) > H_CI_WIDTH)
        h_railed  = h_railed and not h_unident

        fitness[c_idx]        = fitness_map
        fitness_5[c_idx]      = s_cfd[0]; fitness_95[c_idx]      = s_cfd[1]
        homozygosity[c_idx]   = h_map
        homozygosity_5[c_idx] = h_cfd[0]; homozygosity_95[c_idx] = h_cfd[1]
        clonal_index[c_idx]   = i
        fitness_railed[c_idx]            = s_railed
        homozygosity_railed[c_idx]       = h_railed
        homozygosity_unidentified[c_idx] = h_unident

    part.obs['fitness'] = fitness
    part.obs['fitness_5'] = fitness_5
    part.obs['fitness_95'] = fitness_95
    part.obs['homozygosity'] = homozygosity
    part.obs['homozygosity_5'] = homozygosity_5
    part.obs['homozygosity_95'] = homozygosity_95
    part.obs['clonal_index'] = clonal_index
    part.obs['fitness_railed'] = fitness_railed
    part.obs['homozygosity_railed'] = homozygosity_railed
    part.obs['homozygosity_unidentified'] = homozygosity_unidentified

    mut_structure = part.uns['optimal_model']['mutation_structure']
    part.obs['clonal_structure'] = [
        next(s for s in mut_structure if mut in s) for mut in part.obs.index]
    return part


# ---------------------------------------------------------------------------
# Coordinate-refinement (avoids the h_resolution ** n_clones joint grid)
# ---------------------------------------------------------------------------
#
# refine_optimal_model_posterior_vec above builds the FULL joint grid over all
# clones' h simultaneously -- cost = s_resolution * h_resolution ** n_clones.
# That's fine for 1-2 clones but explodes for 3+ (e.g. h_resolution=16 with
# 3 clones = 4096 combos). The clones are coupled only through the shared
# wild-type cell pool (total_cells solves a system involving every clone's h
# at once), so we can't just refine each clone's h independently against a
# fixed background -- but we CAN do coordinate ascent: hold every other
# clone's h at its current best estimate, sweep this clone's h on a fine 1D
# grid, update, move to the next clone, and cycle a few times. Cost becomes
# ~ n_cycles * n_clones * h_resolution instead of h_resolution ** n_clones --
# linear in n_clones rather than exponential.
#
# This trades exactness for tractability: the final per-clone posterior is a
# PROFILE posterior (conditional on the other clones sitting at their point
# estimate), not a full marginal integrating over their uncertainty too. For
# 1-2 clones, prefer refine_optimal_model_posterior_vec (exact); reach for
# this once the joint grid stops being affordable.
#
# IMPORTANT (found empirically): the profile posterior can be MATERIALLY
# narrower than the true joint marginal, not just a coarser version of it --
# coordinate ascent's greedy, one-clone-at-a-time exploration can miss
# regions of real joint mass where several clones are simultaneously away
# from their individual optimum together. Do not substitute this for the
# full-joint method as a "smoothing" shortcut; verified on MDS711P64 that it
# can collapse a broad, genuinely-supported marginal into an artificially
# narrow spike for at least one clone. Prefer pushing full-joint h_resolution
# higher (memory/compute permitting) over switching methods for readability.
# ---------------------------------------------------------------------------

def _eval_h_vec(cs, AO, DP, time_points, observed, carry_idx, s_vec, h_vec,
                beta_lo, beta_hi):
    """Evaluate the full joint log-likelihood (summed s-marginal across all
    clones) for one specific h_vec. Returns (total_log_lik, per_clone_s_curve)
    or (None, None) if infeasible."""
    det, tot, _, h_mut = compute_deterministic_size(
        cs, AO, DP, AO.shape[1], h_vec, observed)
    if not bool(np.all(np.isfinite(np.array(tot)) & (np.array(tot) > 0))):
        return None, None
    out = jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs,
                            det, tot, h_mut, observed, carry_idx,
                            beta_lo, beta_hi)   # (s_res, K)
    s_prior = 1.0 / (float(s_vec.max()) - float(s_vec.min()))
    log_g = np.array(log_trapz(out, x=s_vec, axis=0)) + np.log(s_prior)  # (K,)
    return float(np.sum(log_g)), np.array(out)


def coordinate_refine_h(cs, AO, DP, time_points, observed, carry_idx,
                        s_vec, fine_grids, beta_lo, beta_hi, n_cycles=3, init_h=None):
    """Coordinate-ascent point estimate for each clone's h. Returns the
    converged h vector (numpy array, one value per clone)."""
    n_clones = len(cs)
    # Start every clone at max_h, NOT at each clone's individual floor. Individual
    # floors minimize each clone alone but the clones share one wild-type pool --
    # sitting at clone i's own floor can already exhaust the WHOLE shared budget,
    # making every single-coordinate move for other clones infeasible with no way
    # to climb out (coordinate ascent only ever moves one axis at a time). h=max_h
    # is the corner that MINIMIZES each clone's own budget usage (2v/(1+h) is
    # decreasing in h), so it's the safest feasible starting point.
    h_cur = (np.array(init_h, dtype=float) if init_h is not None
             else np.array([float(g[-1]) for g in fine_grids]))

    if n_clones == 1:
        n_cycles = 1  # nothing to coordinate against

    start_total, _ = _eval_h_vec(cs, AO, DP, time_points, observed, carry_idx,
                                 s_vec, jnp.array(h_cur), beta_lo, beta_hi)
    if start_total is None:
        print("  warning: structure infeasible even at h=max_h for every clone -- "
              "this clonal structure cannot jointly explain the data under this "
              "model regardless of zygosity; results below are unreliable.")

    for _cycle in range(n_cycles):
        moved = False
        for k in range(n_clones):
            grid_k = np.array(fine_grids[k])
            best_val, best_h = -np.inf, h_cur[k]
            for hk in grid_k:
                trial = h_cur.copy()
                trial[k] = hk
                total, _ = _eval_h_vec(cs, AO, DP, time_points, observed,
                                       carry_idx, s_vec, jnp.array(trial),
                                       beta_lo, beta_hi)
                if total is not None and total > best_val:
                    best_val, best_h = total, hk
            if not np.isclose(best_h, h_cur[k]):
                moved = True
            h_cur[k] = best_h
        if not moved:
            break   # converged early -- no clone moved this cycle
    return h_cur


def refine_optimal_model_posterior_coordinate(part, s_resolution=40, h_resolution=16,
                                              min_s=0.01, max_s=3.0, max_h=1.0,
                                              n_cycles=3,
                                              beta_resolution=BETA_RESOLUTION):
    """Multi-clone-friendly alternative to refine_optimal_model_posterior_vec.
    Cost ~ n_cycles * n_clones * h_resolution (+ one more such pass for the
    posterior sweep), instead of h_resolution ** n_clones. See module notes
    above for the profile-posterior caveat -- confirmed empirically to be
    more than cosmetic in at least one case."""
    cs = list(part.uns['model_dict'].values())[0][0]
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)
    n_clones = len(cs)
    beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed)

    fine_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, h_resolution, max_h)

    h_converged = coordinate_refine_h(cs, AO, DP, time_points, observed, carry_idx,
                                      s_vec, fine_grids, beta_lo, beta_hi,
                                      n_cycles=n_cycles)

    part.uns['optimal_model'] = {
        'clonal_structure': cs,
        'mutation_structure': [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
        's_range': s_vec,
        'h_converged': h_converged.copy(),
        'h_grids_fine': [np.array(g) for g in fine_grids],
        'method': 'coordinate_refinement',
    }

    n = part.shape[0]
    fitness        = np.zeros(n); fitness_5       = np.zeros(n); fitness_95       = np.zeros(n)
    homozygosity   = np.zeros(n); homozygosity_5  = np.zeros(n); homozygosity_95  = np.zeros(n)
    clonal_index   = np.zeros(n)
    fitness_railed            = np.zeros(n, dtype=bool)
    homozygosity_railed       = np.zeros(n, dtype=bool)
    homozygosity_unidentified = np.zeros(n, dtype=bool)

    s_top = float(np.array(s_vec).max())
    eps = 1e-9
    H_CI_WIDTH = 0.5
    per_clone_posteriors = []
    rng = np.random.default_rng(0)

    # Final per-clone posterior sweep: hold every OTHER clone at its converged
    # h, sweep this clone's own fine grid, and read off its (s, h) profile.
    for k, c_idx in enumerate(cs):
        grid_k = np.array(fine_grids[k])
        joint_log = np.full((len(grid_k), s_resolution), -np.inf)
        for gi, hk in enumerate(grid_k):
            trial = h_converged.copy()
            trial[k] = hk
            _, out = _eval_h_vec(cs, AO, DP, time_points, observed, carry_idx,
                                 s_vec, jnp.array(trial), beta_lo, beta_hi)
            if out is not None:
                joint_log[gi] = out[:, k]
        mx = np.nanmax(joint_log)
        joint = np.exp(joint_log - mx) if np.isfinite(mx) else np.zeros_like(joint_log)
        per_clone_posteriors.append((grid_k, joint))

        joint_nn = np.nan_to_num(joint, nan=0.0, posinf=0.0, neginf=0.0)
        p_s = joint_nn.sum(axis=0)
        p_h = joint_nn.sum(axis=1)
        if p_s.sum() <= 0 or p_h.sum() <= 0:
            part.uns['warning'] = 'Zero posterior'
            continue
        p_s = p_s / p_s.sum()
        p_h = p_h / p_h.sum()

        fitness_map = float(np.array(s_vec)[np.argmax(p_s)])
        h_map = float(grid_k[np.argmax(p_h)])

        s_cfd = np.quantile(
            rng.choice(np.array(s_vec), p=p_s, size=1_000), [0.05, 0.95])
        if len(grid_k) > 1:
            h_cfd = np.quantile(
                rng.choice(grid_k, p=p_h, size=1_000), [0.05, 0.95])
        else:
            h_cfd = [grid_k[0], grid_k[0]]

        s_railed  = fitness_map >= s_top - eps
        h_railed  = (len(grid_k) > 1) and (h_map >= max_h - eps)
        h_unident = (len(grid_k) > 1) and ((h_cfd[1] - h_cfd[0]) > H_CI_WIDTH)
        h_railed  = h_railed and not h_unident

        fitness[c_idx]        = fitness_map
        fitness_5[c_idx]      = s_cfd[0]; fitness_95[c_idx]      = s_cfd[1]
        homozygosity[c_idx]   = h_map
        homozygosity_5[c_idx] = h_cfd[0]; homozygosity_95[c_idx] = h_cfd[1]
        clonal_index[c_idx]   = k
        fitness_railed[c_idx]            = s_railed
        homozygosity_railed[c_idx]       = h_railed
        homozygosity_unidentified[c_idx] = h_unident

    part.obs['fitness'] = fitness
    part.obs['fitness_5'] = fitness_5
    part.obs['fitness_95'] = fitness_95
    part.obs['homozygosity'] = homozygosity
    part.obs['homozygosity_5'] = homozygosity_5
    part.obs['homozygosity_95'] = homozygosity_95
    part.obs['clonal_index'] = clonal_index
    part.obs['fitness_railed'] = fitness_railed
    part.obs['homozygosity_railed'] = homozygosity_railed
    part.obs['homozygosity_unidentified'] = homozygosity_unidentified

    part.uns['optimal_model']['posteriors'] = per_clone_posteriors

    mut_structure = part.uns['optimal_model']['mutation_structure']
    part.obs['clonal_structure'] = [
        next(s for s in mut_structure if mut in s) for mut in part.obs.index]
    return part


def compute_invalid_combinations(part, pearson_distance_threshold=0.5):
    """Flag mutation pairs that shouldn't share a clone -- either because
    their VAF-vs-time correlations differ too much (existing check), OR
    because they have mutually incompatible fixed zygosity (prevents the
    structure search from ever proposing a clone that would later crash
    build_clone_h_grids' consistency assert)."""
    DP = np.array(part.layers['DP'])              # (n_mut, n_tp)
    AO = np.array(part.layers['AO'])
    observed = DP > 0
    vaf = np.where(observed, AO / np.where(DP > 0, DP, 1.0), np.nan)
    vaf_filled = np.vstack([_fill_nearest(vaf[i], observed[i])
                            for i in range(vaf.shape[0])])
    tp = np.array(part.var['time_points'], dtype=float)

    corr = np.corrcoef(np.vstack([vaf_filled, tp]))
    corr_vec = corr[-1, :-1]
    dist = np.abs(corr_vec - corr_vec[:, None])

    res = []
    for i, j in np.argwhere(dist > pearson_distance_threshold):
        pair = sorted([int(i), int(j)])
        if pair not in res:
            res.append(pair)

    if 'h_fixed' in part.obs.columns:
        h_fixed = np.array(part.obs['h_fixed'].values, dtype=float)
        n = len(h_fixed)
        for i in range(n):
            for j in range(i + 1, n):
                hi, hj = h_fixed[i], h_fixed[j]
                if not (np.isnan(hi) or np.isnan(hj)) and not np.isclose(hi, hj):
                    pair = sorted([i, j])
                    if pair not in res:
                        res.append(pair)

    part.uns['invalid_combinations'] = res


def find_valid_clonal_structures(part, p_distance_threshold=1, filter_invalid=True):
    n_mutations = part.shape[0]
    if n_mutations == 1:
        return [[[0]]]

    if filter_invalid:
        compute_invalid_combinations(part, pearson_distance_threshold=p_distance_threshold)

    cs_list = [cs for cs in partition(list(range(n_mutations)))]
    if not filter_invalid:
        return cs_list

    valid_cs = []
    for cs in cs_list:
        bad = 0
        for clone in cs:
            bad += sum(1 for comb in combinations(clone, 2)
                       if sorted(comb) in part.uns['invalid_combinations'])
        if bad == 0:
            valid_cs.append(cs)
    return valid_cs


from scipy.stats import nbinom as _sp_nbinom

# np.trapezoid was only added in NumPy 2.0 -- older installs (e.g. this repo's
# venv) only have np.trapz, which has the identical signature/behaviour for
# our use here. Resolve once at import time rather than per-call.
_np_trapz = getattr(np, "trapezoid", None) or np.trapz


def _log_trapz_np(log_y, x, axis=-1):
    m = np.max(log_y, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    val = _np_trapz(np.exp(log_y - m), x=x, axis=axis)
    return np.squeeze(m, axis=axis) + np.log(val)

def _nbinom_logpmf_continuous(k, n, p):
    """Continuous extension of the negative-binomial logpmf, matching
    jax.scipy.stats.nbinom.logpmf's gamma-function formula exactly.
    scipy.stats.nbinom.logpmf enforces integer support and returns -inf
    for any non-integer k -- which silently collapsed the ENTIRE twin to
    -inf everywhere, since x is continuous by construction under Fix 3.
    This is the same formula JAX uses internally, so JAX and this twin
    now evaluate the identical function at every node, integer or not."""
    return (_sp_gammaln(k + n) - _sp_gammaln(n) - _sp_gammaln(k + 1.0)
            + n * np.log(p) + k * np.log1p(-p))


def jax_cs_hmm_ll_ref(s_vec, AO, DP, time_points, cs,
                      deterministic_size, total_cells, h_mut,
                      observed, carry_idx, beta_lo=None, beta_hi=None,
                      resolution=BETA_RESOLUTION):
    """NumPy twin of jax_cs_hmm_ll_vec for validate_participant.

    RECONCILED (see _nbinom_logpmf_continuous docstring for the NB fix).
    Also ports the Fix-4 seeding term (explicit Beta-prior-density +
    Jacobian at the first-observed timepoint) in place of the old
    `-np.log(resolution)` shortcut, which no longer matches compute_
    global_variables since Fix 3 dropped the property (true_vaf == beta
    exactly) that shortcut relied on. Both changes are required for this
    to be a valid reference -- fixing only one still leaves a stale twin.
    """
    AO = np.array(AO); DP = np.array(DP)
    observed = np.array(observed, dtype=bool)
    carry_idx = np.array(carry_idx)
    time_points = np.array(time_points, dtype=float)
    deterministic_size = np.array(deterministic_size)
    total_cells = np.array(total_cells)
    h_mut = np.array(h_mut); s_vec = np.array(s_vec)
    n_tp, n_mut = AO.shape
    lamb = 1.3

    DP_safe = np.where(observed, DP, 1.0)
    AO_safe = np.where(observed, AO, 0.0)

    if beta_lo is None or beta_hi is None:
        beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed)
    beta_lo = np.array(beta_lo); beta_hi = np.array(beta_hi)

    N_w_cond = (total_cells[:, None] - deterministic_size)[:, :, None]
    half = (1.0 + h_mut)[None, :, None] / 2.0

    beta_lo_ = beta_lo[:, :, None]
    beta_hi_ = np.minimum(beta_hi[:, :, None], half * (1.0 - 1e-9))
    x_lo = -N_w_cond * beta_lo_ / (beta_lo_ - half)
    x_hi = -N_w_cond * beta_hi_ / (beta_hi_ - half)
    lo = np.minimum(x_lo, x_hi)
    hi = np.maximum(x_lo, x_hi)

    frac = np.linspace(0.0, 1.0, resolution).reshape(1, 1, resolution)
    x = lo + frac * (hi - lo)
    true_vaf = (1.0 + h_mut)[None, :, None] * x / (2.0 * (N_w_cond + x))

    log_p_y = np.array(beta_binom_logpmf(
        jnp.array(AO_safe[:, :, None]),
        jnp.array(DP_safe[:, :, None]),
        jnp.array(true_vaf), 200.))

    # carry-index BEFORE the seeding term, same order as compute_global_variables
    x = np.take_along_axis(x, np.broadcast_to(carry_idx[:, :, None], x.shape), axis=0)
    log_p_y = np.where(observed[:, :, None], log_p_y, 0.0)

    # ------------------------------------------------------------------
    # ported Fix-4 seeding term -- must match compute_global_variables
    # exactly, or the twin validates against a different model than the
    # one actually running.
    # ------------------------------------------------------------------
    idx0 = carry_idx[0, :]                                        # (n_mut,)
    AO0 = AO_safe[idx0, np.arange(n_mut)]                          # (n_mut,)
    DP0 = DP_safe[idx0, np.arange(n_mut)]                          # (n_mut,)
    a0, b0 = AO0 + 1.0, DP0 - AO0 + 1.0

    N_w_cond_0 = N_w_cond[idx0, np.arange(n_mut), :]               # (n_mut, 1)
    half_0 = half[0]                                               # (n_mut, 1)
    x0 = x[0, :, :]                                                # already carry-indexed
    beta0 = (2.0 * half_0) * x0 / (2.0 * (N_w_cond_0 + x0))        # == beta identically

    log_jacobian = np.log(half_0) + np.log(N_w_cond_0) - 2.0 * np.log(N_w_cond_0 + x0)
    prior_logpdf = ((a0[:, None] - 1.0) * np.log(beta0)
                    + (b0[:, None] - 1.0) * np.log1p(-beta0)
                    - _sp_betaln(a0[:, None], b0[:, None]))

    delta_t = np.diff(time_points)
    out = np.zeros((len(s_vec), len(cs)))
    for si, s in enumerate(s_vec):
        exp_term = np.exp(delta_t * s)
        mut_ll = np.zeros(n_mut)
        for m in range(n_mut):
            x_m = x[:, m, :]; lpy_m = log_p_y[:, m, :]
            mean = x_m[:-1] * exp_term[:, None]
            var  = x_m[:-1] * (2 * lamb + s) * exp_term[:, None] * (exp_term[:, None] - 1) / s
            p_nb = mean / var
            n_nb = mean**2 / (var - mean)

            log_rec = lpy_m[0] + prior_logpdf[m] + log_jacobian[m]   # Fix-4 seeding, not -log(resolution)

            for j in range(1, n_tp):
                log_bd = _nbinom_logpmf_continuous(x_m[j][:, None], n_nb[j - 1], p_nb[j - 1])
                log_inner = log_bd + log_rec[None, :]
                log_rec = lpy_m[j] + _log_trapz_np(log_inner, x_m[j - 1], axis=-1)
            mut_ll[m] = _log_trapz_np(log_rec, x_m[-1])
        for ci, c_idx in enumerate(cs):
            out[si, ci] = np.sum(mut_ll[np.array(c_idx)])
    return out

def describe_structure(part, cs, h=None):
    lines = []
    for ci, c_idx in enumerate(cs):
        labels = []
        for j in c_idx:
            row = part.obs.iloc[j]
            gene = row.get('GENE', '')
            prot = row.get('PROTEIN_CHANGE', '')
            labels.append(f"{gene} {prot}".strip())
        htxt = f"  (h={float(np.array(h)[ci]):.2f})" if h is not None else ""
        lines.append(f"    clone {ci}{htxt}: " + " | ".join(labels))
    return "\n".join(lines)


def validate_participant(part, s_resolution=15, h_resolution=3,
                         min_s=0.01, max_s=3.0, max_h=1.0, rtol=1e-3,
                         beta_resolution=BETA_RESOLUTION):
    """Diff jax_cs_hmm_ll_vec against its NumPy twin (same Fix-3 nodes).

    Finiteness is treated as part of the comparison, not a pre-filter:
      * both finite            -> compared with allclose
      * both non-finite (-inf) -> agree (both underflowed identically)
      * finiteness DISAGREES   -> counted as a mismatch, NOT silently dropped
    The old version masked on np.isfinite(ref) only, so any position where
    ref=-inf but vec is finite (or vice versa) was excluded from allclose
    entirely -- which is how a genuine disagreement could sit next to a PASS.
    """
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)
    beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed)

    cs = find_valid_clonal_structures(part, filter_invalid=True)[0]
    h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, h_resolution, max_h)

    h = det = tot = h_mut = vec = None
    for idx in itertools.product(*[range(len(g)) for g in h_grids]):
        cand = jnp.array([h_grids[k][idx[k]] for k in range(len(cs))])
        d, t, _, hm = compute_deterministic_size(cs, AO, DP, AO.shape[1], cand, observed)
        if not bool(jnp.all(jnp.isfinite(t) & (t > 0))):
            continue
        trial = np.array(jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs,
                                           d, t, hm, observed, carry_idx,
                                           beta_lo, beta_hi,
                                           resolution=beta_resolution))
        if np.any(np.isfinite(trial)):
            h, det, tot, h_mut, vec = cand, d, t, hm, trial
            break
    if h is None:
        print("no feasible / finite h-combo found for this structure "
              "(structure is degenerate -> would score ~0 and never be selected).")
        return None

    ref = jax_cs_hmm_ll_ref(s_vec, AO, DP, time_points, cs,
                            det, tot, h_mut, observed, carry_idx,
                            beta_lo, beta_hi, resolution=beta_resolution)

    vec = np.asarray(vec); ref = np.asarray(ref)

    # --- finiteness bookkeeping ------------------------------------------
    fin_v = np.isfinite(vec)
    fin_r = np.isfinite(ref)
    both_finite    = fin_v & fin_r
    both_infinite  = (~fin_v) & (~fin_r)
    finiteness_mismatch = fin_v ^ fin_r          # XOR: exactly one is finite
    n_mismatch = int(finiteness_mismatch.sum())

    # --- numeric agreement on the jointly-finite entries -----------------
    if both_finite.any():
        abs_diff = np.abs(vec[both_finite] - ref[both_finite])
        rel_diff = abs_diff / np.maximum(np.abs(ref[both_finite]), 1e-300)
        max_abs = float(abs_diff.max())
        max_rel = float(rel_diff.max())
        numeric_ok = np.allclose(vec[both_finite], ref[both_finite],
                                 rtol=rtol, atol=1e-6)
    else:
        max_abs = max_rel = np.nan
        numeric_ok = False

    argmax_ok = np.array_equal(vec.argmax(0), ref.argmax(0))

    print(f"participant       : {part.uns.get('participant_id')}")
    print(f"structure         :\n{describe_structure(part, cs, h)}")
    print(f"grid shape        : {vec.shape}  (s_resolution x n_clones)")
    print(f"both finite       : {int(both_finite.sum())} / {vec.size}")
    print(f"both -inf         : {int(both_infinite.sum())} / {vec.size}")
    print(f"finiteness MISMATCH: {n_mismatch} / {vec.size}"
          + ("   <-- REAL DISAGREEMENT" if n_mismatch else ""))
    print(f"max abs diff      : {max_abs:.3e}   (over jointly-finite entries)")
    print(f"max rel diff      : {max_rel:.3e}")
    print(f"argmax-s agree    : {argmax_ok}")

    if n_mismatch:
        # show a few offenders so you can see WHERE they diverge
        rows, cols = np.where(finiteness_mismatch)
        print("  first finiteness mismatches (s_idx, clone, vec, ref):")
        for r, c in list(zip(rows, cols))[:5]:
            print(f"    ({r:>3}, {c}):  vec={vec[r, c]:+.4e}   ref={ref[r, c]:+.4e}")

    ok = numeric_ok and (n_mismatch == 0) and argmax_ok
    print("PASS" if ok else "FAIL")
    return vec, ref



def check_qeps_stability(part, q_eps_list=(1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
                         beta_resolution=4_000, s_resolution=40,
                         h_resolution=3, min_s=0.01, max_s=3.0, max_h=1.0,
                         log_lik_window=20.0):
    """Repurposed for Fix 3: q_eps here only sets how far out the domain
    EDGES sit (compute_beta_bounds), not interior node placement -- so
    unlike the old scheme, this should now show LOW sensitivity to q_eps
    at fixed resolution (the edges moving slightly shouldn't matter much
    once nodes are uniform in x). A large Δp(s) here would mean the domain
    edges themselves are cutting off non-negligible posterior mass, which
    is a different problem (widen q_eps) from the old truncation-error
    story. Compares p(s), not log p(s), for the same reason as before:
    implausible s near max_s can dominate a raw log-space diff without
    reflecting any real probability-mass difference.
    """
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)
    s_arr = np.array(s_vec)
    cs = (list(part.uns['model_dict'].values())[0][0] if 'model_dict' in part.uns
          else find_valid_clonal_structures(part, filter_invalid=True)[0])
    h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, h_resolution, max_h)
    h0 = jnp.array([g[-1] for g in h_grids])
    det, tot, _, h_mut = compute_deterministic_size(cs, AO, DP, AO.shape[1], h0, observed)

    print(f"structure: {cs}  |  s_resolution={s_resolution}  |  "
          f"beta_resolution={beta_resolution} (fixed)  |  "
          f"log_lik_window={log_lik_window}")
    header = (f"{'q_eps':>10}  {'Δp(s) max':>10}  {'in-window':>10}  "
              f"MAP-s per clone")
    print(header)
    print("-" * 80)

    results = {}
    prev_p = None
    for q_eps in q_eps_list:
        beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed, q_eps=q_eps)
        out = np.array(jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs, det, tot,
                                         h_mut, observed, carry_idx,
                                         beta_lo, beta_hi,
                                         resolution=beta_resolution))
        results[q_eps] = out

        # per-clone normalised probability, NOT raw log-lik
        p = np.exp(out - np.nanmax(out, axis=0, keepdims=True))
        p = p / np.nansum(p, axis=0, keepdims=True)

        map_s = [float(s_arr[np.argmax(out[:, k])]) for k in range(out.shape[1])]
        in_window = np.mean(out >= (np.nanmax(out, axis=0, keepdims=True) - log_lik_window))

        if prev_p is None:
            d_p = np.nan
        else:
            d_p = float(np.nanmax(np.abs(p - prev_p)))

        print(f"{q_eps:>10.0e}  {d_p:>10.3e}  {in_window*100:>9.1f}%  "
              + "  ".join(f"{s:.3f}" for s in map_s))
        prev_p = p

    print("-" * 80)
    print("Under Fix 3, Δp(s) here reflects domain-EDGE sensitivity only --")
    print("should be small and roughly flat across q_eps. Use")
    print("check_node_stability (resolution sweep) as the primary")
    print("convergence check now; this test no longer targets interior")
    print("quadrature error the way it did under the old node scheme.")
    return results


def plot_qeps_stability(results, out_png, pid=""):
    import matplotlib.pyplot as plt
    n_clones = next(iter(results.values())).shape[1]
    fig, axes = plt.subplots(1, n_clones, figsize=(5 * n_clones, 4), squeeze=False)
    axes = axes[0]
    for q_eps, out in results.items():
        for k in range(n_clones):
            p_shifted = out[:, k] - np.nanmax(out[:, k])
            axes[k].plot(np.exp(p_shifted), label=f"q_eps={q_eps:.0e}")
    for k, ax in enumerate(axes):
        ax.set_title(f"clone {k}")
        ax.set_xlabel("s index")
        ax.set_ylabel("normalised p(s)")
        ax.legend(fontsize=7)
    fig.suptitle(f"{pid}: p(s) vs q_eps (beta_resolution fixed)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"saved overlay -> {out_png}")


def check_node_magnitudes(part, q_eps_list=(1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
                          h_resolution=3, max_h=1.0):
    """Cheap diagnostic: just the domain-edge values (x_lo, x_hi), no
    likelihood/transition computation. Under Fix 3 this checks how much the
    edges themselves move with q_eps, and how they compare to total_cells
    near the h=max_h pole -- NOT node crowding, since nodes are now uniform
    in x between the edges by construction."""
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    cs = (list(part.uns['model_dict'].values())[0][0] if 'model_dict' in part.uns
          else find_valid_clonal_structures(part, filter_invalid=True)[0])
    h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, h_resolution, max_h)
    h0 = jnp.array([g[-1] for g in h_grids])   # h = max_h, same edge as the real runs
    det, tot, _, h_mut = compute_deterministic_size(cs, AO, DP, AO.shape[1], h0, observed)

    N_w_cond = (np.array(tot)[:, None] - np.array(det))[:, :, None]
    half = (1.0 + np.array(h_mut))[None, :, None] / 2.0
    total_cells_arr = np.array(tot)

    print(f"{'q_eps':>10}  {'max x_hi':>14}  {'max total_cells':>16}  {'ratio':>10}")
    for q_eps in q_eps_list:
        beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed, q_eps=q_eps)
        beta_hi_ = np.minimum(np.array(beta_hi)[:, :, None], half * (1.0 - 1e-9))
        x_hi = -N_w_cond * beta_hi_ / (beta_hi_ - half)
        max_x = float(np.max(x_hi))
        max_tc = float(np.max(total_cells_arr))
        print(f"{q_eps:>10.0e}  {max_x:>14.3e}  {max_tc:>16.3e}  {max_x/max_tc:>10.2e}")


def check_node_stability(part, n_resolutions=(500, 1_000, 2_000, 5_000),
                         s_resolution=15, h_resolution=3, min_s=0.01,
                         max_s=3.0, max_h=1.0, log_lik_window=20.0):
    """PRIMARY Fix-3/Fix-4 convergence check: LL should converge smoothly as
    beta resolution grows, since nodes are now uniform in x (spacing
    shrinks linearly with resolution by construction) and the seeding
    term's prior density is explicit (Fix 4) rather than a
    resolution-dependent artifact.

    Reports BOTH a raw Δmax (nanmax|diff| over the whole array) and a
    WINDOWED Δmax restricted to entries within log_lik_window of their own
    column's max. The raw metric is dominated by implausible high-s
    entries (exp(delta_t*s) astronomically large, ~zero actual probability
    weight) that swing by O(1) regardless of resolution -- confirmed via
    windowed_convergence_check.py, where raw Δmax sat flat around 2.78
    while windowed Δmax shrank cleanly ~4x per resolution doubling (true
    quadratic trapezoid convergence). Use the WINDOWED column to judge
    convergence; a flat raw Δmax alongside a shrinking windowed Δmax is
    expected and not a sign of a bug."""
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)
    cs = find_valid_clonal_structures(part, filter_invalid=True)[0]
    h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, h_resolution, max_h)
    h0 = jnp.array([g[-1] for g in h_grids])   # any fixed, feasible h-combo
    det, tot, _, h_mut = compute_deterministic_size(cs, AO, DP, AO.shape[1], h0, observed)
    beta_lo, beta_hi = compute_beta_bounds(AO, DP, observed)

    results = {}
    prev, prev_window = None, None
    for res in n_resolutions:
        out = np.array(jax_cs_hmm_ll_vec(
            s_vec, AO, DP, time_points, cs, det, tot, h_mut, observed,
            carry_idx, beta_lo, beta_hi, resolution=res))
        results[res] = out
        in_window = out >= (np.nanmax(out, axis=0, keepdims=True) - log_lik_window)
        msg = f"resolution={res:>5d}  max ll={np.nanmax(out):.6f}"
        if prev is not None:
            dmax_all = np.nanmax(np.abs(out - prev))
            mask = in_window & prev_window
            dmax_win = np.nanmax(np.abs(out[mask] - prev[mask])) if mask.any() else np.nan
            msg += f"  Δmax(all)={dmax_all:.3e}  Δmax(in-window)={dmax_win:.3e}"
        print(msg)
        prev, prev_window = out, in_window
    return results