"""
Sub-mutation fitness inference
==============================
For each non-leading mutation within a clone, infers:

  s_sub   : fitness of the sub-mutation relative to the clone's BD background
             Fitted by binomial likelihood on VAF_sub / VAF_lead, using the
             clone's total inferred cell count (from the BD model) as population size.

  s_rel   : s_sub - s_lead  (relative fitness within clone)
             Positive → sub-mutation expanding faster than clone founder
             Negative → sub-mutation lagging behind clone founder
             Zero     → leading mutation by definition

Results stored:
  part.obs columns : sub_fitness, sub_fitness_5, sub_fitness_95, relative_fitness
  part.uns['sub_mutation_inference'] : full posteriors, s_ranges, clone metadata

Designed to be called after refine_optimal_model_posterior_vec has been run,
so that part.uns['optimal_model'] and part.obs['fitness'] already exist.
"""

import numpy as np
from scipy.stats import binom as scipy_binom


# ==============================================================================
# Core inference
# ==============================================================================

def infer_sub_mutation_fitness(
        part,
        s_resolution=100,
        window_multiplier=2.0,
        window_min=0.3,
        min_s=0.01,
        max_s=3.0,
):
    """
    Infer sub-mutation fitness for every non-leading mutation in each clone.

    Parameters
    ----------
    part : AnnData
        Must have part.uns['optimal_model'] and part.obs['fitness'] already set
        by refine_optimal_model_posterior_vec.
    s_resolution : int
        Number of grid points for the s search.
    window_multiplier : float
        Search window half-width = window_multiplier * s_lead.
        e.g. 2.0 means search over [s_lead/2, s_lead*2] (clamped to [min_s, max_s]).
    window_min : float
        Minimum half-width of the search window regardless of s_lead.
    min_s, max_s : float
        Hard bounds on the s search range.

    Returns
    -------
    part : AnnData
        With new obs columns and part.uns['sub_mutation_inference'].
    """

    print("\n" + "=" * 70)
    print("SUB-MUTATION FITNESS INFERENCE")
    print("=" * 70)

    # ── Validate prerequisites ────────────────────────────────────────────────
    if 'optimal_model' not in part.uns:
        raise ValueError(
            "part.uns['optimal_model'] not found. "
            "Run refine_optimal_model_posterior_vec first."
        )
    if 'fitness' not in part.obs.columns:
        raise ValueError(
            "part.obs['fitness'] not found. "
            "Run refine_optimal_model_posterior_vec first."
        )

    optimal = part.uns['optimal_model']
    cs       = optimal['clonal_structure']        # list of lists of mut indices
    joint_results = optimal['joint_inference']    # one dict per clone

    AO = part.layers['AO']   # (n_mut, n_tp)
    DP = part.layers['DP']
    time_points = part.var.time_points.values.astype(float)
    obs_index   = list(part.obs.index)
    n_mut       = part.shape[0]

    # ── Output containers ─────────────────────────────────────────────────────
    sub_fitness     = np.full(n_mut, np.nan)
    sub_fitness_5   = np.full(n_mut, np.nan)
    sub_fitness_95  = np.full(n_mut, np.nan)
    relative_fitness = np.zeros(n_mut)   # 0 by definition for leading mutations

    uns_results = {}   # keyed by mutation name

    # ── Per-clone inference ───────────────────────────────────────────────────
    for clone_idx, clone_muts in enumerate(cs):
        print(f"\nClone {clone_idx}: mutations {clone_muts}")
        print("-" * 70)

        clone_result = joint_results[clone_idx]
        s_lead       = float(clone_result['s_map'])

        # Identify leading mutation (highest mean VAF)
        vaf_matrix  = AO[clone_muts] / np.maximum(DP[clone_muts], 1.0)
        mean_vafs   = vaf_matrix.mean(axis=1)
        lead_local  = int(np.argmax(mean_vafs))   # index within clone_muts
        lead_mut    = clone_muts[lead_local]

        print(f"  Leading mutation : [{lead_mut}] {obs_index[lead_mut]}")
        print(f"  s_lead           : {s_lead:.4f}")

        # VAF of leading mutation across timepoints
        vaf_lead = (AO[lead_mut] / np.maximum(DP[lead_mut], 1.0)).astype(float)

        # Inferred total clone cell count from BD model
        # joint_results stores the VAF trajectory; back-calculate N_clone
        # N_clone(t) ≈ VAF_lead(t) * 2 * N_w   (heterozygous approximation)
        N_w = 1e5
        N_clone = vaf_lead * 2.0 * N_w   # (n_tp,)
        N_clone = np.maximum(N_clone, 1.0)

        # ── Leading mutation: s_relative = 0 by definition ───────────────────
        sub_fitness[lead_mut]    = s_lead
        sub_fitness_5[lead_mut]  = float(clone_result['s_ci'][0])
        sub_fitness_95[lead_mut] = float(clone_result['s_ci'][1])
        relative_fitness[lead_mut] = 0.0

        uns_results[obs_index[lead_mut]] = {
            'role':           'leading',
            'clone_idx':      clone_idx,
            's_sub':          s_lead,
            's_rel':          0.0,
            'posterior':      clone_result['s_posterior'],
            's_range':        clone_result['s_range'],
            'ci_90':          clone_result['s_ci'],
        }

        print(f"  → Leading: s_sub={s_lead:.4f}, s_rel=0.000 (reference)")

        # ── Non-leading mutations ─────────────────────────────────────────────
        for mut_idx in clone_muts:
            if mut_idx == lead_mut:
                continue

            mut_name = obs_index[mut_idx]
            print(f"\n  Sub-mutation [{mut_idx}] {mut_name}")

            ao_sub = AO[mut_idx].astype(float)
            dp_sub = np.maximum(DP[mut_idx].astype(float), 1.0)

            # Effective sub-VAF = AO_sub / (DP_sub * VAF_lead)
            # This is the fraction of clone cells carrying the sub-mutation
            # Clamp to valid probability range
            vaf_sub_effective = ao_sub / np.maximum(dp_sub * vaf_lead, 1.0)
            vaf_sub_effective = np.clip(vaf_sub_effective, 1e-8, 1.0 - 1e-8)

            # Search window centred on s_lead
            half_w  = max(window_multiplier * s_lead, window_min)
            s_lo    = max(min_s, s_lead - half_w)
            s_hi    = min(max_s, s_lead + half_w)
            s_range = np.linspace(s_lo, s_hi, s_resolution)

            print(f"    s search range : [{s_lo:.3f}, {s_hi:.3f}]")

            # ── Initial sub-clone size estimate ───────────────────────────────
            # Use first timepoint VAF to seed N_sub_0
            N_sub_0 = vaf_sub_effective[0] * N_clone[0]
            N_sub_0 = max(N_sub_0, 10.0)

            # ── Binomial log-likelihood over s_range ──────────────────────────
            log_lik = np.zeros(s_resolution)

            for s_i, s_sub in enumerate(s_range):
                ll = 0.0
                N_sub = N_sub_0

                for tp_i, t in enumerate(time_points):
                    if tp_i > 0:
                        dt    = time_points[tp_i] - time_points[tp_i - 1]
                        N_sub = N_sub * np.exp(s_sub * dt)

                    # Sub-mutation can't exceed clone size
                    N_sub = min(N_sub, N_clone[tp_i] * 0.999)

                    # Expected fraction of clone cells with sub-mutation
                    p_sub = N_sub / N_clone[tp_i]
                    p_sub = np.clip(p_sub, 1e-8, 1.0 - 1e-8)

                    # Observed counts: AO_sub out of DP_sub, expected VAF = p_sub * VAF_lead
                    expected_vaf = p_sub * vaf_lead[tp_i]
                    expected_vaf = np.clip(expected_vaf, 1e-8, 1.0 - 1e-8)

                    ao  = int(ao_sub[tp_i])
                    dp  = int(dp_sub[tp_i])
                    ll += scipy_binom.logpmf(ao, dp, expected_vaf)

                log_lik[s_i] = ll

            # ── Posterior ─────────────────────────────────────────────────────
            log_lik  -= log_lik.max()   # numerical stability
            posterior = np.exp(np.clip(log_lik, -700, 0))
            posterior /= posterior.sum() + 1e-300

            # MAP
            s_map = s_range[np.argmax(posterior)]

            # 90% credible interval
            cumsum  = np.cumsum(posterior)
            ci_low  = s_range[np.searchsorted(cumsum, 0.05)]
            ci_high = s_range[np.searchsorted(cumsum, 0.95)]

            # Relative fitness
            s_rel = s_map - s_lead

            print(f"    s_sub  = {s_map:.4f}  [90% CI: {ci_low:.4f} – {ci_high:.4f}]")
            print(f"    s_rel  = {s_rel:+.4f}  (vs s_lead={s_lead:.4f})")

            # Store scalars
            sub_fitness[mut_idx]    = s_map
            sub_fitness_5[mut_idx]  = ci_low
            sub_fitness_95[mut_idx] = ci_high
            relative_fitness[mut_idx] = s_rel

            uns_results[mut_name] = {
                'role':      'sub',
                'clone_idx': clone_idx,
                'lead_mut':  obs_index[lead_mut],
                's_sub':     s_map,
                's_rel':     s_rel,
                's_lead':    s_lead,
                'posterior': posterior,
                's_range':   s_range,
                'ci_90':     (ci_low, ci_high),
            }

    # ── Write back to part ────────────────────────────────────────────────────
    part.obs['sub_fitness']      = sub_fitness
    part.obs['sub_fitness_5']    = sub_fitness_5
    part.obs['sub_fitness_95']   = sub_fitness_95
    part.obs['relative_fitness'] = relative_fitness

    part.uns['sub_mutation_inference'] = uns_results

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for mut_name, res in uns_results.items():
        role = res['role']
        if role == 'leading':
            print(f"  [{res['clone_idx']}] {mut_name:40s}  LEADING   "
                  f"s={res['s_sub']:.4f}  s_rel= 0.0000")
        else:
            direction = '↑' if res['s_rel'] > 0.05 else ('↓' if res['s_rel'] < -0.05 else '~')
            print(f"  [{res['clone_idx']}] {mut_name:40s}  sub {direction}     "
                  f"s={res['s_sub']:.4f}  s_rel={res['s_rel']:+.4f}")

    print("\n✅ Sub-mutation inference complete")
    return part


# ==============================================================================
# Convenience: run on a saved cohort
# ==============================================================================

def run_sub_mutation_inference_on_cohort(
        input_file,
        output_file,
        s_resolution=100,
        window_multiplier=2.0,
        window_min=0.3,
):
    import pickle as pk
    import traceback

    print("=" * 70)
    print("SUB-MUTATION FITNESS INFERENCE — COHORT")
    print("=" * 70)

    with open(input_file, 'rb') as f:
        cohort = pk.load(f)
    print(f"Loaded {len(cohort)} participants from {input_file}")

    results = []
    success, failed = 0, 0

    for i, part in enumerate(cohort):
        pid = part.uns.get('participant_id', f'participant_{i}')
        print(f"\n[{i+1}/{len(cohort)}]  {pid}")

        try:
            part = infer_sub_mutation_fitness(
                part,
                s_resolution=s_resolution,
                window_multiplier=window_multiplier,
                window_min=window_min,
            )
            results.append(part)
            success += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            traceback.print_exc()
            results.append(part)   # keep original, don't drop participant
            failed += 1

    with open(output_file, 'wb') as f:
        pk.dump(results, f, protocol=4)

    print(f"\n{'='*70}")
    print(f"Done.  {success}/{len(cohort)} successful  ·  saved to {output_file}")
    print(f"{'='*70}")

    return results


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == '__main__':
    run_sub_mutation_inference_on_cohort(
        input_file  = '../exports/MDS/MDS_cohort_fitted.pk',
        output_file = '../exports/MDS/MDS_cohort_fitted_sub.pk',
    )