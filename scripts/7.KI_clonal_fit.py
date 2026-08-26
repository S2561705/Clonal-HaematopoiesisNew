import sys
sys.path.append("..")   # fix to import modules from root

from src.general_imports import *

from src.KI_3 import (
    compute_clonal_models_prob_vec,
    refine_optimal_model_posterior_vec
)
from src.manual_structure import (
    set_manual_clonal_structure,
    set_all_heterozygous,   # kept for reference; no longer called in the manual path
)

import numpy as np
import pickle as pk
import traceback


# ---------------------------------------------------------------------------
# Fixed-zygosity gene lists
# ---------------------------------------------------------------------------
# Obligate-heterozygous: spliceosome hotspots essentially never seen
# homozygous. Pinned h=0 for EVERY participant, unconditionally.
OBLIGATE_HET_GENES = {"SF3B1", "SRSF2", "U2AF1"}

# Hemizygous X-linked: in MALES (single X) one active allele means the
# VAF-doubling signal is indistinguishable from homozygosity -> pin h=1,
# same "can't be inferred, so fix it" logic as obligate-het. In FEMALES
# (two X copies) this does NOT apply -> leave unpinned (NaN), infer normally.
HEMIZYGOUS_GENES = {"ZRSR2", "PHF6", "KDM6A"}

# Sex lookup. Preprocessing writes ad.uns['sex'] as 'F' / 'M'.
SEX_KEY = "sex"
MALE_LABELS = {"m"}          # matched case-insensitively after strip


def _participant_is_male(part, sex_key=SEX_KEY, male_labels=MALE_LABELS):
    """True/False for male, or None if sex is missing/unknown. Checks
    part.uns[sex_key] first, then a part.obs column of that name."""
    val = None
    if sex_key in part.uns and part.uns[sex_key] is not None:
        val = part.uns[sex_key]
    elif sex_key in part.obs.columns:
        col = part.obs[sex_key].dropna().unique()
        if len(col) == 1:
            val = col[0]
        elif len(col) > 1:
            print(f"    WARNING: multiple sex values in obs: {col}; treating as unknown")
            return None
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return str(val).strip().lower() in male_labels


def flag_fixed_zygosity(part, obligate_het=OBLIGATE_HET_GENES,
                        hemizygous=HEMIZYGOUS_GENES,
                        sex_key=SEX_KEY, male_labels=MALE_LABELS):
    """Set part.obs['h_fixed'] from gene identity + participant sex, BEFORE
    inference. Obligate-het -> h=0 (all participants). Hemizygous X-linked
    -> h=1 ONLY if male; female/unknown -> left NaN (inferred). Every other
    mutation left NaN. Genes matched via the 'GENE' obs column."""
    if 'GENE' not in part.obs.columns:
        print("    WARNING: no 'GENE' column in obs; skipping zygosity pinning")
        part.obs['h_fixed'] = np.full(part.shape[0], np.nan)
        return part

    genes = part.obs['GENE'].astype(str).values
    h_fixed = np.full(part.shape[0], np.nan)
    is_male = _participant_is_male(part, sex_key, male_labels)

    het_hits, hemi_hits, hemi_skipped = [], [], []
    for i, g in enumerate(genes):
        if g in obligate_het:
            h_fixed[i] = 0.0
            het_hits.append(g)
        elif g in hemizygous:
            if is_male is True:
                h_fixed[i] = 1.0
                hemi_hits.append(g)
            else:
                hemi_skipped.append(g)   # female/unknown -> infer

    part.obs['h_fixed'] = h_fixed

    sex_str = ("male" if is_male is True
               else "female" if is_male is False else "unknown")
    msg = f"  zygosity pinning (sex={sex_str}):"
    if het_hits:      msg += f" het(h=0)->{sorted(set(het_hits))}"
    if hemi_hits:     msg += f" hemizygous(h=1)->{sorted(set(hemi_hits))}"
    if hemi_skipped:  msg += f" X-linked left to infer->{sorted(set(hemi_skipped))}"
    if not (het_hits or hemi_hits or hemi_skipped):
        msg += " none (all zygosity inferred)"
    print(msg)
    return part


# ---------------------------------------------------------------------------
# Heterozygous VAF cap  (CLONE-AWARE)                             # <-- CHANGED
# ---------------------------------------------------------------------------
# A het variant (h=0) structurally cannot exceed VAF 0.5 under this model
# (true_vaf -> (1+h)/2 as x->inf). An observed VAF >= 0.5 on an effectively-
# het mutation is treated as sampling error and pulled just under the ceiling
# by adjusting AO to cap*DP.
#
# CRITICAL: "effectively het" is NOT the same as "h_fixed == 0". build_clone_
# h_grids forces the WHOLE clone to a member's pin -- so an UNPINNED (NaN)
# mutation sharing a clone with an obligate-het (h=0) mutation is silently run
# at h=0 too, yet its h_fixed stays NaN. The old cap (is_het = h_fixed==0)
# skipped exactly those, leaving an uncapped >0.5 het VAF that drove the
# shared cell budget negative -> "Zero posterior". This version propagates
# each clone's pin across its members (matching build_clone_h_grids), so any
# mutation that will actually run at h=0 gets capped. Requires model_dict to
# already exist (i.e. call AFTER set_manual_clonal_structure).
#
# NOTE: pinning is why isolating a pinned mutation in its own clone (e.g.
# [[U2AF1_idx], [others]]) is often the better fix -- it stops a high-VAF,
# possibly-LOH partner from being forced het at all, letting its zygosity be
# inferred instead of clamped. Original AO is stashed in layers['AO_raw'].
# ---------------------------------------------------------------------------
def cap_het_vaf(part, cap=0.499):
    if 'h_fixed' not in part.obs.columns:
        return part

    h_fixed = np.array(part.obs['h_fixed'].values, dtype=float)

    # Propagate clone pins exactly as build_clone_h_grids does: a clone with
    # any pinned member forces that pin on every member of the clone.
    eff_h = h_fixed.copy()
    cs = (list(part.uns['model_dict'].values())[0][0]
          if 'model_dict' in part.uns and part.uns['model_dict'] else None)
    if cs is not None:
        for clone in cs:
            idx = np.array(clone)
            pins = h_fixed[idx]
            pins = pins[~np.isnan(pins)]
            if pins.size:
                eff_h[idx] = pins[0]   # whole clone takes the pin

    is_het = (eff_h == 0.0)            # NaN and h=1 -> False, left alone

    AO = np.array(part.layers['AO'], dtype=float)
    DP = np.array(part.layers['DP'], dtype=float)
    observed = DP > 0
    vaf = np.where(observed, AO / np.where(DP > 0, DP, 1.0), np.nan)
    over = is_het[:, None] & observed & (vaf >= 0.5)

    if over.any():
        if 'AO_raw' not in part.layers:
            part.layers['AO_raw'] = np.array(part.layers['AO']).copy()
        n_hits = int(over.sum())
        max_vaf = float(np.nanmax(vaf[over]))
        AO[over] = cap * DP[over]
        part.layers['AO'] = AO
        print(f"  het VAF cap: clamped {n_hits} timepoint(s) to {cap} "
              f"(max original VAF was {max_vaf:.3f})")

    return part


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE = "../exports/MDS/MDS_cohort_processed.pk"
OUTPUT_FILE = "../exports/MDS/MDS_cohort_fitted.pk"

S_RESOLUTION = 10
# Model-comparison h grid. Linus: at h_res=4 the box-average over a
# sharply peaked high-VAF clone is dominated by whether a grid point lands
# near the peak -- structure-dependent discretisation error that can exceed
# the Fix-1 prior term. Raising this (e.g. 8) reduces that; cost is
# h_res ** n_clones. Keep 4 for speed; bump if structure calls look fragile.
H_RESOLUTION = 6

REFINE_S = 60           # fitness marginal is smooth; 60 is plenty
REFINE_H = 16            # refine: 6^n_clones tractable
# BETA_RESOLUTION lives in KI_3 (default 1000); raise via kwargs if needed.

MIN_S = 0.01
MAX_S = 3.0

MAX_H = 1.0

FILTER_INVALID = True

# ---------------------------------------------------------------------------
# Manual clonal structures (PI's suggested workflow). Populate this dict
# per participant AFTER inspecting that participant's VAF-over-time in a
# separate exploration pass -- hardcode the resulting cs here once decided.
#
# Participants listed here SKIP the automated structure search entirely: the
# hand-picked structure is installed directly, gene-based zygosity pins are
# applied via flag_fixed_zygosity (obligate-het -> h=0; male hemizygous
# X-linked -> h=1; everything else left to be INFERRED), het VAFs >=0.5 are
# clamped by cap_het_vaf (clone-aware), and zygosity for all unpinned
# mutations is then inferred by refine_optimal_model_posterior_vec below.
#
# NOTE: build_clone_h_grids forces a whole clone to any pinned member's h,
# and asserts pins WITHIN a clone agree. So (a) do NOT group an obligate-het
# (h=0) with a male-hemizygous (h=1) mutation in one clone -- that raises
# "clone mixes conflicting fixed-h values"; and (b) grouping a high-VAF,
# possibly-LOH mutation with an obligate-het mutation forces the LOH one to
# het (then clamped) -- usually you want it in its OWN clone so its zygosity
# is inferred instead.
#
# Format: participant_id -> cs (list of lists of mutation POSITIONAL
# indices, 0-based, matching that participant's part.obs row order).
# ---------------------------------------------------------------------------
MANUAL_STRUCTURES = {
   "MDS581N49": [[0], [1, 2]],
   "MDS671W51": [[0], [1, 2, 3], [4]],
   "MDS711P64": [[0, 2], [1]],
   "MDS759K49": [[0, 1]],
   "MDS760G64": [[0, 2], [1]],
   "MDS766C49": [[1, 2], [0, 3], [4]],
   "MDS889H46": [[0, 1], [2]],
   "MDS893B85": [[0], [1]],
   "MDS907K41": [[0, 1, 2, 3]],
   "MDS918V64": [[0, 1, 2, 3]],
   "MDS1134R53": [[0, 1]],
   "MDS1135H55": [[0, 1, 2], [3]],
}

# ---------------------------------------------------------------------------
# Load cohort
# ---------------------------------------------------------------------------

with open(INPUT_FILE, "rb") as f:
    cohort = pk.load(f)

print(f"Loaded {len(cohort)} participants")
if MANUAL_STRUCTURES:
    print(f"Manual structures defined for: {sorted(MANUAL_STRUCTURES)}")


# ---------------------------------------------------------------------------
# Run clonal inference
# ---------------------------------------------------------------------------

processed_part_list = []

for i, part in enumerate(cohort):

    pid = part.uns.get("participant_id")
    print(f"\nParticipant {i + 1} of {len(cohort)}  ({pid})")

    try:
        part.uns["warning"] = None
        part.uns["fit_failed"] = False
        part.uns["fit_failed_reason"] = None

        # -------------------------------------------------------------------
        # Fixed-zygosity pinning (obligate-het + sex-conditional hemizygous)
        # -- must happen BEFORE the structure search so it informs cs_list.
        # -------------------------------------------------------------------
        part = flag_fixed_zygosity(part)

        if pid in MANUAL_STRUCTURES:
            # -----------------------------------------------------------
            # Manual structure path -- skip the automated partition search.
            # Zygosity pins are already set by flag_fixed_zygosity above:
            # obligate-het -> h=0, male hemizygous -> h=1, everything else
            # left NaN and INFERRED by the refine step below. cap_het_vaf
            # runs AFTER the structure is installed (needs model_dict) so it
            # can propagate clone pins and clamp effectively-het mutations.
            # -----------------------------------------------------------
            cs = MANUAL_STRUCTURES[pid]
            part = set_manual_clonal_structure(part, cs)
            part = cap_het_vaf(part)   # clone-aware; needs model_dict from above
            print(f"  Using MANUAL structure: {cs}  "
                  f"(obligate-het/hemizygous pinned; rest inferred)")
        else:
            # -----------------------------------------------------------
            # Automated structure search (now runs WITH the gene pins)
            # -----------------------------------------------------------
            part = compute_clonal_models_prob_vec(
                part,
                s_resolution=S_RESOLUTION,
                h_resolution=H_RESOLUTION,
                min_s=MIN_S,
                max_s=MAX_S,
                max_h=MAX_H,
                filter_invalid=FILTER_INVALID,
                disable_progressbar=False,
            )

            if part.uns.get("warning") is not None:
                print(f"  WARNING after model comparison: {part.uns['warning']}")

            if "model_dict" not in part.uns or len(part.uns["model_dict"]) == 0:
                raise RuntimeError("No valid models found")

            top_model = list(part.uns["model_dict"].values())[0]
            print(f"  Top model (automated): {top_model[0]}")
            print(f"  Top model raw probability: {top_model[1]:.3e}")

        # -------------------------------------------------------------------
        # Posterior refinement -- fitness + zygosity inference (zygosity now
        # genuinely inferred for every unpinned mutation).
        # -------------------------------------------------------------------

        part = refine_optimal_model_posterior_vec(
            part,
            s_resolution=REFINE_S,
            h_resolution=REFINE_H,
            max_h=MAX_H,
        )

        if part.uns.get("warning") is not None:
            print(f"  WARNING after refinement: {part.uns['warning']}")

        if "fitness" in part.obs.columns:
            print(part.obs[
                [
                    "fitness",
                    "fitness_5",
                    "fitness_95",
                    "homozygosity",
                    "homozygosity_5",
                    "homozygosity_95",
                    "clonal_index",
                ]
            ].to_string())

        processed_part_list.append(part)

        print(f"  -> participant {i + 1} OK")

    except Exception as e:
        print(f"  -> participant {i + 1} FAILED: {e}")
        traceback.print_exc()

        part.uns["fit_failed"] = True
        part.uns["fit_failed_reason"] = str(e)

        processed_part_list.append(part)


# ---------------------------------------------------------------------------
# Save fitted cohort
# ---------------------------------------------------------------------------

with open(OUTPUT_FILE, "wb") as f:
    pk.dump(processed_part_list, f, protocol=4)

print(f"\nSaved {len(processed_part_list)} participants -> {OUTPUT_FILE}")
