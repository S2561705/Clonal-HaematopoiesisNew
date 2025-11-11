# ==========================================
#   CLONAL DYNAMICS PIPELINE WITH CLONAL COMPETITION - FIXED
# ==========================================

# --- Step 1: Setup Environment ---
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 2: Load Input File ---
DATA_PATH = "/content/MDS_COHORT_3.csv"
OUT_DIR = Path("/content/clonal_results")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Configuration
CONFIG = {
    "total_cells": 1e5,
    "model_params": {
        "draws": 800,
        "tune": 800,
        "chains": 2,
        "target_accept": 0.95,  # Higher for better convergence
        "random_seed": 123
    }
}

# --- Step 3: Preprocess / Clean Data ---
def add_age_from_dates(df):
    df['SAMPLE_DATE'] = pd.to_datetime(df['SAMPLE_DATE'], errors='coerce')
    df['DATE_DIAGNOSIS'] = pd.to_datetime(df['DATE_DIAGNOSIS'], errors='coerce')
    df['years_since_diagnosis'] = (df['SAMPLE_DATE'] - df['DATE_DIAGNOSIS']).dt.days / 365.25
    df['age'] = df['AGE_AT_DIAGNOSIS'] + df['years_since_diagnosis']
    df = df[df['age'].notna() & (df['age'] > 0)]
    return df

def ensure_age_column(df):
    if 'age' in df.columns:
        return df
    elif {'AGE_AT_DIAGNOSIS','DATE_DIAGNOSIS','SAMPLE_DATE'}.issubset(df.columns):
        return add_age_from_dates(df)
    elif 'VISIT_NUMBER' in df.columns:
        df['age'] = df['VISIT_NUMBER']
        return df
    else:
        raise ValueError("No valid columns to compute age.")

def clean_data(df: pd.DataFrame):
    df.columns = df.columns.str.strip()
    rename_map = {'SAMPLE_ID':'participant_id','GENE':'PreferredSymbol','VAF':'AF','READ_DEPTH':'DP','AGE':'age'}
    df = df.rename(columns=rename_map)

    # AF and DP cleanup
    if 'AF' in df.columns:
        df['AF'] = pd.to_numeric(df['AF'].astype(str).str.replace(',','.').str.replace('<',''), errors='coerce') / 100.0
    if 'DP' in df.columns:
        df['DP'] = df['DP'].replace('NULL', np.nan)
        df['DP'] = pd.to_numeric(df['DP'], errors='coerce')
        df = df.dropna(subset=['DP'])
        df = df[df['DP'] > 0]

    # Compute AO
    if {'AF','DP'}.issubset(df.columns):
        df['AO'] = np.round(df['AF'] * df['DP']).astype(int)
        df = df[(df['AO'] >= 0) & (df['AO'] <= df['DP'])]

    # Ensure age
    df = ensure_age_column(df)

    # Mutation identifier
    if {'PreferredSymbol','PROTEIN_CHANGE'}.issubset(df.columns):
        df['Gene_protein'] = df['PreferredSymbol'] + '_' + df['PROTEIN_CHANGE']

    # Drop missing essential data
    required_cols = ['participant_id','age','PreferredSymbol','PROTEIN_CHANGE','AF','DP','AO']
    df = df.dropna(subset=required_cols)

    return df

# --- Step 4: Filter longitudinal participants ---
def filter_participants(df: pd.DataFrame):
    timepoint_counts = df.groupby("participant_id")["age"].nunique()
    keep_ids = timepoint_counts[timepoint_counts>=2].index
    df_filtered = df[df["participant_id"].isin(keep_ids)].copy()
    return df_filtered

# --- Step 5: SIMPLIFIED CLONAL COMPETITION FITNESS INFERENCE ---
def infer_clonal_competition(participant_data, draws=800, tune=800, seed=123):
    """
    Simplified clonal competition model with fixed PyMC syntax
    """
    # Get unique timepoints and mutations
    times = participant_data['age'].unique()
    times.sort()
    mutations = participant_data['Gene_protein'].unique()

    n_timepoints = len(times)
    n_mutations = len(mutations)

    print(f"    Modeling {n_mutations} mutations across {n_timepoints} timepoints")

    # Create observation matrices
    AO_matrix = np.zeros((n_timepoints, n_mutations), dtype=int)
    DP_matrix = np.zeros((n_timepoints, n_mutations), dtype=int)

    # Fill observation matrices
    mutation_idx = {mut: i for i, mut in enumerate(mutations)}
    time_idx = {t: i for i, t in enumerate(times)}

    for _, row in participant_data.iterrows():
        t_idx = time_idx[row['age']]
        m_idx = mutation_idx[row['Gene_protein']]
        AO_matrix[t_idx, m_idx] = row['AO']
        DP_matrix[t_idx, m_idx] = row['DP']

    t = times - times[0]  # Relative times

    with pm.Model() as model:
        # FIXED: Use tighter prior for s
        s = pm.Normal("s", 0, 0.5, shape=n_mutations)  # Tighter prior for stability
        H0 = pm.Uniform("H0", 0, 1e6, shape=n_mutations)  # Uniform priors
        M0 = pm.Uniform("M0", 0, 1e6, shape=n_mutations)    # Uniform priors

        # Population dynamics for each mutation
        x_het = H0[:, None] * pm.math.exp(s[:, None] * t[None, :])
        x_hom = M0[:, None] * pm.math.exp(s[:, None] * t[None, :])

        # Total mutant cells across all clones (competition happens here!)
        total_mutant_cells = pm.math.sum(x_het + x_hom, axis=0)

        # VAF for each mutation: (het_i + 2*hom_i) / (2 * (total_cells + total_mutant_cells))
        denominator = 2.0 * (CONFIG["total_cells"] + total_mutant_cells)

        # VAF for each mutation at each timepoint
        p_matrix = (x_het + 2.0 * x_hom) / denominator
        p_clip = pm.math.clip(p_matrix, 1e-4, 0.9999)

        # Likelihood for each mutation - SIMPLIFIED to avoid complex indexing
        for i in range(n_mutations):
            pm.Binomial(f"y_{i}", n=DP_matrix[:, i], p=p_clip[i, :], observed=AO_matrix[:, i])

        # Sample
        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=0.95,  # Higher for better convergence
            random_seed=seed,
            chains=2,
            cores=1,
            progressbar=False,
            compute_convergence_checks=False  # Reduce warnings
        )

    return idata, mutations, times

# --- Step 6: Summarize and normalize fitness posterior ---
def summarize_clonal_idata(idata, mutations):
    """Summarize results from clonal competition model with normalization"""
    try:
        summary = az.summary(idata, var_names=["s", "H0", "M0"], kind="stats", round_to=6)

        results = []
        s_samples = idata.posterior["s"].values.reshape(-1, len(mutations))
        # Normalize posterior for s
        s_norm = s_samples / np.sum(np.abs(s_samples), axis=0)

        for i, mutation in enumerate(mutations):
            hdi_low_col = 'hdi_3%' if 'hdi_3%' in summary.columns else 'hdi_2.5%'
            hdi_high_col = 'hdi_97%' if 'hdi_97%' in summary.columns else 'hdi_97.5%'

            s_stats = summary.loc[f"s[{i}]"]
            H0_stats = summary.loc[f"H0[{i}]"]
            M0_stats = summary.loc[f"M0[{i}]"]

            results.append({
                "mutation": mutation,
                "gene": mutation.split('_')[0],
                "protein_change": '_'.join(mutation.split('_')[1:]),
                "s_mean": float(s_stats["mean"]),
                "s_mean_normalized": float(np.mean(s_norm[:, i])),
                "s_hdi_low": float(s_stats[hdi_low_col]),
                "s_hdi_high": float(s_stats[hdi_high_col]),
                "H0_mean": float(H0_stats["mean"]),
                "M0_mean": float(M0_stats["mean"]),
                "H0_hdi_low": float(H0_stats[hdi_low_col]),
                "H0_hdi_high": float(H0_stats[hdi_high_col]),
                "M0_hdi_low": float(M0_stats[hdi_low_col]),
                "M0_hdi_high": float(M0_stats[hdi_high_col])
            })

        return results
    except Exception as e:
        print(f"    Summary failed: {e}")
        return [{"mutation": mut, "error": f"Summary failed: {e}"} for mut in mutations]

# --- Step 7: Main pipeline and results saving ---
def main():
    df = pd.read_csv(DATA_PATH)
    df_clean = clean_data(df)
    df_filtered = filter_participants(df_clean)
    results = []

    participants = df_filtered['participant_id'].unique()
    print(f"Processing {len(participants)} participants with clonal competition model...")

    for pid in tqdm(participants, desc="Participants"):
        sdf = df_filtered[df_filtered['participant_id'] == pid].copy()
        mutations_in_participant = sdf['Gene_protein'].unique()
        if len(mutations_in_participant) < 1:
            continue
        print(f"\n🔬 Processing {pid} with {len(mutations_in_participant)} mutations: {list(mutations_in_participant)}")
        timepoint_counts = sdf.groupby('Gene_protein')['age'].nunique()
        if any(timepoint_counts < 2):
            print(f"    ⚠️  Skipping - some mutations have <2 timepoints")
            continue
        try:
            idata, mutations, times = infer_clonal_competition(
                sdf,
                draws=CONFIG["model_params"]["draws"],
                tune=CONFIG["model_params"]["tune"],
                seed=CONFIG["model_params"]["random_seed"]
            )
            mutation_results = summarize_clonal_idata(idata, mutations)
            for result in mutation_results:
                if "error" not in result:
                    result["participant_id"] = pid
                    result["n_timepoints"] = len(times)
                    result["n_mutations_in_participant"] = len(mutations)
                    result["timepoints"] = list(times)
            results.extend(mutation_results)
            print(f"    ✅ Success: inferred fitness for {len(mutations)} mutations")
        except Exception as e:
            print(f"    ❌ Clonal competition failed: {str(e)}")
            for mutation_id in mutations_in_participant:
                results.append({
                    "participant_id": pid,
                    "mutation": mutation_id,
                    "error": f"Inference failed: {str(e)}"
                })
    results_df = pd.DataFrame(results)
    results_path = OUT_DIR / "clonal_competition_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n📈 PIPELINE COMPLETED!")
    print(f"   Results saved to: {results_path}")

if __name__ == "__main__":
    main()
# ...existing code for plotting and main pipeline...
