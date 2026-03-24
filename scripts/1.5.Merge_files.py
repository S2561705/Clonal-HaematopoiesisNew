"""
merge_cohorts.py
----------------
Combines the original MDS_COHORT.csv with updated/corrected data from
Cameron_Patients.xlsx (one sheet per participant, wide format).

Excel takes priority for any participant present in both files.
cDNA changes in the Excel include transcript prefixes (e.g. NM_012433.3:c.1876A>G);
these are stripped to match the CSV convention (c.1876A>G).

Special VAF values handled:
  'Not in panel' → NaN  (gene not on panel at that timepoint)
  'ND'           → NaN  (not detected)
  'NA'           → NaN  (not available)
  '<1%' / '<1'   → 0.5  (below LOD, consistent with CSV cleaning)

VAF in the Excel is stored as a fraction (0–1); multiplied by 100 to match CSV.
"""

import pandas as pd
import numpy as np
from openpyxl import load_workbook
from datetime import datetime


# ── helpers ───────────────────────────────────────────────────────────────────

MISSING_SENTINELS = {'not in panel', 'nd', 'na', 'none', ''}

def parse_vaf(val):
    """Convert an Excel VAF cell to a percentage float, or NaN."""
    if val is None:
        return np.nan
    s = str(val).strip()
    if s.lower() in MISSING_SENTINELS:
        return np.nan
    if s.startswith('<'):
        return 0.5          # below LOD → 0.5 % (consistent with CSV)
    try:
        return float(s) * 100   # Excel stores as fraction e.g. 0.38
    except ValueError:
        return np.nan

def parse_depth(val):
    """Convert an Excel depth cell to float, or NaN."""
    if val is None:
        return np.nan
    s = str(val).strip()
    if s.lower() in MISSING_SENTINELS:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

def strip_transcript(cdna):
    """Remove transcript prefix: 'NM_012433.3:c.1876A>G' → 'c.1876A>G'."""
    if cdna is None:
        return None
    s = str(cdna).strip()
    if ':' in s:
        s = s.split(':', 1)[1]
    return s


# ── 1. Parse Excel ────────────────────────────────────────────────────────────

wb = load_workbook('../data/Cameron_Patients.xlsx', read_only=True)
excel_rows = []

for sheet_name in wb.sheetnames:
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))

    # Row layout (0-indexed):
    #   0 – filler
    #   1 – sample type (BM / PB) per timepoint
    #   2 – dates, one per timepoint pair
    #   3 – headers: ..., 'Gene', 'cDNA', 'Protein', then 'VAF','Sequencing depth' pairs
    #   4+ – mutation data

    date_row   = rows[2]
    header_row = rows[3]

    # Collect (date, vaf_col_idx, depth_col_idx) for each timepoint
    timepoints = []
    i = 5
    while i < len(header_row):
        cell = header_row[i]
        if cell is not None and str(cell).strip().lower() == 'vaf':
            date_val = date_row[i] if i < len(date_row) else None
            if isinstance(date_val, datetime):
                date_val = date_val.strftime('%Y-%m-%d')
            depth_idx = i + 1 if (i + 1) < len(header_row) else None
            timepoints.append((date_val, i, depth_idx))
            i += 2
        else:
            i += 1

    if not timepoints:
        print(f"  WARNING: no timepoint columns in sheet {sheet_name}, skipping")
        continue

    # Derive VISIT_NUMBER from date order (1-based)
    visit_numbers = list(range(1, len(timepoints) + 1))

    # Parse mutation rows (skip if gene cell is None → blank/filler row)
    for row in rows[4:]:
        gene    = row[2] if len(row) > 2 else None
        cdna    = row[3] if len(row) > 3 else None
        protein = row[4] if len(row) > 4 else None

        if gene is None or str(gene).strip() == '':
            continue

        # Skip annotation-only rows (no cDNA column data that fits VAF pattern)
        cdna_clean = strip_transcript(cdna)

        for visit_num, (date_val, vaf_idx, depth_idx) in zip(visit_numbers, timepoints):
            vaf_raw   = row[vaf_idx]   if vaf_idx   is not None and vaf_idx   < len(row) else None
            depth_raw = row[depth_idx] if depth_idx is not None and depth_idx < len(row) else None

            excel_rows.append({
                'SAMPLE_ID':      sheet_name,
                'VISIT_NUMBER':   visit_num,
                'SAMPLE_DATE':    date_val,
                'GENE':           str(gene).strip(),
                'cDNA_CHANGE':    cdna_clean,
                'PROTEIN_CHANGE': str(protein).strip() if protein else None,
                'VAF':            parse_vaf(vaf_raw),
                'READ_DEPTH':     parse_depth(depth_raw),
                'source':         'excel',
            })

excel_df = pd.DataFrame(excel_rows)
print(f"Excel parsed: {excel_df['SAMPLE_ID'].nunique()} participants, {len(excel_df)} rows")
print(f"  VAF NaN rate: {excel_df['VAF'].isna().mean():.1%}")


# ── 2. Load and clean CSV ─────────────────────────────────────────────────────

csv_df = pd.read_csv('../data/MDS_COHORT.csv', delimiter=';', dtype=str)
csv_df['source'] = 'csv'

def clean_csv_numeric(series):
    series = series.replace('NULL', np.nan)
    series = series.apply(lambda x: str(x).replace(',', '.') if pd.notnull(x) else x)
    series = series.apply(lambda x: 0.5 if pd.notnull(x) and str(x).startswith('<') else x)
    return pd.to_numeric(series, errors='coerce')

csv_df['VAF']        = clean_csv_numeric(csv_df['VAF'])
csv_df['READ_DEPTH'] = clean_csv_numeric(csv_df['READ_DEPTH'])

excel_participants = set(excel_df['SAMPLE_ID'].unique())
csv_keep = csv_df[~csv_df['SAMPLE_ID'].isin(excel_participants)].copy()

print(f"CSV:   {csv_df['SAMPLE_ID'].nunique()} participants total, "
      f"keeping {csv_keep['SAMPLE_ID'].nunique()} not covered by Excel")


# ── 3. Align and combine ──────────────────────────────────────────────────────

shared_cols = ['SAMPLE_ID', 'VISIT_NUMBER', 'SAMPLE_DATE', 'GENE',
               'cDNA_CHANGE', 'PROTEIN_CHANGE', 'VAF', 'READ_DEPTH', 'AGE', 'SEX', 'source']

# For the excel_df, add placeholder columns
excel_df['AGE'] = np.nan
excel_df['SEX'] = np.nan

combined = pd.concat(
    [csv_keep[shared_cols], excel_df[shared_cols]],
    ignore_index=True
)
combined = combined.sort_values(['SAMPLE_ID', 'VISIT_NUMBER', 'GENE']).reset_index(drop=True)

print(f"\nCombined: {combined['SAMPLE_ID'].nunique()} participants, {len(combined)} rows")
print(f"Source breakdown:\n{combined['source'].value_counts().to_string()}")
print(f"\nAll participants: {sorted(combined['SAMPLE_ID'].unique())}")

# Quick sanity: flag any rows with NaN VAF for review
nan_vaf = combined[combined['VAF'].isna()]
if not nan_vaf.empty:
    print(f"\n⚠ {len(nan_vaf)} rows with NaN VAF (ND / Not in panel / missing):")
    print(nan_vaf[['SAMPLE_ID','VISIT_NUMBER','GENE','cDNA_CHANGE']].to_string(index=False))


# ── 4. Save ───────────────────────────────────────────────────────────────────

out_path = '../data/MDS_COHORT_combined.csv'
combined.to_csv(out_path, index=False, sep=';')
print(f"\nSaved → {out_path}")