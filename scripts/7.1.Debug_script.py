import sys
sys.path.append("..")
from src.general_imports import *
import pickle as pk
import numpy as np

# Load data
print("Loading MDS cohort data...")
with open('../exports/MDS/MDS_cohort_processed.pk', 'rb') as f:
    MDS_cohort = pk.load(f)

print(f"Loaded {len(MDS_cohort)} participants")

# Examine the first participant in detail
part = MDS_cohort[0]
print(f"\n--- First Participant Details ---")
print(f"Participant ID: {part.uns.get('participant_id', 'Unknown')}")
print(f"Shape: {part.shape}")
print(f"Observation names: {list(part.obs.index[:5])}")
print(f"Variable names: {list(part.var.columns)}")

# Check layers
print(f"\nAvailable layers: {list(part.layers.keys())}")

# Check AO and DP layers
if 'AO' in part.layers and 'DP' in part.layers:
    AO = np.array(part.layers['AO'])
    DP = np.array(part.layers['DP'])
    
    print(f"\nAO shape: {AO.shape}")
    print(f"DP shape: {DP.shape}")
    
    # Check for NaNs or infinities
    print(f"AO has NaNs: {np.isnan(AO).any()}")
    print(f"DP has NaNs: {np.isnan(DP).any()}")
    print(f"AO has inf: {np.isinf(AO).any()}")
    print(f"DP has inf: {np.isinf(DP).any()}")
    
    # Check for zeros in DP (can cause division by zero)
    print(f"DP has zeros: {(DP == 0).any()}")
    
    # Check VAF values
    VAF = AO / np.where(DP > 0, DP, 1)  # Avoid division by zero
    print(f"VAF range: [{VAF.min():.4f}, {VAF.max():.4f}]")
    print(f"Number of VAF > 1: {(VAF > 1).sum()}")
    print(f"Number of VAF > 0.5: {(VAF > 0.5).sum()}")
    
    # Check time points
    if 'time_points' in part.var:
        print(f"\nTime points: {part.var['time_points'].values}")
        print(f"Number of time points: {len(part.var['time_points'])}")
    
    # Check X matrix
    print(f"\nX shape: {part.X.shape}")
    print(f"X sample: {part.X[:2, :5]}")
    
else:
    print("❌ ERROR: AO or DP layers missing!")