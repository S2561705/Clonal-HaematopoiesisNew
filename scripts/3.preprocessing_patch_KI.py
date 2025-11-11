import sys
sys.path.append("..")
from src.general_imports import *
import pickle as pk

# Load and inspect the processed data
print("Loading MDS cohort data...")
with open('../exports/MDS/MDS_cohort_processed.pk', 'rb') as f:
    MDS_cohort = pk.load(f)

print(f"Loaded {len(MDS_cohort)} participants")

for i, part in enumerate(MDS_cohort):
    print(f"\n--- Participant {i+1} ---")
    print(f"Participant ID: {part.uns.get('participant_id', 'Unknown')}")
    print(f"Shape: {part.shape}")
    print(f"Layers: {list(part.layers.keys())}")
    
    if 'AO' in part.layers:
        print(f"AO layer shape: {part.layers['AO'].shape}")
        print(f"AO layer type: {type(part.layers['AO'])}")
        print(f"AO sample: {part.layers['AO'][:2] if part.layers['AO'].size > 0 else 'Empty'}")
    else:
        print("❌ AO layer missing!")
    
    if 'DP' in part.layers:
        print(f"DP layer shape: {part.layers['DP'].shape}")
    else:
        print("❌ DP layer missing!")
        
    print(f"var columns: {list(part.var.columns)}")
    if 'time_points' in part.var:
        print(f"time_points: {part.var['time_points'].values}")
    else:
        print("❌ time_points missing from var!")
