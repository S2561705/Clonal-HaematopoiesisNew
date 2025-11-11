import sys
sys.path.append("..")
from src.general_imports import *
import pickle as pk
import os

# Check the MDS directory
mds_dir = '../exports/MDS'
print(f"Checking directory: {mds_dir}")
files = os.listdir(mds_dir)
print(f"Files: {files}")

# Load and inspect the pickle file
pickle_path = '../exports/MDS/MDS_cohort_processed.pk'
print(f"\nLoading: {pickle_path}")

try:
    with open(pickle_path, 'rb') as f:
        data = pk.load(f)
    
    print(f"Successfully loaded data")
    print(f"Type: {type(data)}")
    print(f"Length: {len(data)} participants")
    
    for i, participant in enumerate(data):
        print(f"\n--- Participant {i+1} ---")
        print(f"Type: {type(participant)}")
        
        if hasattr(participant, 'shape'):
            print(f"Shape: {participant.shape}")
        else:
            print(f"No shape attribute")
            
        if hasattr(participant, 'layers'):
            print(f"Layers: {list(participant.layers.keys())}")
            if 'AO' in participant.layers:
                ao_layer = participant.layers['AO']
                print(f"AO layer shape: {ao_layer.shape}")
                print(f"AO layer type: {type(ao_layer)}")
                print(f"AO sample data: {ao_layer[:2] if ao_layer.size > 0 else 'Empty'}")
        else:
            print(f"No layers attribute")
            
        if hasattr(participant, 'uns'):
            print(f"uns keys: {list(participant.uns.keys())}")
            if 'participant_id' in participant.uns:
                print(f"Participant ID: {participant.uns['participant_id']}")
        else:
            print(f"No uns attribute")
            
        if hasattr(participant, 'var'):
            print(f"var columns: {list(participant.var.columns)}")
            if 'time_points' in participant.var.columns:
                print(f"time_points: {participant.var['time_points'].values}")
        else:
            print(f"No var attribute")
            
except Exception as e:
    print(f"Error loading pickle file: {e}")
    import traceback
    traceback.print_exc()