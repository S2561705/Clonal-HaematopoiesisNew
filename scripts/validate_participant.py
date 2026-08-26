import sys
sys.path.append('..')
import pickle as pk
from src.KI_3 import validate_participant

with open('../exports/MDS/MDS_cohort_fitted.pk', 'rb') as f:
    cohort = pk.load(f)

for pid in ['MDS711P64', 'MDS1134R53']:
    part = next((p for p in cohort if p.uns.get('participant_id') == pid), None)
    if part is None:
        print(f'{pid}: not found in current cohort file')
        continue
    print(f'--- {pid} ---')
    validate_participant(part)
    print()
