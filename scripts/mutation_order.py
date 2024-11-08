# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from src.deterministic_aux import *

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle

from scipy import stats

# Import results
with open('../exports/all_processed_with_deterministic.pk', 'rb') as f:
    cohort = pk.load(f)

# %%

cohort = [order_of_mutations(part) for part in cohort]

# Import results
with open('../exports/all_processed_with_deterministic.pk', 'wb') as f:
    pk.dump(cohort, f)