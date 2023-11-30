# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *

with open('../exports/fabre.pk', 'rb') as f:
    part_list = pk.load(f)
# %%

all_mutations = []
for part in part_list:
    all_mutations.extend(part.obs.index.values)

unique_mutations = list(set(all_mutations))
# %%

sns.histplot(x=[all_mutations.count(i) for i in unique_mutations])
plot_part(part_list[0])
[print(i) for i in unique_mutations if all_mutations.count(i)>3]