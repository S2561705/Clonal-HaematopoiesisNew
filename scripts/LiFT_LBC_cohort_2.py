# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from src.LiFT import *
from src import distributions
import pickle

# set random seed
np.random.seed(123)
with open('../exports/LBC/LBC_syn_cohort_2.pk', 'rb') as f:
    participant_syn = pk.load(f)

with open('../exports/LBC/LBC_non_syn_cohort_2.pk', 'rb') as f:
    participant_ns = pk.load(f)

# create dictionary of mutations
mut_dict_syn = create_mut_dict(participant_syn)
mut_dict_ns = create_mut_dict(participant_ns)

# Create priors from synonymous data
prior_list = fit_artefact_priors(mut_dict_syn)

# %%
mut_obj_list_syn = []
# Artefact model probability
for key, mut_data_list in mut_dict_syn.items():
    mut_obj_list_syn.append(
        mutation_class(key=key, data=mut_data_list)
    )

mut_obj_list_ns = []
# Artefact model probability
for key, mut_data_list in mut_dict_ns.items():
    mut_obj_list_ns.append(
        mutation_class(key=key, data=mut_data_list)
    )
# %%
# Compute model comparison
for i, mut in tqdm(enumerate(mut_obj_list_ns)):
    # print(f'Mutation {i} of {len(mut_obj_list_ns)}')

    if len(mut.data) > 15:
        mut.artefact_prob = 1
        mut.mut_prob = 0.001
    
    else:
        artefact_probability_p(mut, prior_list)
        real_mut_probability(mut, min_s=0.05, max_s=0.5, disable_progressbar=True)

# Compute model comparison
for i, mut in tqdm(enumerate(mut_obj_list_syn)):
    # print(f'Mutation {i} of {len(mut_obj_list_syn)}')

    if len(mut.data) > 15:
        mut.artefact_prob = 1
        mut.mut_prob = 0.001
    
    else:
        artefact_probability_p(mut, prior_list)
        real_mut_probability(mut, min_s=0.05, max_s=0.5, disable_progressbar=True)

# %%
BF = [mut.mut_prob/mut.artefact_prob for mut in mut_obj_list_ns]
BF_syn = [mut.mut_prob/mut.artefact_prob for mut in mut_obj_list_syn]

BF_threshold = np.linspace(0.2, 5, 91)
proportion = []
proportion_syn = []
for threshold in BF_threshold:
    proportion.append(100*(np.array(BF)>threshold).sum()/ len(mut_obj_list_ns))
    proportion_syn.append(100*(np.array(BF_syn)>threshold).sum()/ len(mut_obj_list_syn))

sns.lineplot(x=BF_threshold, y= proportion, label='non-syn')
sns.lineplot(x=BF_threshold, y= proportion_syn, label='syn')

plt.xlabel('BF_threshold (confidence)')
plt.ylabel('total % of mutations')
plt.savefig('../exports/LBC/cohort_2/BF_plot.png')
plt.savefig('../exports/LBC/cohort_2/BF_plot.svg')
# %%

participant_syn = update_cohort_LiFT(participant_syn, mut_obj_list_syn)
participant_ns = update_cohort_LiFT(participant_ns, mut_obj_list_ns)

with open('../exports/LBC/cohort_2/LBC_cohort_2_ns_LiFT_updated.pk', 'wb') as f:
    pk.dump(participant_ns, f)
   
with open('../exports/LBC/cohort_2/LBC_cohort_2_syn_LiFT_updated.pk', 'wb') as f:
    pk.dump(participant_syn, f) 
# %%
   
filtered_ns = [part[part.obs.LiFT_value>0.9] for part in participant_ns]

filtered_ns = [part for part in filtered_ns if part.shape[0]>0]

filtered_ns[0].obs