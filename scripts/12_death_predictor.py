# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.aux import *
from src.survival_predictor_aux import *

import pandas as pd
import matplotlib as mpl
import anndata as ad

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
import statsmodels.formula.api as smf

survival_df = pd.read_csv('../results/participant_df.csv', index_col=0)
survival_df = survival_df[survival_df['dead'].notna()]

data = survival_df[survival_df.dead==1].copy()

# %%
print('How good is age as a predictor of survival ages?')
# Select features and target
features = ['age_wave_1', 'Female']
target = 'from_wave_1'

model_age = train_model(data, features, target)

# %%
# Select features and target
features = [ 'max_fitness', 'max_size_prediction_120_z_score',  'max_VAF_z_score', 'Female', 'age_wave_1']

target = 'from_wave_1'

models = train_model(data, features, target)

# %%


import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Assuming your dataframe is called 'survival_df'

# Select the features and target
features = ['max_VAF_z_score', 'max_size_prediction_120_z_score', 'max_fitness_z_score', 'Female']
target = 'from_wave_1'

# Convert 'sex' to binary (assuming F=0, M=1)
# survival_df['sex_binary'] = (survival_df['sex'] == 'M').astype(int)

# Prepare the dataset for CoxPH
cph_data = survival_df[features + [target, 'dead']].copy()
# cph_data['sex'] = cph_data['sex_binary']
cph_data.rename(columns={'age_death': 'duration', 'dead': 'event'}, inplace=True)

# Split the data into training and testing sets
train_data, test_data = train_test_split(cph_data, test_size=0.2, random_state=42)

# Fit the CoxPH model
cph = CoxPHFitter()
cph.fit(train_data, duration_col='duration', event_col='event')

# Print the model summary
print(cph.print_summary())

# Make predictions on the test set
predictions = cph.predict_median(test_data)
# %%
plt.figure(figsize=(10, 8))
plt.scatter(test_data['duration'], predictions, alpha=0.5)
plt.plot([test_data['duration'].min(), test_data['duration'].max()], 
         [test_data['duration'].min(), test_data['duration'].max()], 'r--', lw=2)
plt.xlabel('Actual Age of Death')
plt.ylabel('Predicted Age of Death')
plt.title('Predicted vs Actual Age of Death')
plt.tight_layout()
plt.show()

# Calculate and print the Mean Absolute Error
mae = np.mean(np.abs(predictions - test_data['duration']))
print(f"Mean Absolute Error: {mae:.2f} years")
# %%


survival_df
sns.regplot(survival_df, x='age_wave_1', y ='max_size_prediction_120_z_score')
sns.regplot(survival_df, x='age_wave_1', y ='max_VAF_z_score')
sns.regplot(survival_df, x='age_wave_1', y ='Female')
