# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from functools import partial

with open('../exports/fabre.pk', 'rb') as f:
    participant_list = pk.load(f)

# Plot sample participant
part = participant_list[3]
plot_part(part)

# Global computations
# %%
cs = [[0,1], [2]]
s = jnp.array([0.01, 0.02])

from scipy.optimize import minimize

%timeit 
func = minimize(nll, np.array([0.1, 0.1]), args=(part, cs), method='SLSQP')

def nll (s, part, cs):
    time_points = jnp.array(part.var.time_points)
    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    s = jnp.array(s)

    clonal_prob = jax_cs_hmm_ll(s, AO, DP, time_points, cs,lamb=1.3)
    
    return -jnp.sum(jnp.log(clonal_prob))

from jax.scipy.optimize import jax_minimize

minimize(nll, jnp.array([0.1, 0.1]), args=(part, cs), method='BFGS')

def jax_cs_hmm_ll(s, AO, DP, time_points,
                  cs,
                  lamb=1.3):