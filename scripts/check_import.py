import sys
sys.path.append("..")
import src.KI_3 as ki3

print("module file:")
print(ki3.__file__)
print()
import inspect
print("compute_clonal_models_prob_vec source (not jitted, should work directly):")
print(inspect.getsource(ki3.compute_clonal_models_prob_vec)[:500])