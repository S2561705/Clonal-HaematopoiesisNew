# test_mixed_deterministic.py
import sys
sys.path.append("..")
import jax.numpy as jnp
import numpy as np

# Import our modified function
from src.mix_inference import compute_deterministic_size_mixed

# Create test data
AO = jnp.array([[10, 20], [30, 40]])  # 2 timepoints, 2 mutations
DP = jnp.array([[100, 100], [100, 100]])
cs = [[0, 1]]  # Both mutations in same clone
n_mutations = 2

# Test cases
test_cases = [
    ("All heterozygous", [0.0, 0.0]),
    ("All homozygous", [1.0, 1.0]),
    ("Mixed 50/50", [0.5, 0.5]),
    ("Different zygosity", [0.0, 1.0]),  # First het, second hom
]

for desc, alpha in test_cases:
    print(f"\n{desc}: α={alpha}")
    
    # Test with VAFs ~0.5 (heterozygous level)
    deterministic_size, total_cells = compute_deterministic_size_mixed(
        cs, AO, DP, jnp.array(alpha), n_mutations
    )
    
    print(f"  Deterministic size shape: {deterministic_size.shape}")
    print(f"  Total cells: {total_cells}")
    print(f"  Size sample: {deterministic_size[0, :2]}")