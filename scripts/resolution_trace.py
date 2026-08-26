"""
resolution_trace.py
====================
Wraps compute_clonal_models_prob_vec / refine_optimal_model_posterior_vec so
their actual resolution arguments get logged every time they're called --
then runs the real pipeline script (7.KI_clonal_fit.py) unmodified through
runpy, so we see EXACTLY what parameters the pipeline used on this
invocation, without editing the pipeline file itself.

Run this twice in a row (two separate `python` processes) and diff
resolution_trace.log between them. If the logged (s_resolution, h_resolution,
beta_resolution) differ between runs, that's the non-determinism source --
config/parameter drift, not the math (which we already confirmed is
deterministic within a process).
"""
import sys, os, runpy
sys.path.append("..")

import src.KI_3 as ki3

LOG_FILE = "resolution_trace.log"

_orig_compare = ki3.compute_clonal_models_prob_vec
_orig_refine  = ki3.refine_optimal_model_posterior_vec

def _traced_compare(part, s_resolution=20, h_resolution=4, min_s=0.01,
                    max_s=3.0, max_h=1.0, filter_invalid=True,
                    disable_progressbar=False, beta_resolution=None):
    beta_res = beta_resolution if beta_resolution is not None else ki3.BETA_RESOLUTION
    pid = part.uns.get('participant_id', '?')
    with open(LOG_FILE, "a") as f:
        f.write(f"[compare] pid={pid}  s_resolution={s_resolution}  "
                f"h_resolution={h_resolution}  min_s={min_s}  max_s={max_s}  "
                f"max_h={max_h}  beta_resolution={beta_res}  "
                f"filter_invalid={filter_invalid}\n")
    kwargs = dict(s_resolution=s_resolution, h_resolution=h_resolution,
                 min_s=min_s, max_s=max_s, max_h=max_h,
                 filter_invalid=filter_invalid,
                 disable_progressbar=disable_progressbar)
    if beta_resolution is not None:
        kwargs['beta_resolution'] = beta_resolution
    return _orig_compare(part, **kwargs)

def _traced_refine(part, s_resolution=40, h_resolution=6, min_s=0.01,
                   max_s=3.0, max_h=1.0, beta_resolution=None):
    beta_res = beta_resolution if beta_resolution is not None else ki3.BETA_RESOLUTION
    pid = part.uns.get('participant_id', '?')
    with open(LOG_FILE, "a") as f:
        f.write(f"[refine]  pid={pid}  s_resolution={s_resolution}  "
                f"h_resolution={h_resolution}  min_s={min_s}  max_s={max_s}  "
                f"max_h={max_h}  beta_resolution={beta_res}\n")
    kwargs = dict(s_resolution=s_resolution, h_resolution=h_resolution,
                 min_s=min_s, max_s=max_s, max_h=max_h)
    if beta_resolution is not None:
        kwargs['beta_resolution'] = beta_resolution
    return _orig_refine(part, **kwargs)

ki3.compute_clonal_models_prob_vec = _traced_compare
ki3.refine_optimal_model_posterior_vec = _traced_refine

with open(LOG_FILE, "a") as f:
    f.write(f"\n=== run start, pid={os.getpid()} ===\n")
    f.write(f"BETA_RESOLUTION (module default) = {ki3.BETA_RESOLUTION}\n")
    if hasattr(ki3, 'Q_EPS'):
        f.write(f"Q_EPS (module default) = {ki3.Q_EPS}\n")

# run the real pipeline script unmodified, using the patched functions
runpy.run_path("7.KI_clonal_fit.py", run_name="__main__")

with open(LOG_FILE, "a") as f:
    f.write(f"=== run end ===\n")

print(f"\nTrace written to {LOG_FILE} -- run this script again "
      f"(fresh `python resolution_trace.py`) and diff the two blocks.")