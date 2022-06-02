#%%
import time
from pprint import pprint as pp

# %%
def print_run_time(main_func):
    tic = time.perf_counter()
    main_func()
    toc = time.perf_counter()
    interval = toc - tic
    if interval < 60:
        pp(f"{interval:.2f} seconds")
    elif interval < 300 * 60:
        pp(f"{interval/60:.2f} minutes")
    else:
        pp(f"{interval/(60*60):.2f} hours")
    
# %%
