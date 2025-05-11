#!/usr/bin/env bash
set -e

# 1. Load the data
bash scripts/fetch_raw.sh

# 2. Execute all notebooks in order，output to data/english_only、data/samples
#    如果想无头执行，需要用 nbconvert 或 papermill
jupyter nbconvert --to notebook --execute code/1_Filter_English_Community_Notes_English.ipynb \
  --output executed-1.ipynb \
  --ExecutePreprocessor.timeout=3600

jupyter nbconvert --to notebook --execute code/2_Filter_Related_Data.ipynb \
  --output executed-2.ipynb \
  --ExecutePreprocessor.timeout=600

jupyter nbconvert --to notebook --execute code/3_Random_Sampling.ipynb \
  --output executed-3.ipynb \
  --ExecutePreprocessor.timeout=600