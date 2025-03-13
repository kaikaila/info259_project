#!/usr/bin/env python
# coding: utf-8

# # Twitter Community Notes Data Sampler
# 
# This notebook is used to extract sample data from large TSV files for quick exploration and topic model testing.
# 
# ## Objectives:
# - Randomly extract 10,000-50,000 rows from each TSV file
# - Control sample file size to be within 50-100MB
# - Prepare appropriately sized data for topic modeling

# In[1]:


import os
import pandas as pd
import numpy as np
import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)  # a seed to render the sample files deterministic 

# Set more attractive chart styles
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


# ## 1. Configuration Parameters
# 
# First, set the input and output paths, as well as sampling parameters

# In[2]:


# Configure paths - Modify this to match your folder structure
RAW_DATA_DIR = os.path.expanduser("/Users/yunkaili/spring2025/NLP/project/data/english_only")
OUTPUT_DIR = os.path.expanduser("/Users/yunkaili/spring2025/NLP/project/data/samples")

# Create output directory (if it doesn't exist)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sampling parameters
MIN_SAMPLE_ROWS = 10000  # Minimum sample rows
MAX_SAMPLE_ROWS = 50000  # Maximum sample rows
TARGET_SAMPLE_SIZE_MB = 75  # Target sample size (MB)
MAX_SAMPLE_SIZE_MB = 100  # Maximum sample size (MB)


# ## 2. Find All TSV Files

# In[3]:


# Find all TSV files
tsv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.tsv"))

print(f"Found {len(tsv_files)} TSV files:")
for i, file in enumerate(tsv_files):
    file_size_mb = os.path.getsize(file) / (1024 * 1024)
    print(f"{i+1}. {os.path.basename(file)} ({file_size_mb:.2f} MB)")


# ## 3. Define Sampling Function
# 
# This function will automatically determine how to best sample from large files

# In[4]:


def create_sample(file_path, output_path, min_rows=MIN_SAMPLE_ROWS, max_rows=MAX_SAMPLE_ROWS, 
                   target_size_mb=TARGET_SAMPLE_SIZE_MB, max_size_mb=MAX_SAMPLE_SIZE_MB):
    """
    Create an appropriately sized sample file from a TSV file
    - If the original file is not large, directly use all rows less than the maximum row count
    - For large files, first estimate the average size per row, then sample an appropriate number of rows
    """
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Processing file: {os.path.basename(file_path)} ({file_size_mb:.2f} MB)")
    
    # 1. First calculate the total number of rows in the file
    total_rows = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in tqdm(f, desc="Calculating total rows"):
            total_rows += 1
    
    print(f"Total rows in file: {total_rows}")
    
    # 2. Estimate average row size
    avg_row_size_bytes = (file_size_mb * 1024 * 1024) / total_rows
    print(f"Average row size: {avg_row_size_bytes:.2f} bytes")
    
    # 3. Calculate target row count, keeping it between min and max row counts
    target_rows = int((target_size_mb * 1024 * 1024) / avg_row_size_bytes)
    sample_rows = min(max(target_rows, min_rows), max_rows)
    sample_rows = min(sample_rows, total_rows)  # Don't exceed total rows
    
    print(f"Will sample {sample_rows} rows (approx. {(sample_rows * avg_row_size_bytes)/(1024*1024):.2f} MB)")
    
    # 4. Decide sampling strategy based on file size
    if file_size_mb <= max_size_mb and total_rows <= max_rows:
        # Small file: read entire file
        print("File is small, keeping all data")
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
        df.to_csv(output_path, sep='\t', index=False)
    else:
        # Large file: random sampling
        print("File is large, using random sampling")
        # Determine sampling method (chunking or random indices)
        if total_rows > 1000000:
            # Very large file: use chunked sampling
            sample_fraction = sample_rows / total_rows
            chunks = []
            
            for chunk in tqdm(pd.read_csv(file_path, sep='\t', chunksize=100000, low_memory=False),
                             desc="Sampling chunks"):
                chunk_sample = chunk.sample(frac=sample_fraction)
                chunks.append(chunk_sample)
            
            df = pd.concat(chunks)
            # If result exceeds required rows, sample again
            if len(df) > sample_rows:
                df = df.sample(n=sample_rows)
            
            df.to_csv(output_path, sep='\t', index=False)
        else:
            # Medium-sized file: use random row indices
            random_indices = sorted(np.random.choice(total_rows, size=sample_rows, replace=False))
            
            # Read header row (column names)
            header = pd.read_csv(file_path, sep='\t', nrows=0)
            header.to_csv(output_path, sep='\t', index=False)
            
            # Append sampled rows
            current_idx = 0
            with open(file_path, 'r', encoding='utf-8') as infile, \
                 open(output_path, 'a', encoding='utf-8') as outfile:
                # Skip header row
                next(infile)
                
                for i, line in tqdm(enumerate(infile), desc="Sampling random rows"):
                    if current_idx < len(random_indices) and i == random_indices[current_idx]:
                        outfile.write(line)
                        current_idx += 1
                        
                    # Exit loop if all sampled rows have been processed
                    if current_idx >= len(random_indices):
                        break
    
    # 5. Display result information
    sample_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Sample file saved: {os.path.basename(output_path)} ({sample_size_mb:.2f} MB)")
    print("-" * 80)
    
    return sample_size_mb


# ## 4. Process All TSV Files, Create Samples

# In[5]:


# Create samples for each TSV file
sample_files = []
sample_sizes = []

for file_path in tsv_files:
    base_name = os.path.basename(file_path)
    output_path = os.path.join(OUTPUT_DIR, f"sample_{base_name}")
    
    sample_size = create_sample(file_path, output_path)
    
    sample_files.append(base_name)
    sample_sizes.append(sample_size)


# ## 5. Sample Size Visualization

# In[6]:


# Create sample size bar chart
plt.figure(figsize=(12, 8))
bars = plt.barh(sample_files, sample_sizes, color='skyblue')
plt.xlabel('File Size (MB)')
plt.title('Sample File Sizes')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f} MB',
            ha='left', va='center')

plt.tight_layout()
plt.show()


# ## 6. Perform Basic Statistical Analysis on Each Sample File

# In[7]:


def analyze_sample(file_path):
    """
    Perform basic analysis on a sample file
    """
    print(f"Analyzing file: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    
    # print(f"Row count: {len(df)}")
    print(f"Column count: {len(df.columns)}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nFirst 5 rows:")
    display(df.head())
    
    print("\nMissing value statistics:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    print("-" * 80)
    
    return df


# In[8]:


# Analyze the first sample file as an example
if len(sample_files) > 0:
    for i in range(len(sample_files)): 
        a_sample = os.path.join(OUTPUT_DIR, f"sample_{sample_files[i]}")
        sample_df = analyze_sample(a_sample)


# ## 8. Summary
# 
# 1. Sample files have been created from the original TSV files
# 2. Each sample file is controlled to be within the target size range (50-100MB)
# 3. Basic analysis has been performed on the first sample file
# 
# Next, you can upload these sample files to Google Colab and begin experimenting with topic modeling.

# In[ ]:




