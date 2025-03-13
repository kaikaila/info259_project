#!/usr/bin/env python
# coding: utf-8

# In[2]:


import langdetect


# In[1]:


# English language filtering for Community Notes
# This script filters the Community Notes dataset to keep only English notes

import pandas as pd
import numpy as np
from langdetect import detect, LangDetectException
import os
from tqdm.notebook import tqdm
import re

# Define file paths
input_filepath = "../raw/notes-00000.tsv"
output_filepath = "../english_only/english_notes-00000.tsv"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_filepath), exist_ok=True)


# In[2]:


# Function to detect language with error handling
def detect_language(text):
    try:
        # Check if text is NaN or empty
        if pd.isna(text) or text.strip() == '':
            return 'unknown'
        
        # Clean text - remove URLs, mentions, hashtags and special characters
        cleaned_text = re.sub(r'http\S+|@\S+|#\S+', '', text)
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
        
        if cleaned_text.strip() == '':
            return 'unknown'
            
        # Detect language
        return detect(cleaned_text)
    except LangDetectException:
        return 'unknown'

# Read the TSV file
print("Reading the TSV file...")
notes_df = pd.read_csv(input_filepath, sep='\t')

# Display basic information about the data
print(f"Total number of notes: {len(notes_df)}")
display(notes_df.head())
print(notes_df.columns.tolist())

# Check if 'summary' column exists
if 'summary' not in notes_df.columns:
    raise ValueError("The 'summary' column is not found in the dataset.")

# Detect language for each note's summary
print("Detecting language for each note's summary...")
tqdm.pandas()
notes_df['language'] = notes_df['summary'].progress_apply(detect_language)


# In[3]:


# Display language distribution
language_counts = notes_df['language'].value_counts()
print("Language distribution:")
print(language_counts)

# Filter out non-English notes
english_notes_df = notes_df[notes_df['language'] == 'en']
print(f"Number of English notes: {len(english_notes_df)}")

# Save English notes to a new TSV file
print(f"Saving English notes to {output_filepath}...")
english_notes_df.to_csv(output_filepath, sep='\t', index=False)

print("Done!")

# Optional: Create a list of noteIds for English notes
# This will be useful for filtering other related files in step 1.b
english_note_ids = english_notes_df['noteId'].tolist()
np.save("../english_only/english_note_ids.npy", english_note_ids)
print(f"Saved {len(english_note_ids)} English note IDs to english_note_ids.npy")


# ## visualization

# In[4]:


# Create a pie chart to visualize language distribution in Community Notes
# This code should be run after language detection is completed

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[5]:


# Set the style for better visualization
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

# Assuming notes_df with language column is already available from previous code
# If running separately, uncomment the following lines:
# input_filepath = "../raw/notes-00000.tsv"
# notes_df = pd.read_csv(input_filepath, sep='\t')
# notes_df['language'] = ... # language detection code

# Get language counts
language_counts = notes_df['language'].value_counts()

# For better visualization, group less frequent languages
# Define a threshold (e.g., languages that make up less than 1% of the data)
threshold = len(notes_df) * 0.01
major_languages = language_counts[language_counts >= threshold]
other_count = language_counts[language_counts < threshold].sum()

# Create a new Series with major languages and 'Other'
if other_count > 0:
    plot_data = pd.concat([major_languages, pd.Series({'Other': other_count})])
else:
    plot_data = major_languages

# Calculate percentages for labels
total = plot_data.sum()
plot_data_percent = (plot_data / total * 100).round(1)
labels = [f'{lang}: {count} ({percent}%)' for lang, count, percent in 
          zip(plot_data.index, plot_data.values, plot_data_percent.values)]

# Create a colormap
colors = sns.color_palette('viridis', len(plot_data))

# Create the pie chart
plt.pie(plot_data, labels=None, colors=colors, autopct='', startangle=90, shadow=False)

# Add a circle at the center to make it a donut chart (optional)
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Add legend
plt.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))

# Add title and styling
plt.title('Language Distribution in Community Notes', fontsize=16, pad=20)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Save the figure
output_path = "../english_only/language_distribution.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Pie chart saved to {output_path}")

# Display the chart
plt.show()

# Print some statistics
print(f"Total notes analyzed: {len(notes_df)}")
print(f"Number of different languages detected: {len(language_counts)}")
print(f"Percentage of English notes: {plot_data_percent.get('en', 0)}%")


# In[ ]:




