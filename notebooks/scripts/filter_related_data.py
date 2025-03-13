#!/usr/bin/env python
# coding: utf-8

# ### Filter related TSV files based on English Community Notes
# ### This script processes other TSV files to keep only data related to English notes
# ### Fixed the issue with different user ID column names across different files

# In[11]:


import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm


# In[12]:


# Define file paths
english_notes_path = "../data/english_only/english_notes-00000.tsv"
english_note_ids_path = "../data/english_only/english_note_ids.npy"

# Input files
ratings_path = "../data/raw/ratings-00003.tsv"
note_status_history_path = "../data/raw/noteStatusHistory-00000.tsv"
user_enrollment_path = "../data/raw/userEnrollment-00000.tsv"

# Output files
output_dir = "../data/english_only/"
output_ratings_path = os.path.join(output_dir, "english_ratings-00003.tsv")
output_note_status_history_path = os.path.join(output_dir, "english_noteStatusHistory-00000.tsv")
output_user_enrollment_path = os.path.join(output_dir, "english_userEnrollment-00000.tsv")


# In[13]:


# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


# In[14]:


# Load English note IDs
print("Loading English note IDs...")
if os.path.exists(english_note_ids_path):
    english_note_ids = np.load(english_note_ids_path, allow_pickle=True)
    print(f"Loaded {len(english_note_ids)} English note IDs")
else:
    # If the IDs file doesn't exist, read the English notes file and extract IDs
    print("English note IDs file not found. Reading from English notes file...")
    english_notes_df = pd.read_csv(english_notes_path, sep='\t')
    english_note_ids = english_notes_df['noteId'].tolist()
    print(f"Extracted {len(english_note_ids)} English note IDs")

# Convert to set for faster lookups
english_note_ids_set = set(english_note_ids)

# Get authors of English notes
print("Getting authors of English notes...")
english_notes_df = pd.read_csv(english_notes_path, sep='\t')
# Check user ID column name in notes file
notes_user_id_column = None
notes_possible_id_columns = ['participantId', 'noteAuthorParticipantId']
for col in notes_possible_id_columns:
    if col in english_notes_df.columns:
        notes_user_id_column = col
        print(f"Found user ID column in notes file: {notes_user_id_column}")
        break

if notes_user_id_column:
    english_authors = set(english_notes_df[notes_user_id_column].unique())
    print(f"Found {len(english_authors)} unique authors of English notes")
else:
    print("Warning: Could not find user ID column in notes file")
    english_authors = set()


# In[15]:


# Process ratings file
if os.path.exists(ratings_path):
    print(f"Processing ratings file: {ratings_path}")
    # Read the file in chunks to handle large files
    chunk_size = 100000
    chunks = []
    
    # Read first chunk to check column names
    first_chunk = next(pd.read_csv(ratings_path, sep='\t', chunksize=1))
    print(f"Ratings file columns: {list(first_chunk.columns)}")
    
    # Determine user ID column name
    ratings_user_id_column = None
    ratings_possible_id_columns = ['participantId', 'raterParticipantId', 'userId', 'ratingParticipantId']
    for col in ratings_possible_id_columns:
        if col in first_chunk.columns:
            ratings_user_id_column = col
            print(f"Found user ID column in ratings file: {ratings_user_id_column}")
            break
    
    if ratings_user_id_column is None:
        print("Warning: Could not find user ID column in ratings file, cannot extract raters")
    
    # Count total rows in original file
    total_ratings_count = 0
    for chunk in pd.read_csv(ratings_path, sep='\t', chunksize=chunk_size):
        total_ratings_count += len(chunk)
    print(f"Original ratings count: {total_ratings_count}")
    
    # Now filter the data
    for chunk in tqdm(pd.read_csv(ratings_path, sep='\t', chunksize=chunk_size)):
        # Filter rows where noteId is in english_note_ids_set
        filtered_chunk = chunk[chunk['noteId'].isin(english_note_ids_set)]
        chunks.append(filtered_chunk)
    
    # Combine all filtered chunks
    filtered_ratings_df = pd.concat(chunks)
    print(f"Filtered ratings count: {len(filtered_ratings_df)}")
    
    # Save filtered ratings
    filtered_ratings_df.to_csv(output_ratings_path, sep='\t', index=False)
    print(f"Saved filtered ratings to {output_ratings_path}")
    
    # Extract unique user IDs from ratings
    if ratings_user_id_column:
        english_raters = set(filtered_ratings_df[ratings_user_id_column].unique())
        print(f"Found {len(english_raters)} unique participants who rated English notes")
    else:
        english_raters = set()
        print("Could not extract rater IDs (column not found)")
else:
    print(f"Ratings file not found: {ratings_path}")
    english_raters = set()


# # Process note status history file

# In[9]:


# Process note status history file
if os.path.exists(note_status_history_path):
    print(f"Processing note status history file: {note_status_history_path}")
    chunk_size = 100000
    chunks = []
    
    # Count total rows in original file
    total_status_count = 0
    for chunk in pd.read_csv(note_status_history_path, sep='\t', chunksize=chunk_size):
        total_status_count += len(chunk)
    print(f"Original note status history count: {total_status_count}")
    
    # Now filter the data
    for chunk in tqdm(pd.read_csv(note_status_history_path, sep='\t', chunksize=chunk_size)):
        filtered_chunk = chunk[chunk['noteId'].isin(english_note_ids_set)]
        chunks.append(filtered_chunk)
    
    filtered_note_status_history_df = pd.concat(chunks)
    print(f"Filtered note status history count: {len(filtered_note_status_history_df)}")
    
    filtered_note_status_history_df.to_csv(output_note_status_history_path, sep='\t', index=False)
    print(f"Saved filtered note status history to {output_note_status_history_path}")
else:
    print(f"Note status history file not found: {note_status_history_path}")


# # Process note status history file

# In[17]:


# Process note status history file
if os.path.exists(note_status_history_path):
    print(f"Processing note status history file: {note_status_history_path}")
    
    # Read first chunk to check column names
    first_status_chunk = next(pd.read_csv(note_status_history_path, sep='\t', chunksize=1))
    print(f"Note status history file columns: {list(first_status_chunk.columns)}")
    
    # Determine user ID column name
    status_user_id_column = None
    status_possible_id_columns = ['participantId', 'noteAuthorParticipantId', 'authorId']
    for col in status_possible_id_columns:
        if col in first_status_chunk.columns:
            status_user_id_column = col
            print(f"Found user ID column in status history file: {status_user_id_column}")
            break
    
    chunk_size = 100000
    chunks = []
    
    # Count total rows in original file
    total_status_count = 0
    for chunk in pd.read_csv(note_status_history_path, sep='\t', chunksize=chunk_size):
        total_status_count += len(chunk)
    print(f"Original note status history count: {total_status_count}")
    
    # Now filter the data
    for chunk in tqdm(pd.read_csv(note_status_history_path, sep='\t', chunksize=chunk_size)):
        filtered_chunk = chunk[chunk['noteId'].isin(english_note_ids_set)]
        chunks.append(filtered_chunk)
    
    filtered_note_status_history_df = pd.concat(chunks)
    print(f"Filtered note status history count: {len(filtered_note_status_history_df)}")
    
    filtered_note_status_history_df.to_csv(output_note_status_history_path, sep='\t', index=False)
    print(f"Saved filtered note status history to {output_note_status_history_path}")
    
    # Extract authors from status history (if not already extracted from notes file)
    if status_user_id_column and not english_authors:
        status_authors = set(filtered_note_status_history_df[status_user_id_column].unique())
        print(f"Found {len(status_authors)} unique authors of English notes from status history")
        english_authors = status_authors
else:
    print(f"Note status history file not found: {note_status_history_path}")

# Merge all English-related user IDs
english_users = english_authors.union(english_raters) if english_raters else english_authors
print(f"Total unique English-involved users: {len(english_users)}")


# # Process user enrollment file

# In[18]:


# Process user enrollment file
if os.path.exists(user_enrollment_path):
    print(f"Processing user enrollment file: {user_enrollment_path}")
    
    # Read first chunk to check column names
    first_user_chunk = next(pd.read_csv(user_enrollment_path, sep='\t', chunksize=1))
    print(f"User enrollment file columns: {list(first_user_chunk.columns)}")
    
    # Determine user ID column name
    enrollment_user_id_column = None
    enrollment_possible_id_columns = ['participantId', 'userId', 'user_id']
    for col in enrollment_possible_id_columns:
        if col in first_user_chunk.columns:
            enrollment_user_id_column = col
            print(f"Found user ID column in user enrollment file: {enrollment_user_id_column}")
            break
    
    if enrollment_user_id_column is None:
        print("Warning: Could not find user ID column in enrollment file, cannot filter users")
        # Exit this part of processing if no ID column is found
    else:
        # Read and filter user enrollment data
        chunk_size = 100000
        chunks = []
        
        # Count total rows in original file
        total_enrollment_count = 0
        for chunk in pd.read_csv(user_enrollment_path, sep='\t', chunksize=chunk_size):
            total_enrollment_count += len(chunk)
        print(f"Original user enrollment count: {total_enrollment_count}")
        
        # Now filter the data
        for chunk in tqdm(pd.read_csv(user_enrollment_path, sep='\t', chunksize=chunk_size)):
            filtered_chunk = chunk[chunk[enrollment_user_id_column].isin(english_users)]
            chunks.append(filtered_chunk)
        
        filtered_user_enrollment_df = pd.concat(chunks) if chunks else pd.DataFrame()
        print(f"Filtered user enrollment count: {len(filtered_user_enrollment_df)}")
        
        filtered_user_enrollment_df.to_csv(output_user_enrollment_path, sep='\t', index=False)
        print(f"Saved filtered user enrollment to {output_user_enrollment_path}")
else:
    print(f"User enrollment file not found: {user_enrollment_path}")


# In[19]:


print("All processing complete!")


# In[20]:


# Save the set of English-involved user IDs for future reference
english_users_list = list(english_users) if 'english_users' in locals() else []
np.save(os.path.join(output_dir, "english_user_ids.npy"), english_users_list)
print(f"Saved {len(english_users_list)} English user IDs to english_user_ids.npy")

# Summary statistics
print("\nSummary:")
print(f"Total English notes: {len(english_note_ids)}")
if 'english_authors' in locals():
    print(f"Total unique authors of English notes: {len(english_authors)}")
if 'english_raters' in locals():
    print(f"Total unique raters of English notes: {len(english_raters)}")
if 'english_users' in locals():
    print(f"Total unique English-involved users: {len(english_users)}")


# ## find the cut-off date

# In[23]:


# convert timestampt to dates
filtered_note_status_history_df["timestampMillisOfMostRecentStatusChange"] = pd.to_datetime(filtered_note_status_history_df["timestampMillisOfMostRecentStatusChange"], unit="ms")

# find the max
latest_date = filtered_note_status_history_df["timestampMillisOfMostRecentStatusChange"].max()
print("Cut-off Date (timestampMillisOfMostRecentStatusChange):", latest_date)


# In[ ]:




