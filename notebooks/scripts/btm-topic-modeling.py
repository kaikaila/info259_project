#!/usr/bin/env python
# coding: utf-8

# # BTM Topic Modeling for Twitter Community Notes
# 
# This notebook implements Biterm Topic Model (BTM) on Twitter Community Notes data. BTM is particularly effective for short texts, making it well-suited for analyzing Twitter-related content.
# 
# ## Why BTM for Twitter Data?
# - Works well with short texts (tweets, notes)
# - Handles sparse word co-occurrence patterns
# - Models word-pair (biterm) occurrences across the corpus
# - Often outperforms LDA for short text topic discovery

# In[1]:


# Install required packages
get_ipython().system('pip install pandas numpy scikit-learn matplotlib seaborn wordcloud btm tqdm nltk')


# In[2]:


import sys
print(sys.executable)


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# For text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# For BTM
from btm import oBTM

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


# ## 1. Load and Explore Data

# In[ ]:


# Set paths - update these to match your environment
# For Google Colab, you might want to mount Google Drive first
# from google.colab import drive
# drive.mount('/content/drive')
# SAMPLE_PATH = "/content/drive/MyDrive/path/to/your/sample_notes-00000.tsv"

# For local use
SAMPLE_PATH = os.path.expanduser("~/Desktop/samples/sample_notes-00000.tsv")

# Load the sample data
df = pd.read_csv(SAMPLE_PATH, sep='\t')
print(f"Loaded data shape: {df.shape}")

# Display the first few rows
df.head()


# In[ ]:


# Check if 'summary' column exists
if 'summary' in df.columns:
    text_column = 'summary'
elif 'noteText' in df.columns:
    text_column = 'noteText'
else:
    # Find potential text columns (columns with string type and longer average length)
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            non_null = df[col].dropna()
            if len(non_null) > 0:
                avg_len = non_null.astype(str).str.len().mean()
                if avg_len > 50:  # Assume text fields have avg length > 50
                    text_columns.append((col, avg_len))
    
    text_columns = sorted(text_columns, key=lambda x: x[1], reverse=True)
    if text_columns:
        text_column = text_columns[0][0]
        print(f"Using '{text_column}' as the text column (avg length: {text_columns[0][1]:.1f})")
    else:
        raise ValueError("No suitable text column found in the data")

# Show some sample texts
print(f"\nSample texts from '{text_column}' column:")
for i, text in enumerate(df[text_column].dropna().head(3)):
    print(f"\nText {i+1}:\n{text}")


# In[ ]:


# Basic statistics of text length
text_lengths = df[text_column].dropna().astype(str).str.len()
print(f"Text length statistics:")
print(f"Mean: {text_lengths.mean():.1f} characters")
print(f"Median: {text_lengths.median():.1f} characters")
print(f"Min: {text_lengths.min()} characters")
print(f"Max: {text_lengths.max()} characters")

# Plot text length distribution
plt.figure(figsize=(12, 6))
sns.histplot(text_lengths, bins=50, kde=True)
plt.title(f"Distribution of '{text_column}' Text Lengths")
plt.xlabel("Number of Characters")
plt.ylabel("Count")
plt.show()


# ## 2. Text Preprocessing

# In[ ]:


def preprocess_text(text):
    """Preprocess text for topic modeling"""
    if pd.isna(text):
        return []
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Filter out very short words
    tokens = [token for token in tokens if len(token) > 2]
    
    return tokens


# In[ ]:


# Preprocess all texts
print("Preprocessing texts...")
docs = df[text_column].dropna().reset_index(drop=True)
tokenized_docs = [preprocess_text(doc) for doc in tqdm(docs)]

# Filter out empty documents
tokenized_docs = [doc for doc in tokenized_docs if len(doc) > 0]
print(f"Retained {len(tokenized_docs)} non-empty documents after preprocessing")

# Show a sample of tokenized documents
print("\nSample of preprocessed documents:")
for i, doc in enumerate(tokenized_docs[:3]):
    print(f"Document {i+1}: {' '.join(doc[:20])}{'...' if len(doc) > 20 else ''}")


# In[ ]:


# Vocabulary statistics
all_words = [word for doc in tokenized_docs for word in doc]
unique_words = set(all_words)
print(f"Total words: {len(all_words)}")
print(f"Unique words: {len(unique_words)}")

# Word frequency
from collections import Counter
word_freq = Counter(all_words)
print("\nTop 20 most frequent words:")
for word, count in word_freq.most_common(20):
    print(f"{word}: {count}")


# ## 3. BTM Topic Modeling

# In[ ]:


# Convert tokenized documents to format required by BTM
texts_for_btm = [' '.join(doc) for doc in tokenized_docs]

# Create vectorizer to get vocabulary
vec = CountVectorizer(stop_words='english')
X = vec.fit_transform(texts_for_btm)
vocabulary = vec.get_feature_names_out()
print(f"Vocabulary size for BTM: {len(vocabulary)}")

# Function to train BTM and evaluate topic coherence
def train_evaluate_btm(texts, num_topics, iterations=100):
    print(f"Training BTM with {num_topics} topics...")
    
    # Initialize BTM model
    btm = oBTM(num_topics=num_topics, V=vocabulary)
    
    # Fit the model
    X_btm = btm.fit_transform(texts, iterations=iterations)
    
    # Get topic-word distributions
    topics = btm.transform_topics()
    
    return btm, X_btm, topics


# In[ ]:


# Try different numbers of topics
topic_counts = [5, 10, 15, 20]
models = {}

for n_topics in topic_counts:
    btm_model, doc_topic_matrix, topic_word_matrix = train_evaluate_btm(texts_for_btm, n_topics, iterations=50)
    models[n_topics] = {
        'model': btm_model,
        'doc_topics': doc_topic_matrix,
        'topics': topic_word_matrix
    }


# In[ ]:


# Functions to display topic results
def print_topics(topics, n_words=15):
    """Print top words for each topic"""
    for i, topic_dist in enumerate(topics):
        # Get the top n_words for the topic
        top_words_idx = topic_dist.argsort()[-n_words:][::-1]
        top_words = [vocabulary[idx] for idx in top_words_idx]
        print(f"Topic #{i+1}: {', '.join(top_words)}")
        print()

def plot_wordclouds(topics, n_words=100):
    """Plot word clouds for each topic"""
    n_topics = len(topics)
    n_cols = 2
    n_rows = (n_topics + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, topic_dist in enumerate(topics):
        if i < len(axes):
            # Create word-weight dictionary for word cloud
            word_weights = {}
            for word_idx, weight in enumerate(topic_dist):
                word = vocabulary[word_idx]
                word_weights[word] = weight
            
            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, 
                                 background_color='white',
                                 max_words=n_words,
                                 relative_scaling=0.5,
                                 colormap='viridis').generate_from_frequencies(word_weights)
            
            # Plot
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'Topic #{i+1}')
            axes[i].axis('off')
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


# ## 4. Analyze Results for Different Topic Counts

# In[ ]:


# For each model, display topics and word clouds
for n_topics, model_data in models.items():
    print(f"\n{'=' * 80}")
    print(f"RESULTS FOR {n_topics} TOPICS")
    print(f"{'=' * 80}\n")
    
    print("Top words for each topic:")
    print_topics(model_data['topics'])
    
    print("Word clouds for each topic:")
    plot_wordclouds(model_data['topics'])


# ## 5. Choose Best Model and Analyze Document-Topic Distribution

# In[ ]:


# Choose the best model (replace with your chosen number of topics)
best_n_topics = 10  # You can change this after reviewing results
best_model = models[best_n_topics]

# Get document-topic distributions
doc_topic_dist = best_model['doc_topics']

# Analyze topic prevalence
topic_prevalence = doc_topic_dist.mean(axis=0)
topic_ids = np.arange(1, best_n_topics + 1)

# Plot topic prevalence
plt.figure(figsize=(12, 6))
bars = plt.bar(topic_ids, topic_prevalence, color='skyblue')
plt.xlabel('Topic ID')
plt.ylabel('Average Topic Probability')
plt.title('Topic Prevalence in the Community Notes Corpus')
plt.xticks(topic_ids)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# ## 6. Examining Documents for Each Topic
# 
# Now let's look at some example documents that are strongly associated with each topic.

# In[ ]:


def get_top_docs_for_topic(topic_idx, doc_topic_dist, texts, top_n=5):
    """Get the top documents for a specific topic"""
    # Sort documents by their probability for the given topic
    topic_probs = doc_topic_dist[:, topic_idx]
    top_doc_indices = topic_probs.argsort()[-top_n:][::-1]
    
    return [(texts[idx], topic_probs[idx]) for idx in top_doc_indices]

# Display example documents for each topic
for topic_idx in range(best_n_topics):
    print(f"\n{'=' * 80}")
    print(f"EXAMPLE DOCUMENTS FOR TOPIC #{topic_idx+1}")
    print(f"{'=' * 80}\n")
    
    # Get top words for this topic
    topic_dist = best_model['topics'][topic_idx]
    top_words_idx = topic_dist.argsort()[-10:][::-1]
    top_words = [vocabulary[idx] for idx in top_words_idx]
    print(f"Top words: {', '.join(top_words)}\n")
    
    # Get top documents for this topic
    top_docs = get_top_docs_for_topic(topic_idx, doc_topic_dist, docs, top_n=3)
    
    # Display top documents
    for i, (doc, prob) in enumerate(top_docs):
        print(f"Document {i+1} (Topic probability: {prob:.4f}):\n{doc}\n")


# ## 7. Manual Topic Interpretation
# 
# Based on the word distributions and example documents, we can now interpret what each topic represents. This is a manual step that requires human judgment.

# In[ ]:


# Example interpretation (replace with your own interpretations after analyzing results)
topic_interpretations = {
    1: "Topic 1: [Your interpretation here]",
    2: "Topic 2: [Your interpretation here]",
    3: "Topic 3: [Your interpretation here]",
    # Add interpretations for all topics...
    best_n_topics: f"Topic {best_n_topics}: [Your interpretation here]"
}

# Display interpretations
print("TOPIC INTERPRETATIONS:")
for topic_id, interpretation in topic_interpretations.items():
    print(interpretation)


# ## 8. Save Model Results

# In[ ]:


# Create results directory
results_dir = os.path.expanduser("~/Desktop/topic_model_results")
os.makedirs(results_dir, exist_ok=True)

# Save topic-word distributions
topic_word_df = pd.DataFrame(best_model['topics'], 
                             columns=vocabulary)
topic_word_df.to_csv(f"{results_dir}/topic_word_dist_{best_n_topics}_topics.csv")

# Save document-topic distributions
doc_topic_df = pd.DataFrame(doc_topic_dist, 
                           columns=[f"Topic_{i+1}" for i in range(best_n_topics)])
doc_topic_df.to_csv(f"{results_dir}/doc_topic_dist_{best_n_topics}_topics.csv")

# Save top words for each topic
with open(f"{results_dir}/top_words_{best_n_topics}_topics.txt", "w") as f:
    for i, topic_dist in enumerate(best_model['topics']):
        top_words_idx = topic_dist.argsort()[-20:][::-1]
        top_words = [vocabulary[idx] for idx in top_words_idx]
        f.write(f"Topic #{i+1}: {', '.join(top_words)}\n\n")

print(f"Results saved to {results_dir}")


# ## 9. Conclusion
# 
# In this notebook, we've applied the Biterm Topic Model (BTM) to Twitter Community Notes data. BTM is particularly well-suited for short text analysis, making it an appropriate choice for social media content. The model identified several coherent topics in the data.
# 
# Key observations:
# 1. The notes tend to cluster around [number] main themes (based on your interpretation)
# 2. The most prevalent topics appear to be related to [your insights here]
# 3. Examples of clear topics include [your examples here]
# 
# Next steps:
# 1. Further refine the preprocessing steps if needed
# 2. Consider comparing BTM results with other topic models (LDA, NMF)
# 3. Apply the topic model to the entire dataset (not just the sample)
# 4. Analyze how topics have evolved over time if timestamp data is available
