import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from kneed import KneeLocator
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv('nlp-dataset.csv')  

# Question 1: Tokenization
def tokenize(text):
    return word_tokenize(text.lower())
data['tokens'] = data['text'].apply(tokenize)

# Question 2: Stemming
def stem(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
data['stemmed_tokens'] = data['tokens'].apply(stem)

# Question 3: Lemmatization
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
data['lemmatized_tokens'] = data['stemmed_tokens'].apply(lemmatize)

# Prepare text for clustering
processed_texts = [' '.join(tokens) for tokens in data['lemmatized_tokens']]

# Feature extraction with TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(processed_texts)

# Question 4: Clustering with K-means
inertia = []
max_k = min(len(processed_texts), 10)  # Limit the number of clusters
for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_tfidf)
    inertia.append(kmeans.inertia_)
kneedle = KneeLocator(range(1, max_k + 1), inertia, curve='convex', direction='decreasing')
optimal_k_tfidf = kneedle.elbow  # Determine the optimal number of clusters
if optimal_k_tfidf is None or optimal_k_tfidf < 2:  # Ensure there's at least 2 clusters
    optimal_k_tfidf = 2
kmeans_tfidf = KMeans(n_clusters=optimal_k_tfidf, random_state=42)
clusters_tfidf = kmeans_tfidf.fit_predict(X_tfidf)

# Question 5: Visualization using Bar Charts for most common words in each cluster
for i in range(optimal_k_tfidf):
    cluster_text = ' '.join([' '.join(words) for words in data['lemmatized_tokens'].iloc[clusters_tfidf == i]])
    wordcloud = WordCloud(width=800, height=400).generate(cluster_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for Cluster {i+1}')
    plt.axis('off')
    plt.show()
