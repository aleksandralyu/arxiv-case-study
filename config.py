# config.py - Central configuration for all notebooks

# Paths
RAW_DATA_PATH = 'data/raw/arxiv-metadata-oai-snapshot.json'
PROCESSED_DIR = 'data/processed/'
RESULTS_DIR = 'results/'

# TF-IDF Parameters
TFIDF_MAX_FEATURES = 1000
TFIDF_MIN_DF = 10
TFIDF_MAX_DF = 0.7
TFIDF_NGRAM_RANGE = (1, 2)

# Clustering Parameters
CLUSTERING_K_RANGE = [10, 15, 20, 25, 30, 35, 40, 45, 50]
CLUSTERING_BATCH_SIZE = 1024

# Random Seeds (for reproducibility)
RANDOM_STATE = 42