import pickle
from sklearn.decomposition import PCA
import numpy as np

embedding_name = 'turkish_berturk_128k'
# Load the original embeddings
with open(f'{embedding_name}.pkl', 'rb') as file:
    embeddings_dict = pickle.load(file)

# Convert embeddings to a list for PCA
embeddings_list = list(embeddings_dict.values())

pca = PCA(n_components=128)

reduced_embeddings = pca.fit_transform(embeddings_list)
reduced_embeddings_dict = {word: vec for word, vec in zip(embeddings_dict.keys(), reduced_embeddings)}

# Save the reduced embeddings
with open(f'../NLPClassifierTool/r_{embedding_name}.pkl', 'wb') as file:
    pickle.dump(reduced_embeddings_dict, file)


"""
nltk tokenized data (stopwords not removed) => bert => %95.8 acc
nltk tokenized data (stopwords removed) => bert => ???
bert tokenized data => bert => ???

"""
