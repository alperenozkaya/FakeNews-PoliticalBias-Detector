import pickle
from sklearn.decomposition import PCA

# Load the original embeddings
with open('../NLPClassifierTool/embeddings_dict.pkl', 'rb') as file:
    embeddings_dict = pickle.load(file)

# Convert embeddings to a list for PCA
embeddings_list = list(embeddings_dict.values())

pca = PCA(n_components=128)

reduced_embeddings = pca.fit_transform(embeddings_list)
reduced_embeddings_dict = {word: vec for word, vec in zip(embeddings_dict.keys(), reduced_embeddings)}

# Save the reduced embeddings
with open('../NLPClassifierTool/reduced_embeddings_dict.pkl', 'wb') as file:
    pickle.dump(reduced_embeddings_dict, file)


"""
nltk tokenized data (stopwords not removed) => bert => %95.8 acc
nltk tokenized data (stopwords removed) => bert => ???
bert tokenized data => bert => ???

"""
