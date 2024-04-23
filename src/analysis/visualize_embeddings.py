# SETUP --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import io
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import statsmodels.api as sm
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import CountVectorizer
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from gensim.models import Word2Vec

# MAGICKS --------------------------
sPathEmbeddings = "data/processed/summary_embeddings"
sNameEmbeddingA = "tangible_hair.pkl"
sNameEmbeddingB = "intangible_hair.pkl"

# FUNCTIONS --------------------------
# https://stackoverflow.com/questions/56369030/runtimeerror-attempting-to-deserialize-object-on-a-cuda-device
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def load_embeddings(sPathEmbeddings, sNameEmbeddingA, sNameEmbeddingB):
    
    # Load pickle file
    with open(sPathEmbeddings + "/" + sNameEmbeddingA, 'rb') as f:
        lEmbeddingsA = CPU_Unpickler(f).load()
    
    with open(sPathEmbeddings + "/" + sNameEmbeddingB, 'rb') as f:
        lEmbeddingsB = CPU_Unpickler(f).load()
        
    tEmbeddingA = torch.concatenate(lEmbeddingsA, dim=1)
    tEmbeddingB = torch.concatenate(lEmbeddingsB, dim=1)
    
    tEmbeddingA = tEmbeddingA.squeeze()
    tEmbeddingB = tEmbeddingB.squeeze()
        
    return tEmbeddingA, tEmbeddingB

class EmbeddingVisualizer:
    def __init__(self, lEmbeddingA, lEmbeddingB):
        self.lEmbeddingA = lEmbeddingA
        self.lEmbeddingB = lEmbeddingB
        
        
def visualize_correlations(similarity_matrix, sName, bEmbedding=False):

    plt.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f'Similarity Matrix ({sName})')
    plt.xticks(range(similarity_matrix.shape[0]), range(1, similarity_matrix.shape[0] + 1))
    plt.yticks(range(similarity_matrix.shape[0]), range(1, similarity_matrix.shape[0] + 1))
    plt.show()
    
def visualize_pca(embeddings, sName, vColorAssign=None):
    
    if vColorAssign is None:
        vColorAssign = np.zeros(embeddings.shape[0])
        vColorAssign[:embeddings.shape[0]//2] = 0
        vColorAssign[embeddings.shape[0]//2:] = 1
        vColorAssign = np.where(vColorAssign == 0, 'orange', 'green')
    
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)
    plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=vColorAssign)
    for i, (x, y) in enumerate(zip(pca_embeddings[:, 0], pca_embeddings[:, 1])):
        plt.text(x, y, str(i+1), color='black', fontsize=10)
    plt.title(f'PCA ({sName})')
    plt.show()

def visualize_tsne(embeddings, sName, iPerplexity, vColorAssign=None):
    
    if vColorAssign is None:
        vColorAssign = np.zeros(embeddings.shape[0])
        vColorAssign[:embeddings.shape[0]//2] = 0
        vColorAssign[embeddings.shape[0]//2:] = 1
        vColorAssign = np.where(vColorAssign == 0, 'orange', 'green')
    
    tsne = TSNE(n_components=2, perplexity=iPerplexity, method="exact")
    tsne_embeddings = tsne.fit_transform(embeddings)
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=vColorAssign)
    for i, (x, y) in enumerate(zip(tsne_embeddings[:, 0], tsne_embeddings[:, 1])):
        plt.text(x, y, str(i+1), color='black', fontsize=10)
    plt.title(f't-SNE ({sName})')
    plt.show()
    
def create_benchmarks(lClaims):
    
    # NOTE: This function de-means automatically!

    # Tokenize the sentences into words
    tokenized_sentences = [sentence.split() for sentence in lClaims]
    model = Word2Vec(tokenized_sentences, min_count=1)

    # Encode sentences to obtain sentence embeddings
    embeddings = np.array([np.mean([model.wv[word] for word in sentence if word in model.wv.key_to_index], axis=0) 
                        for sentence in tokenized_sentences])

    embeddings_demeaned = embeddings - np.mean(embeddings, axis=0)

    # Compute pairwise cosine similarity between sentence embeddings
    similarity_matrix_word2vec = np.corrcoef(embeddings_demeaned)

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the sentences and convert to tensors
    inputs = tokenizer(lClaims, padding=True, truncation=True, return_tensors="pt")

    # Pass the input tensors through the BERT model to obtain the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the embeddings for the [CLS] token, which represents the entire sentence
    # This index corresponds to the first token in each sequence
    embeddings = outputs.last_hidden_state[:, 0, :]

    # Convert embeddings to numpy array
    embeddings_demeaned = embeddings - torch.mean(embeddings, dim=0)

    # Compute pairwise cosine similarity between embeddings
    similarity_matrix_bert = torch.corrcoef(embeddings_demeaned).detach().numpy()

    vectorizer = CountVectorizer().fit_transform(lClaims)
    embeddings_demeaned = vectorizer.toarray() - np.mean(vectorizer.toarray(), axis=0)
    similarity_matrix_bow = np.corrcoef(embeddings_demeaned)

    return similarity_matrix_word2vec, similarity_matrix_bert, similarity_matrix_bow


def loo_regression(tEmbeddingsA, tEmbeddingsB, tEmbeddings_demeaned, sName, vColorAssign = None):

    lResults = list()
    for iFocal in range(tEmbeddings_demeaned.shape[0]):
        tEmbeddingFocal = tEmbeddings_demeaned[iFocal]

        if iFocal < tEmbeddingsA.shape[0]:
            tEmbeddingsA_mean = np.mean(tEmbeddingsA[np.arange(tEmbeddingsA.shape[0]) != iFocal], axis=0)
            tEmbeddingsB_mean = np.mean(tEmbeddingsB, axis=0)
        else:
            tEmbeddingsA_mean = np.mean(tEmbeddingsA, axis=0)
            tEmbeddingsB_mean = np.mean(tEmbeddingsB[np.arange(tEmbeddingsB.shape[0]) != iFocal], axis=0)

        # design
        X = np.concatenate([tEmbeddingsA_mean.reshape(-1, 1), tEmbeddingsB_mean.reshape(-1, 1)], axis=1)
        X = sm.add_constant(X)
        y = tEmbeddingFocal

        # fit the model
        model = sm.OLS(y, X)

        results = model.fit()
        lResults.append(results)

    aEstimates = np.concatenate([result.params.reshape(1, -1) for result in lResults], axis=0)

    # Split the data into two halves
    if vColorAssign is None:
        vColorAssign = np.zeros(tEmbeddings_demeaned.shape[0])
        vColorAssign[:tEmbeddings_demeaned.shape[0]//2] = 0
        vColorAssign[tEmbeddings_demeaned.shape[0]//2:] = 1
        vColorAssign = np.where(vColorAssign == 0, 'orange', 'green')
    
    # Create the scatterplot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(aEstimates[:, 1], aEstimates[:, 2], c=vColorAssign)
    plt.xlabel('beta_A')
    plt.ylabel('beta_B')
    plt.title(f'Leave-One-Out Regression ({sName}), orange = A, green = B')
    plt.legend()

    # Plot histogram of intercept
    plt.subplot(1, 2, 2)
    plt.hist(aEstimates[:, 0])
    plt.xlabel('Intercept')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Intercept; Mean = {np.mean(aEstimates[:, 0]):.3f}; Sd = {np.std(aEstimates[:, 0]):.3f}')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()
    
    return lResults
class SummaryEmbeddings:
    def __init__(self, lClaims):
        self.lClaims = lClaims
    
    def bert(self):
        # Load pre-trained BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # Tokenize the sentences and convert to tensors
        inputs = tokenizer(self.lClaims, padding=True, truncation=True, return_tensors="pt")

        # Pass the input tensors through the BERT model to obtain the embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the embeddings for the [CLS] token, which represents the entire sentence
        # This index corresponds to the first token in each sequence
        embeddings = outputs.last_hidden_state[:, 0, :]
        print(embeddings.shape)
        return embeddings.detach().numpy()
    
    def word2vec(self):
        # Tokenize the sentences into words
        tokenized_sentences = [sentence.split() for sentence in self.lClaims]
        model = Word2Vec(tokenized_sentences, min_count=1)

        # Encode sentences to obtain sentence embeddings
        embeddings = np.array([np.mean([model.wv[word] for word in sentence if word in model.wv.key_to_index], axis=0) 
                            for sentence in tokenized_sentences])
        
        return embeddings
        
    def bow(self):
        vectorizer = CountVectorizer().fit_transform(self.lClaims)
        return vectorizer.toarray()


def calc_similarity_matrix(tEmbeddings):
    similarity_matrix = np.corrcoef(tEmbeddings - np.mean(tEmbeddings, axis=0))
    return similarity_matrix


# MAIN --------------------------
if __name__ == "__main__":
    embeddingA, embeddingB = load_embeddings(sPathEmbeddings, sNameEmbeddingA, sNameEmbeddingB)


