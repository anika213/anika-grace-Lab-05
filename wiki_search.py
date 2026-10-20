import numpy as np
from termcolor import colored
import argparse
from sentence_transformers import SentenceTransformer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from WikiDataReader import *

#####################################################################
# Part 2:
# Define your new subclasses below!
#####################################################################

class SentenceTransformerDocumentVectors(WikiDocumentVectors):
    def __init__(self):
        """
        Initializing a SentenceTransformer instance with the model "sentence-transformers/all-mpnet-base-v2". This model will be used to encode documents into vector representations.
        """
        self.vectorizer = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.words_compressed = None

    def train(self, documents):
        """
        Trains a vector semantics model on the input documents,
        and returns a matrix containing the resulting vector
        representations of each document. 
        """
        documents = list(documents)
        term_doc_matrix = self.vectorizer.encode(documents)
        return term_doc_matrix

    def get_document_vector(self, new_doc):
        """
        Using the trained vector semantics model, computes and returns
        the vector representation of a new input document (i.e., a
        document that was not previously seen during training). 
        """
        print("new_doc", new_doc)
        return self.vectorizer.encode(new_doc)
    

class SVDDocumentVectors(WikiDocumentVectors):

    def __init__(self):
        
        """
        Initializing a Tfid insteance of TfidfVectorizer with max_vocab=10000 and max_df=0.7.
        Set up a place to store our V that gets output from svd
        """
        self.vectorizer = TfidfVectorizer(max_features=10000, max_df=0.7)
        # apply svd to the term-document matrix to reduce its dimensionality to 100
        self.words_compressed = None

    def train(self, documents):
        """
        Trains a vector semantics model on the input documents,
        and returns a matrix containing the resulting vector
        representations of each document represneting the outpiut of SVD 
        on the term-document matrix. The matrix is
        indexed the same as the input documents (i.e., the ith row
        is the vector representation of the ith document). We know 
        SVD decomposes into U @ s @ Vh, so we save Vh as self.Vh

        Parameters:
        documents: an iterable over documents, where each document is represented as a string

        Returns:
        A matrix containing the vector representations of each document, such that the ith row is the vector for the ith document
        """
        term_doc_matrix = self.vectorizer.fit_transform(documents)
        docs_compressed, s, words_compressed = svds(term_doc_matrix, k=100)
        words_compressed = words_compressed.transpose()
        self.words_compressed = words_compressed
        return docs_compressed

    def get_document_vector(self, new_doc):
        """
        Using the trained vector semantics model, computes and returns
        the vector representation of a new input document (i.e., a
        document that was not previously seen during training). This
        method can only be called after train has been called.

        Parameters:
        new_doc: a string representing the new document to process

        Returns:
        The vector representation of new_doc, as computed by the vector semantics model
        """
        transformed_doc = self.vectorizer.transform([new_doc])
        print("dimension of transformed_doc", transformed_doc.shape)
        print("dimension of self.vt", self.words_compressed.shape)
        concept_query = transformed_doc @ self.words_compressed
        return concept_query
        

class CountDocumentVectors(WikiDocumentVectors):
    
    def __init__(self):
        
        """
        Initializing a Tfid insteance of TfidfVectorizer with max_vocab=10000 and max_df=0.7.
        """
        self.vectorizer = TfidfVectorizer(max_features=10000, max_df=0.7)
    
        
    def train(self, documents):
        """
        Creates the term document matrix using the fit_transform function of the TfidfVectorizer instance for the documents in the corpus. The term document matrix is then returned.
        """
        return self.vectorizer.fit_transform(documents)
        
        
    def get_document_vector(self, new_doc):
        """ 
        Uses transform on the vectorizer model to transform new_doc into a term-document matrix. new_doc should be an iterable so we wrap it in a list. Model must be trained first since we only do transofrm (no fit). 
        """
        return self.vectorizer.transform([new_doc])




#####################################################################
# WikiSearchEngine
#####################################################################

class WikiSearchEngine():

    def __init__(self, articles, doc_vector_model):
        
        """
        This is a constructor which initalizes articles, word_vector_model and wiki_word_vectors attributes to the respective input values or None.
        """
        self.articles = articles
        self.doc_vector_model = doc_vector_model # object of type WikiDocumentVectors
        self.wiki_corpus_vectors = None
        
    def train_vector_model(self):
        '''
        Uses the train function in WikiDocumentVectors class to get vectors for the corpus (A matrix containing the vector representations of each document, such that the ith row is the vector for the ith document), then normalizes that matrix.
        '''
        print(colored("Training vector model. Please wait, this may take a while...", "magenta"))
        self.wiki_corpus_vectors = self.doc_vector_model.train(self.articles.get_abstracts())
        self.wiki_corpus_vectors = normalize(self.wiki_corpus_vectors)
        print(colored("Done!", "green"))

    def compute_rankings(self, query):
        
        """
        This function first checks if the train_vector_model function is enabled. If  yes, it then normalizes the query vector itself
        and runs cosine similarity between the query and all document vecotrs in the corpus. It then saes the simlarity scores to a list/numpy array called sims, indexed in the same order as self.wiki_corpus_vectors. It then returns the sims and the ranking of the sims in descending order.
        """
        if self.wiki_corpus_vectors is None:
            raise RuntimeError("Please run train_vector_model first!")

        query_vector = normalize(self.doc_vector_model.get_document_vector(query).reshape(1,-1))
        
    
        print("dimension of wiki_corpus_vectors matrix", self.wiki_corpus_vectors.shape)
    
        #the following line with code that computes the
        # cosine similarity between the query vector and all document
        # vectors in the self.wiki_corpus_vectors matrix, and saves
        # the similarity scores to a list or numpy array called sims,
        # indexed in the same order as self.wiki_corpus_vectors.

        # we tested dimension
        # dimension of query vector (1, 10000)
        # dimension of wiki_corpus_vectors matrix (121569, 10000)
        # so we have to transpose quety vector
        # sims is the dot product of the query vector and the wiki_corpus_vectors matrix normalized by the magnitude of the query vector
        # we already normalized, so both have magnitude 1; no need to divide! (yay!)

        # transpose the query vector to make it a column vector
        query_vector = query_vector.T
        sims = self.wiki_corpus_vectors @ query_vector
        if hasattr(sims, 'toarray'):
            sims = sims.toarray()
        sims = sims.flatten()

        return sims, np.argsort(sims)[::-1]

    def run_search(self, query, n_results=10):
        """
         Uses compute_rankings to get ranking & similarity for the whole input, then prints the ranking of the best n results by ranking (default 10 results) with the similarity scores. 
        """
        print(f"Top {n_results} search results for query {query}:")
        sims, ranking = self.compute_rankings(query)
        for rank, i in enumerate(ranking[:n_results]):
            print(f"{rank+1}. {self.articles.get_names()[i]} (similarity={sims[i]})")

def main():
    """
    This is the main function to run the entire pipeline. It first processes the command line arguments, then initializes the WikiArticles and WikiSearchEngine classes, and finally runs the search for the input query.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="Directory containing the Wikipedia corpus files")
    parser.add_argument("-q", "--query", type=str, metavar="Q", required=True,
                        help="The search query to run")
    parser.add_argument("-n", "--num_articles", type=int, metavar="N", default=None,
                        help="Only keep the first N articles parsed from the corpus files")
    
    parser.add_argument("-m", "--model", type=str, metavar="M", default="tfidf",
                        help="The vectorization model to use (default: tfidf). Options are: tfidf, svd, sent (for sentence transformer document embedding models)", choices=["tfidf", "svd", "sent"])
    
    
    args = parser.parse_args()

    articles = WikiArticles(args.data_dir, args.num_articles)
    if (args.model == "tfidf"):
        doc_vector_model = CountDocumentVectors()
    elif (args.model == "sent"):
        doc_vector_model = SentenceTransformerDocumentVectors()
    elif (args.model == "svd"):
        doc_vector_model = SVDDocumentVectors()

    my_google = WikiSearchEngine(articles, doc_vector_model)
    my_google.train_vector_model()
    my_google.run_search(args.query)

if __name__ == "__main__":
    main()
