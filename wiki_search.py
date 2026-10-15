import numpy as np
from termcolor import colored
import argparse


from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from WikiDataReader import *

#####################################################################
# Part 2:
# Define your new subclasses below!
#####################################################################

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

        query_vector = normalize(self.doc_vector_model.get_document_vector(query))
        
    
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
        sims = sims.toarray().flatten()

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
    
    args = parser.parse_args()

    articles = WikiArticles(args.data_dir, args.num_articles)
    # TODO replace the following line with an initialization of the appropriate WikiDocumentVectors subclass
    doc_vector_model = CountDocumentVectors()

    my_google = WikiSearchEngine(articles, doc_vector_model)
    my_google.train_vector_model()
    my_google.run_search(args.query)

if __name__ == "__main__":
    main()
