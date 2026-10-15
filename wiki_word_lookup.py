import numpy as np
import argparse
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from WikiDataReader import *

#####################################################################
# Part 3:
# Define your new subclasses below!
#####################################################################

class SVDDocumentVectors(WikiDocumentVectors):

    def __init__(self):
        
        """
        Initializing a Tfid insteance of TfidfVectorizer with max_vocab=10000 and max_df=0.7.
        Set up a place to store our V that gets output from svd
        """
        self.vectorizer = TfidfVectorizer(max_features=10000, max_df=0.7)
        # apply svd to the term-document matrix to reduce its dimensionality to 100
        self.Vt = None

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
        u, s, vh = svds(term_doc_matrix, k=100)
        self.Vt = vh

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
        ## TODO: NOT DONE!!!! have to implement svd
        print("dimension of transformed_doc", transformed_doc.shape)




#####################################################################
# WikiWordLookup
#####################################################################

class WikiWordLookup():

    def __init__(self, articles, word_vector_model):
        """
        This is a constructor which initalizes articles, word_vector_model and wiki_word_vectors attributes to the respective input values or None.
        """
        self.articles = articles
        self.word_vector_model = word_vector_model # object of type WikiWordVector
        self.wiki_word_vectors = None

    def train_vector_model(self):
        '''
        Gets and wiki word vectors using the WikiWordVector class's train function (which returns a matrix where the ith row is the vector representation of the ith vocab word), then normalizes that matrix. 
        '''
        print("Training vector model. Please wait, this may take a while...")
        self.wiki_word_vectors = self.word_vector_model.train(self.articles.get_abstracts())
        self.wiki_word_vectors = normalize(self.wiki_word_vectors)
        print("Done!")

    def compute_rankings(self, input_word):
        """
        
        This function first checks if the train_vector_model function is enabled. If  yes, it then normalizes the word vector itself
        TODO
        """
        if self.wiki_word_vectors is None:
            raise RuntimeError("Please run train_vector_model first!")

        word_vector = normalize(self.word_vector_model.get_word_vector(input_word))
        if word_vector is None:
            return None, None

        # TODO: REPLACE the following line with code that computes the
        # cosine similarity between the vector for input_word and all
        # other word vectors in the self.wiki_word_vectors matrix, and
        # saves the similarity scores to a list or numpy array called 
        # sims, indexed in the same order as self.wiki_word_vectors.
        sims = []

        return sims, np.argsort(sims)[::-1][1:]

    def run_search(self, input_word, n_results=10):
        """
        Uses compute_rankings to get ranking & similarity for each input word, then prints the ranking of the best n results by ranking (default 10 results)
        """
        input_word = input_word.lower()
        print(f"Top {n_results} most similar words to {input_word}:")
        sims, ranking = self.compute_rankings(input_word)
        if ranking is None:
            print(f"'{input_word}' not found in the vocabulary!")
        else:
            for rank, i in enumerate(ranking[:n_results]):
                print(f"{rank+1}. {self.word_vector_model.get_vocabulary()[i]} (similarity={sims[i]})")

def main():
    
    """
    This is the main function to run the entire pipeline. It first processes the command line arguments, then initializes the WikiArticles and WikiWordLookup classes, and finally runs the search for the input word.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="Directory containing the Wikipedia corpus files")
    parser.add_argument("-w", "--word", type=str, metavar="W", required=True,
                        help="The word to look up")
    parser.add_argument("-n", "--num_articles", type=int, metavar="N", default=None,
                        help="Only keep the first N articles parsed from the corpus files")
    
    args = parser.parse_args()

    articles = WikiArticles(args.data_dir, args.num_articles)
    # TODO replace the following line with an initialization of the appropriate WikiWordVectors subclass
    word_vector_model = None

    word_lookup = WikiWordLookup(articles, word_vector_model)
    word_lookup.train_vector_model()
    word_lookup.run_search(args.word)

if __name__ == "__main__":
    main()
