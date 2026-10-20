import numpy as np
import argparse
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from WikiDataReader import *
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

#####################################################################
# Part 3:
# Define your new subclasses below!
#####################################################################


class Word2Vectors(WikiWordVectors):

    def __init__(self):
        """
        Initializing a Word2Vec model that we will train later.
        """
        self.model = None

        
    def train(self, documents):
        """
        Trains a vector semantics model on the input documents,
        using Word2Vec, with dimensiality 100 (so we have 100 concepts),
        window=5 so we expect the context to be within 5 words of our target,
        min_count=1 so we look at every word that appears at least once,
        and returns a matrix containing the resulting vector representations of 
        each word in the vocabulary (i.e., the ith row of the matrix should be 
        the vector representation of the ith vocabulary word)

        Parameters:
        documents: an iterable over documents, where each document is represented as a string

        Returns:
        A matrix where the ith row is the vector representation of the ith vocab word
        """
        sentences = [simple_preprocess(doc) for doc in documents]
        self.model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)
        return self.model.wv.vectors


    def get_word_index(self, word):
        """
        Returns the index of the word in the vocabulary that was learned during train, or -1 if the word isn't in the vocab

        Uses the word vectors' key to index, and returns -1 if no index found.
        """
        return self.model.wv.key_to_index.get(word, -1)

    def get_vocabulary(self):
        """
        Returns:
        The vocabulary that was learned during train, as a list of words
        """
        vocabulary = list(self.model.wv.key_to_index.keys())
        return vocabulary
    
    def get_word_vector(self, word):
        """
        Given a word that is in the vocabulary learned during train,
        looks up and returns the vector for that word. This
        method can only be called after train has been called.

        If a word is not in the vocab, we return None. 
        """
        if word not in self.model.wv.key_to_index:
            return None
        return self.model.wv[word]

class SVDWordVectors(WikiWordVectors):
    
    def __init__(self):
        """
        Initializing a Tfid insteance of TfidfVectorizer with max_vocab=10000 and max_df=0.7.
        Set up a place to store our V that gets output from svd
        """
        self.vectorizer = TfidfVectorizer(max_features=10000, max_df=0.7)
        # apply svd to the term-document matrix to reduce its dimensionality to 100
        self.words_compressed = None
        self.vocabulary = None

    def train(self, documents):
        """
        Trains a SVD vector semantics model on the input documents,
        and returns a tuple containing (1) a vocabulary learned
        from the input documents, and (2) a matrix containing the
        resulting vector representations of each word in the vocabulary
        (i.e., the ith row of the matrix should be the vector representation
        of the ith vocabulary word)

        Parameters:
        documents: an iterable over documents, where each document is represented as a string

        Returns:
        A matrix where the ith row is the vector representation of the ith vocab word
        """
        term_doc_matrix = self.vectorizer.fit_transform(documents)
        docs_compressed, s, words_compressed = svds(term_doc_matrix, k=100)
        words_compressed = words_compressed.transpose()
        self.words_compressed = words_compressed
        self.vocabulary = self.vectorizer.vocabulary_
        return words_compressed

    def get_word_index(self, word):
        """
        Returns:
        The index of the word in the vocabulary that was learned during train, or -1 if the word isn't in the vocab

        Must be called after model has been trained
        """
        if word not in self.vocabulary:
            return -1
        
        index = self.vocabulary[word]
        return index

    def get_vocabulary(self):
        """
        Returns:
        The vocabulary that was learned during train, as a list of words
        """
        # get keys from the vocabulary dictionary and convert to list
        return list(self.vectorizer.get_feature_names_out())
        
    
    def get_word_vector(self, word):
        """
        Given a word that is in the vocabulary learned during train,
        looks up and returns the vector for that word. This
        method can only be called after train has been called.

        Parameters:
        word: a string representing the word that we want the vector for

        Returns:
        The vector representation of word, or None if word is not in the vocab.
        """
        index = self.get_word_index(word)
        if index == -1:
            return None
        else:
            return self.words_compressed[index]


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
        Gets a wiki word vectors using the WikiWordVector class's train function (which returns a matrix where the ith row is the vector representation of the ith vocab word), then normalizes that matrix. 
        '''
        print("Training vector model. Please wait, this may take a while...")
        self.wiki_word_vectors = self.word_vector_model.train(self.articles.get_abstracts())
        self.wiki_word_vectors = normalize(self.wiki_word_vectors)
        print("Done!")

    def compute_rankings(self, input_word):
        """
        This function first checks if the train_vector_model function is enabled. If  yes, it then normalizes the word vector itself
        """
        
        if self.wiki_word_vectors is None:
            raise RuntimeError("Please run train_vector_model first!")

        word_vector = self.word_vector_model.get_word_vector(input_word)
        if word_vector is None:
            return None, None
        else:
            word_vector = normalize(word_vector.reshape(1,-1))

        query_vector = word_vector.T
        sims = self.wiki_word_vectors @ query_vector
        if hasattr(sims, 'toarray'):
            sims = sims.toarray()
        sims = sims.flatten()

        return sims, np.argsort(sims)[::-1]


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
    parser.add_argument("-v", "--wordvec", action="store_true",
                        help="Use word2vec word vectors instead of SVD")
    
    
    args = parser.parse_args()

    articles = WikiArticles(args.data_dir, args.num_articles)
    
    if args.wordvec:
        word_vector_model = Word2Vectors()
    else:
        word_vector_model = SVDWordVectors()

    word_lookup = WikiWordLookup(articles, word_vector_model)
    word_lookup.train_vector_model()
    word_lookup.run_search(args.word)

if __name__ == "__main__":
    main()
