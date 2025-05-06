from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re
import string
import random
from collections import defaultdict, Counter

def preprocess_function(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub("[" + string.punctuation + "]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class ImpliedStatementGenerator:
    def __init__(self, n=3):
        """Initialize the n-gram implied statement generator"""
        self.n = n
        self.sentence_vectorizer = CountVectorizer(ngram_range=(1, n))
        self.implied_vectorizer = CountVectorizer(ngram_range=(1, n))
        self.ngram_associations = defaultdict(list)
        self.start_words = []
        self.transitions = defaultdict(list)
        self.implied_statements = []
        
    def fit(self, df):
        """Train the model on sentence and implied statement pairs"""
        print("Training the n-gram model...")
        if 'clean_sentence' not in df.columns:
            df['clean_sentence'] = df['sentence'].apply(preprocess_function)
        if 'clean_implied' not in df.columns:
            df['clean_implied'] = df['implied_statement'].apply(preprocess_function)
        
        clean_sentences = df['clean_sentence'].tolist()
        clean_implied = df['clean_implied'].tolist()
        self.implied_statements = clean_implied
        
        self.sentence_vectorizer.fit(clean_sentences)
        self.implied_vectorizer.fit(clean_implied)
        
        sentence_ngrams = self.sentence_vectorizer.get_feature_names_out()
        implied_ngrams = self.implied_vectorizer.get_feature_names_out()
        
        print(f"Found {len(sentence_ngrams)} unique n-grams in sentences")
        print(f"Found {len(implied_ngrams)} unique n-grams in implied statements")
        
        X_sentence = self.sentence_vectorizer.transform(clean_sentences)
        
        for i in range(len(clean_sentences)):
            sentence_indices = X_sentence[i].nonzero()[1]
            sentence_grams = [sentence_ngrams[idx] for idx in sentence_indices]
            
            implied = clean_implied[i]
            
            words = implied.split()
            if words:
                self.start_words.append(words[0])
                
                for j in range(len(words) - 1):
                    self.transitions[words[j]].append(words[j + 1])
            
            for gram in sentence_grams:
                self.ngram_associations[gram].append(implied)
    
    def generate(self, sentence):
        """Generate an implied statement for a given sentence"""
        clean_sentence = preprocess_function(sentence)
        
        X = self.sentence_vectorizer.transform([clean_sentence])
            
        ngram_indices = X.nonzero()[1]
            
        if len(ngram_indices) == 0:
            return "No matching n-grams found"
            
        ngrams = [self.sentence_vectorizer.get_feature_names_out()[idx] for idx in ngram_indices]
            
        return self._generate_by_matching(ngrams)

    
    def _generate_by_matching(self, ngrams):
        """Generate implied statement by matching n-grams"""
        all_implications = []
        
        for gram in ngrams:
            if gram in self.ngram_associations:
                all_implications.extend(self.ngram_associations[gram])
        
        if all_implications:
            counter = Counter(all_implications)
            return counter.most_common(1)[0][0]
        
        if self.implied_statements:
            return random.choice(self.implied_statements)
        return "No matching implications found"

def main(path="../data/gen.csv"):
    """Main function to train and test the model"""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} sentence-implication pairs")
    
    df['clean_sentence'] = df['sentence'].apply(preprocess_function)
    df['clean_implied'] = df['implied_statement'].apply(preprocess_function)
    
    generator = ImpliedStatementGenerator(n=3)
    generator.fit(df)
    
    test_sentences = [
        "capitalism  like communism  is an enemy of the white race .",
        "The visa policy is allowing more immigrants to come into our country and take our jobs",
        "Pakistani people are moving into our area",
        "Democrats have invented new conspiracy theories",
        "Capitalism is destroying our society"
    ]
    
    print("\n=== Testing with sample sentences ===")
    for sentence in test_sentences:
        print(f"\nOriginal: \"{sentence}\"")
        
        matching_impl = generator.generate(sentence)
        
        print(f"Matching: \"{matching_impl}\"")
        print("-" * 60)

if __name__ == "__main__":
    main()
    
    
"""
N-gram Based Approach for Generating Implied Statements: Implementation and Analysis
Initial Approach
Our initial approach leveraged n-gram analysis to identify and generate implicit statements from explicit text. N-grams—contiguous sequences of n items from a given text—were used to build associations between explicit sentences and their implied meanings. We implemented three different generation methods:

Matching Method: This approach directly associated n-grams in input sentences with implied statements from the training data, returning the most commonly associated implication when a match was found.

Markov Method: This generative approach built a probabilistic model of word transitions in implied statements and used it to generate novel text influenced by n-grams from the input sentence.

Random Method: This fallback method simply returned randomly selected implied statements from the training data when no direct matches were available.

Observations from Testing
Testing the three methods on a dataset of sentences with known implicit biases revealed several patterns:

Matching Method Performance: Produced grammatically coherent and semantically relevant implications when matching n-grams were found in the training data. However, it tended to repeat the same implications for different inputs when the vocabulary overlap was similar (e.g., "immigrants should be deported" was generated for multiple different inputs).

Markov Method Limitations: While creative, the Markov approach often produced grammatically awkward or semantically incoherent statements like "lefties are substandard people are subversive to the" and "blacks are superior to the holocaust was to."

Random Method Inadequacies: This method produced complete sentences but with little or no relevance to the input text, making it unsuitable for insight generation.

Selection of the Matching Method
After analyzing the results, we selected the Matching method as our primary approach for the following reasons:

Semantic Coherence: The matching method consistently produced grammatically complete and semantically coherent statements.

Relevance to Input: When matches existed in the training data, the generated implications were directly related to the input sentence.

Predictability: The matching method provided consistent results based on the training data, making it more reliable for analysis.

Safety for Deployment: The method was less likely to generate grammatically incorrect or nonsensical content compared to the Markov method.

Attempted Improvements and Outcomes
We attempted several improvements to the basic matching approach, including:

TF-IDF Weighting: We tried replacing simple counting with TF-IDF (Term Frequency-Inverse Document Frequency) to better capture the importance of terms in context.

N-gram Length Weighting: We implemented a weighting scheme designed to give higher importance to longer n-grams than unigrams.

Fallback Mechanisms: We experimented with similarity-based fallbacks when direct n-gram matches failed.

However, these more complex approaches did not yield substantial improvements in our specific dataset context. The enhanced models either:

Continued producing similar results to the basic matching algorithm
Introduced new forms of bias in the selection process
Added computational complexity without corresponding performance gains
Given these outcomes, we decided to retain the original, simpler matching algorithm which offers:

Computational Efficiency: The basic algorithm processes inputs quickly with minimal resource requirements
Transparency: The direct n-gram matching process is easily interpretable and debuggable
Consistent Results: While limited in variety, the output implications are reliably coherent
Conclusion and Future Work
Our n-gram approach with basic matching provides a computationally efficient method for generating implied statements. The main challenge identified is the tendency to default to common implications when specific matches aren't found. This reflects a fundamental limitation of frequency-based n-gram approaches when working with limited training data.

Future work could focus primarily on:

Expanding the Training Dataset: A larger, more diverse corpus of sentence-implication pairs would allow for more varied and specific matches.

Domain-Specific Training: Creating separate models for different domains (politics, race, gender, etc.) might improve specificity of implications.

Exploration of Neural Approaches: While beyond our current scope, transformer-based models like BERT or GPT could potentially generate more nuanced implications with sufficient fine-tuning.

Despite its limitations, our basic n-gram matching approach serves as a functional baseline system for identifying potential implied meanings in text, especially when working within computational constraints.
"""