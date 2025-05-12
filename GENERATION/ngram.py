from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re
import string
import random
from collections import defaultdict, Counter
import pickle
import os

def preprocess_function(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub("[" + string.punctuation + "]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class ImpliedStatementGenerator:
    def __init__(self, n=3):
        self.n = n
        self.sentence_vectorizer = CountVectorizer(ngram_range=(1, n))
        self.implied_vectorizer = CountVectorizer(ngram_range=(1, n))
        self.ngram_associations = defaultdict(list)
        self.start_words = []
        self.transitions = defaultdict(list)
        self.implied_statements = []
        
    def fit(self, df):
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
        clean_sentence = preprocess_function(sentence)
        
        X = self.sentence_vectorizer.transform([clean_sentence])
            
        ngram_indices = X.nonzero()[1]
            
        if len(ngram_indices) == 0:
            return "No matching n-grams found"
            
        ngrams = [self.sentence_vectorizer.get_feature_names_out()[idx] for idx in ngram_indices]
            
        return self._generate_by_matching(ngrams)

    
    def _generate_by_matching(self, ngrams):
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
    
    os.makedirs("models", exist_ok=True)
    
    with open("models/ngram_model.pkl", "wb") as f:
        pickle.dump(generator, f)
    
    print("Model saved to models/ngram_model.pkl")
    
    test_sentences = [
        "capitalism  like communism  is an enemy of the white race .",
        "The visa policy is allowing more immigrants to come into our country and take our jobs",
        "Pakistani people are moving into our area",
        "Democrats have invented new conspiracy theories",
        "Capitalism is destroying our society",
        "if you are white in that state watch yourself !",
    ]
    
    print("\n=== Testing with sample sentences ===")
    for sentence in test_sentences:
        print(f"\nOriginal: \"{sentence}\"")
        
        matching_impl = generator.generate(sentence)
        
        print(f"Matching: \"{matching_impl}\"")
        print("-" * 60)

if __name__ == "__main__":
    main()
    
    