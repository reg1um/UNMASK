import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from gensim.models import KeyedVectors
from gensim.downloader import load as gensim_load

# Load Word2Vec model from binary file


def load_word2vec_model():
    return gensim_load("glove-wiki-gigaword-50")

# Convert a sentence to its average Word2Vec vector


def vectorize_sentence(sentence, w2v_model):
    words = [w for w in sentence.split() if w in w2v_model]
    if not words:
        return np.zeros(w2v_model.vector_size)
    return np.mean([w2v_model[w] for w in words], axis=0)


def load_data(data_path, w2v_model):
    dataframe = pd.read_csv(data_path)
    X_sentences = dataframe["sentence"].values
    y = dataframe["explicitness"] - 1
    X_vectors = np.array([vectorize_sentence(s, w2v_model)
                         for s in X_sentences])
    return X_vectors, y


def train_model(data_path):
    w2v_model = load_word2vec_model()
    X, y = load_data(data_path, w2v_model)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, w2v_model


def predict(model, w2v_model, sentence):
    vec = vectorize_sentence(sentence, w2v_model).reshape(1, -1)
    return model.predict(vec)[0] + 1


def main():
    data_path = "../data/GOAT.csv"
    model, w2v_model = train_model(data_path)

    sentences = {
        "This book belongs on the top shelf. 😊": 1,
        "I'm not sure if this book belongs on the top shelf... 🤔": 2,
        "Right, because heavy textbooks at eye level is such a brilliant idea!": 3
    }

    for sentence, expected in sentences.items():
        prediction = predict(model, w2v_model, sentence)
        print(f"Prediction: {prediction} | Expected: {
              expected} | Sentence: {sentence}")

    # Save the model
    with open("../models/logistic_regression_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save the vectorizer
    vectorizer_path = "../models/logistic_regression_vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(w2v_model, f)


if __name__ == "__main__":
    main()
