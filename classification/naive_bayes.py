from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import pandas as pd


def load_data(path):
    dataframe = pd.read_csv(path)
    return dataframe


def train_model(path):
    data = load_data(path)
    X = data["sentence"].values
    y = data["explicitness"] - 1
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_vectorized, y)

    # Save the model
    with open("../models/naive_bayes_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save the vectorizer
    vectorizer_path = "../models/naive_bayes_vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer


def predict(path, sentence):
    model, vectorizer = train_model(path)
    input_vector = vectorizer.transform([sentence])
    return model.predict(input_vector)[0] + 1


def main():
    path = "../data/GOAT.csv"
    sentences = {
        "This book belongs on the top shelf. 😊": 1,
        "I'm not sure if this book belongs on the top shelf... 🤔": 2,
        "Right, because heavy textbooks at eye level is such a brilliant idea!": 3,
    }
    for sentence, explicitness in sentences.items():
        prediction = predict(path, sentence)
        print(f"Prediction: {prediction} | Expected: {
              explicitness} | Sentence: {sentence}")


if __name__ == "__main__":
    main()
