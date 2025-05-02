from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def load_data(path):
  dataframe = pd.read_csv(path)
  return dataframe

def train_model(path):
  data = load_data(path)
  X = data["sentence"].values
  y = data["explicitness"] - 1
  vector = CountVectorizer()
  X = vector.fit_transform(X)
  model = LogisticRegression()
  model.fit(X, y)
  return model, vector

def predict(path, sentence):
  model, vector = train_model(path)
  input_d = vector.transform([sentence])
  return model.predict(input_d)[0] + 1

def main():
  path = "../data/GOAT.csv"
  sentences = {
    "This book belongs on the top shelf. 😊":1,
    "I'm not sure if this book belongs on the top shelf... 🤔":2,
    "Right, because heavy textbooks at eye level is such a brilliant idea!":3,
  }
  for sentence, explicitness in sentences.items():
    prediction = predict(path, sentence)
    print(f"The explicitness of the sentence is: {prediction}")
  
if __name__ == "__main__":
  main()