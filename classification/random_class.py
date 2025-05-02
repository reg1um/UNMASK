import random as rd

def predict(path, sentence): 
    """
    This function predicts the explicitness of a given sentence using a random number generator.
    It simulates the behavior of a machine learning model by returning a random integer between 1 and 3.
    """
    return rd.randint(1, 3)
  
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