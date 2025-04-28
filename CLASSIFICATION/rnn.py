# Reccurent Neural Network for Explicitness Classification

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.feature_extraction.text import TfidfVectorizer


class RNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out


def main():
    data = pd.read_excel(
        "../data/GOAT.xlsx")

    train_data, test_data = train_test_split(data, test_size=0.2)

    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_data["sentence"])
    test_vectors = vectorizer.transform(test_data["sentence"])

    train_vectors = torch.Tensor(train_vectors.toarray())
    test_vectors = torch.Tensor(test_vectors.toarray())

    input_size = train_vectors.shape[1]
    hidden_size = 10
    output_size = 3
    learning_rate = 0.01

    model = RNNetwork(input_size, hidden_size, output_size)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    train_data["explicitness"] = train_data["explicitness"] - 1

    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(train_vectors)
        explicitness = torch.tensor(train_data["explicitness"].values).long()
        loss_value = loss(outputs, explicitness)
        loss_value.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss_value.item()}")

    outputs = model(test_vectors)
    _, predicted = torch.max(outputs, 1)

    predicted += 1

    accuracy = (predicted == test_data["explicitness"].values).sum(
    ).item() / len(test_data)
    print(f"Accuracy: {accuracy}")

    sentence1 = "I don't like traffic."
    sentence2 = "There are so much cars on the road."
    sentence3 = "I LOVE waiting 3 hours in the traffic..."

    output = model(torch.Tensor(
        vectorizer.transform([sentence1]).toarray()))
    probabilities = torch.softmax(output, dim=1)
    print("Probabilities:", probabilities)
    _, predicted = torch.max(output, 1)
    predicted += 1
    print("Predicted explicitness for \"", sentence1, "\":", predicted.item())

    output = model(torch.Tensor(
        vectorizer.transform([sentence2]).toarray()))
    probabilities = torch.softmax(output, dim=1)
    print("Probabilities:", probabilities)
    _, predicted = torch.max(output, 1)
    predicted += 1
    print("Predicted explicitness for \"", sentence2, "\":", predicted.item())

    output = model(torch.Tensor(
        vectorizer.transform([sentence3]).toarray()))
    probabilities = torch.softmax(output, dim=1)
    print("Probabilities:", probabilities)
    _, predicted = torch.max(output, 1)
    predicted += 1
    print("Predicted explicitness for \"", sentence3, "\":", predicted.item())

    input("Write a sentence to classify: (q to exit)")
    while True:
        sentence = input()
        if sentence == "q":
            break
        output = model(torch.Tensor(
            vectorizer.transform([sentence]).toarray()))
        probabilities = torch.softmax(output, dim=1)
        print("Probabilities:", probabilities)
        _, predicted = torch.max(output, 1)
        predicted += 1
        print("Predicted explicitness for \"",
              sentence, "\":", predicted.item())


if __name__ == "__main__":
    main()
