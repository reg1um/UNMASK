# Feedforward Neural Network for Explicitness Classification
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score


class FFNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def main():
    data = pd.read_excel(
        "../data/GOAT.xlsx")

    # Split into train and test
    train_data, test_data = train_test_split(data, test_size=0.3)

    # Vectorize the data with TF-IDF
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_data["sentence"])
    test_vectors = vectorizer.transform(test_data["sentence"])

    # Convert to PyTorch tensors
    train_vectors = torch.Tensor(train_vectors.toarray())
    test_vectors = torch.Tensor(test_vectors.toarray())

    # Define the model
    input_size = train_vectors.shape[1]
    hidden_size = 200
    output_size = 3
    learning_rate = 0.001

    model = FFNetwork(input_size, hidden_size, output_size)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # CrossEntropyLoss expects labels to be 0, 1, 2
    train_data["explicitness"] = train_data["explicitness"] - 1

    # Train the model
    num_epochs = 80
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(train_vectors)
        explicitness = torch.tensor(train_data["explicitness"].values).long()
        loss_value = loss(outputs, explicitness)
        loss_value.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss_value.item()}")

    # Test the model
    outputs = model(test_vectors)
    _, predicted = torch.max(outputs, 1)

    # Adding 1 to match original labels
    predicted += 1

    accuracy = (predicted == test_data["explicitness"].values).sum(
    ).item() / len(test_data)
    print(f"Accuracy: {accuracy}")

    # Calculate F1 Score
    f1 = f1_score(test_data["explicitness"], predicted, average='weighted')
    print(f"F1 Score: {f1}")

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

    """
    sentence = "I don't like traffic"
    output = model(torch.Tensor(
        vectorizer.transform([sentence]).toarray()))
    probabilities = torch.softmax(output, dim=1)
    print("Probabilities:", probabilities)
    _, predicted = torch.max(output, 1)
    predicted += 1
    print("Predicted explicitness for \"", sentence, "\":", predicted.item())
    """


if __name__ == "__main__":
    main()
