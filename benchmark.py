import os
import numpy as np
import random as rd
import torch
import torch.nn as nn
import pandas as pd
import pickle
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from classification.feedforward_network import FFNetwork
from classification.recurrent_nn import RNNetwork
from classification.pretrained_transformers import predict as transformer_predict
from classification.trained_transformers import predict as trained_transformer_predict
import tabulate as tb


DEBUG = True
OUTPUT_SIZE = 3


# Used for logistic regression as a way to simulate the vectorization process
def sentences_to_avg_vectors(sentences, keyed_vectors_model, embedding_dim):
    vectors = []
    for sentence in sentences:
        if not isinstance(sentence, str):
            vectors.append(np.zeros(embedding_dim))
            continue

        # Use the same logic as your training script for finding words
        words_in_vocab = [w for w in sentence.lower().split()
                          if w in keyed_vectors_model]

        if not words_in_vocab:
            vectors.append(np.zeros(embedding_dim))
        else:
            # Calculate mean of vectors for words found in the model
            vectors.append(np.mean([keyed_vectors_model[w]
                           for w in words_in_vocab], axis=0))
    return np.array(vectors)


def main():

    # INFO: To change if models are being modified
    models_config = {
        "feedforward": {
            "class": FFNetwork,
            "model_path": "models/feedforward_model.pth",
            "vectorizer_path": "models/feedforward_vectorizer.pkl",
            "type": "pytorch-tfidf",
            "hidden_size": 200,
        },
        "recurrent": {
            "class": RNNetwork,
            "model_path": "models/recurrent_model.pth",
            "vectorizer_path": "models/recurrent_vectorizer.pkl",
            "type": "pytorch-tfidf",
            "hidden_size": 200,
        },
        "logistic_regression": {
            "class": None,
            "model_path": "models/logistic_regression_model.pkl",
            "vectorizer_path": "models/logistic_regression_vectorizer.pkl",
            "hidden_size": None,
            "type": "sklearn-word2vec",
            "embedding_dim": 50
        },
        "naive_bayes": {
            "class": None,
            "model_path": "models/naive_bayes_model.pkl",
            "vectorizer_path": "models/naive_bayes_vectorizer.pkl",
            "hidden_size": None,
            "type": "sklearn-tfidf",
        },
        "random": {
            "class": None,
            "model_path": None,
            "vectorizer_path": None,
            "hidden_size": None,
            "type": "random",
        },
        "pretrained_transformer": {
            "class": None,
            "model_path": None,
            "vectorizer_path": None,
            "hidden_size": None,
            "type": "pt_transformer",
        },
        "trained_transformer": {
            "class": None,
            "model_path": "classification/trained_transformers",
            "vectorizer_path": None,
            "hidden_size": None,
            "type": "transformer",
        }
    }

    # INFO: Add any new dataset to benchmark here
    datasets = {
        "GOAT": "data/GOAT.xlsx",
        "Benchmark Data": "data/benchmark_data.csv",
    }

    # Stats struct to save stats for each model and dataset
    stats = {}
    for model_name_key in models_config.keys():
        stats[model_name_key] = {
            "dataset": [],
            "accuracy": [],
            "f1": [],
            "precision": [],
            "recall": []
        }

    # Start the benchmarking for each model and dataset

    for dataset_name, dataset_path in datasets.items():
        print("" + "=" * 20, end="")
        print(f"Testing dataset: {dataset_name}", end="")
        print("" + "=" * 20)

        # Small sanity check
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_name} not found. Skipping...")
            continue

        # The dataset will be used as a test set for benchmarking comparison
        if dataset_path.endswith('.csv'):
            test_data = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.xlsx'):
            test_data = pd.read_excel(dataset_path)

        for model_name, config in models_config.items():

            print("" + "=" * 10, end="")
            print(f"Testing model: {model_name}", end="")
            print("" + "=" * 10)

            model_instance = None
            test_vectors = None
            input_size = 0

            # RNN and FFN with PyTorch TF-IDF
            if config["type"] == "pytorch-tfidf":
                if "vectorizer_path" not in config or not os.path.exists(config["vectorizer_path"]):
                    print(f"Vectorizer for {
                          model_name} not found. Skipping...")
                    continue

                with open(config["vectorizer_path"], 'rb') as f:
                    vectorizer = pickle.load(f)

                input_size = len(vectorizer.vocabulary_)

                if DEBUG:
                    print(f"[DEBUG] Vectorizer vocabulary size: {
                          len(vectorizer.vocabulary_)}")

                test_vectors_sparse = vectorizer.transform(
                    test_data["sentence"])
                test_vectors_tensor = torch.Tensor(
                    test_vectors_sparse.toarray())

                # Load the model
                modelClass = config["class"]
                model_instance = modelClass(input_size,
                                            config["hidden_size"], OUTPUT_SIZE)
                state_dict = torch.load(
                    config["model_path"], map_location=torch.device('cpu'), weights_only=True)
                model_instance.load_state_dict(state_dict)
                model_instance.eval()

                # Perform inference
                with torch.no_grad():
                    outputs = model_instance(test_vectors_tensor)

                _, predicted = torch.max(outputs, 1)
                predicted += 1

            # WORD2VEC LOGISTIC REGRESSION
            elif config["type"] == "sklearn-word2vec":
                if "vectorizer_path" not in config or not os.path.exists(config["vectorizer_path"]):
                    print(f"Word2Vec Vectorizer path not configured or file '{config.get(
                        'vectorizer_path')}' not found for model '{model_name}'. Skipping.")
                    continue
                try:
                    with open(config["vectorizer_path"], 'rb') as f:
                        # This is the gensim KeyedVectors model
                        word2vec_keyedvectors = pickle.load(f)

                    # Determine the embedding_dim that matches the model
                    current_embedding_dim = config.get("embedding_dim")

                    # If the embedding_dim is not set in the config, we need to infer it
                    # By default, the embedding_dim is set at 50 (what's set in the "glove-wiki-gigaword-50" gensim model
                    # used in logistic_regression.py so this doesn't really matter for us --')
                    if hasattr(word2vec_keyedvectors, 'vector_size'):
                        inferred_dim = word2vec_keyedvectors.vector_size
                        if current_embedding_dim is not None and current_embedding_dim != inferred_dim:
                            print(f"[WARN] Model '{model_name}': Configured embedding_dim ({current_embedding_dim}) "
                                  f"differs from inferred KeyedVectors.vector_size ({inferred_dim}). Using inferred value: {inferred_dim}.")
                        elif current_embedding_dim is None and DEBUG:
                            print(f"[DEBUG] Model '{
                                  model_name}': Inferred embedding_dim from KeyedVectors.vector_size: {inferred_dim}.")
                        current_embedding_dim = inferred_dim

                    if current_embedding_dim is None:
                        print(f"[ERROR] Model '{model_name}': Could not determine embedding_dim. "
                              "Please provide it in models_config or ensure vectorizer has 'vector_size' attribute. Skipping.")
                        continue

                    if DEBUG and config.get("embedding_dim") is None:
                        print(f"[DEBUG] Model '{model_name}': Successfully using embedding_dim: {
                              current_embedding_dim}")

                    # Use the function that matches the training script's vectorization logic
                    input_features_for_model = sentences_to_avg_vectors(
                        test_data["sentence"],
                        word2vec_keyedvectors,
                        current_embedding_dim
                    )
                    if DEBUG:
                        print(f"[DEBUG] Model '{model_name}': Word2Vec features generated. Shape: {
                              input_features_for_model.shape}")

                    with open(config["model_path"], 'rb') as f:
                        # This is the scikit-learn LogisticRegression model
                        model_instance = pickle.load(f)

                except Exception as e:
                    print(
                        f"[ERROR] Error setting up scikit-learn Word2Vec model '{model_name}': {e}. Skipping.")
                    continue

                raw_predictions = model_instance.predict(
                    input_features_for_model)
                predicted = raw_predictions + 1
                if DEBUG:
                    print(f"[DEBUG] Model '{model_name}': Raw sklearn predictions sample: {
                          raw_predictions[:5]}, Adjusted: {predicted[:5]}")

            # SKLEARN NAIVE BAYES
            elif config["type"] == "sklearn-tfidf":
                if "vectorizer_path" not in config or not os.path.exists(config["vectorizer_path"]):
                    print(f"Vectorizer for {
                          model_name} not found. Skipping...")
                    continue

                with open(config["vectorizer_path"], 'rb') as f:
                    vectorizer = pickle.load(f)

                test_vectors_sparse = vectorizer.transform(
                    test_data["sentence"])
                test_vectors_tensor = torch.Tensor(
                    test_vectors_sparse.toarray())

                with open(config["model_path"], 'rb') as f:
                    # This is the scikit-learn Naive Bayes model
                    model_instance = pickle.load(f)

                raw_predictions = model_instance.predict(
                    test_vectors_tensor)
                predicted = raw_predictions + 1
                if DEBUG:
                    print(f"[DEBUG] Model '{model_name}': Raw sklearn predictions sample: {
                          raw_predictions[:5]}, Adjusted: {predicted[:5]}")

            # RANDOM PREDICT
            elif config["type"] == "random":
                predicted = rd.choices(
                    [1, 2, 3], k=len(test_data["sentence"]))

            # PRETRAINED TRANSFORMER
            elif config["type"] == "pt_transformer":
                predicted = []
                for sentence in test_data["sentence"]:
                    predicted.append(transformer_predict(sentence))

            # TRAINED TRANSFORMER
            elif config["type"] == "transformer":
                predicted = []
                for sentence in test_data["sentence"]:
                    output = trained_transformer_predict(
                        sentence, model_path=config["model_path"])

                    # Convert the label to the corresponding explicitness value
                    if output == "Implicit":
                        predicted.append(3)
                    elif output == "Neutral":
                        predicted.append(2)
                    elif output == "Explicit":
                        predicted.append(1)
            else:
                print(f"Model {model_name} not recognized. Skipping...")
                continue

            # METRICS CALCULATION TIMEE
            # Calculate accuracy
            accuracy = accuracy_score(
                test_data["explicitness"].values, predicted)
            print(f"Accuracy: {accuracy}")

            # Calculate F1 Score
            f1 = f1_score(
                test_data["explicitness"].values, predicted, average='weighted')
            print(f"F1 Score: {f1}")

            # Calculate precision
            precision = precision_score(
                test_data["explicitness"].values, predicted, average='weighted')
            print(f"Precision: {precision}")

            # Calculate recall
            recall = recall_score(
                test_data["explicitness"].values, predicted, average='weighted')
            print(f"Recall: {recall}")

            # Save the stats
            stats[model_name]["dataset"].append(dataset_name)
            stats[model_name]["accuracy"].append(accuracy)
            stats[model_name]["f1"].append(f1)
            stats[model_name]["precision"].append(precision)
            stats[model_name]["recall"].append(recall)

    # OUTPUT RESULTS (Beautiful)
    print("\n\n" + "=" * 25 + " Overall Benchmark Results " + "=" * 25)

    headers = ["Dataset", "Accuracy",
               "F1 Score (W)", "Precision (W)", "Recall (W)"]

    for model_name_key, model_data in stats.items():
        print(f"\n--- Results for Model: {model_name_key} ---")

        table_rows = []
        # Check if any data was collected for this model
        if not model_data["dataset"]:
            print("No results recorded for this model.")
            continue

        for i in range(len(model_data["dataset"])):
            table_rows.append([
                model_data["dataset"][i],
                f"{model_data['accuracy'][i]:.4f}",
                f"{model_data['f1'][i]:.4f}",
                f"{model_data['precision'][i]:.4f}",
                f"{model_data['recall'][i]:.4f}",
            ])

        if table_rows:
            print(tb.tabulate(table_rows, headers=headers, tablefmt="grid"))
        else:  # Should be caught by the earlier check, but as a safeguard
            print("No results to display in table for this model.")


if __name__ == "__main__":
    main()
