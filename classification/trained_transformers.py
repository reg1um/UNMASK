from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import evaluate
from datasets import Dataset
import torch
import torch.cuda as cuda
import numpy as np

print("✅ CUDA is available:", cuda.is_available(), flush=True)
if cuda.is_available():
    print("🖥️  GPU Name:", cuda.get_device_name(0), flush=True)
    device = "cuda"
else:
    print("🖥️  CPU Name:", "CPU", flush=True)
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
accuarcy = evaluate.load("accuracy")

id2label = {0: "Explicit", 1: "Neutral", 2: "Implicit"}
label2id = {
    "Explicit": 0,
    "Neutral": 1,
    "Implicit": 2
}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

output_dir = "trained_transformers"

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)


def preprocess_function(dataset):
    return tokenizer(dataset["sentence"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuarcy.compute(predictions=predictions, references=labels)


def predict(sentence, model_path=output_dir, verbose=False):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")

    model.to(device)
    model.eval()

    inputs = tokenizer(sentence, return_tensors="pt",
                       truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1)
    confidence, predictions = torch.max(probabilities, dim=-1)

    label = id2label[predictions.item()]
    if verbose:
        print(f"Prediction: {label}, Confidence: {confidence.item():.2f}")
    return label


def main(path="../data/GOAT.csv"):
    data = pd.read_csv(path)

    data["label"] = data["explicitness"] - 1

    dataset = Dataset.from_pandas(data)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    res = trainer.evaluate()
    print(res)

    sentences = [
        "The soup is hot.",
        "The soup temperature seems adequate for serving.",
        "This soup is PERFECTLY scalding my entire mouth!",
        "How efficient to interview me for three hours for a position already taken! 😑"
    ]
    for sentence in sentences:
        print(f"Classifying: \"{sentence}\"")
        result = predict(sentence)


if __name__ == "__main__":
    main()
