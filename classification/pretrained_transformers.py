from transformers import pipeline
import torch

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

print("✅ CUDA is available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("🖥️  GPU Name:", torch.cuda.get_device_name(0), flush=True)

candidate_labels = [
    "direct factual statement",
    "uncertain or vague statement",
    "sarcastic or ironic statement"  
]

# Update your label map
label_map = {
    "direct factual statement": 1,
    "uncertain or vague statement": 2,
    "sarcastic or ironic statement": 3
}

def test():
        
    test_phrases = [
        "The soup is hot.",
        "The soup temperature seems adequate for serving.",
        "This soup is PERFECTLY scalding my entire mouth!",
        "How efficient to interview me for three hours for a position already taken! 😑"
    ]

    print("\n=== CLASSIFICATION RESULTS ===")

    for sentence in test_phrases:
        print(f"\nClassifying: \"{sentence}\"")
        result = classifier(
            sentence, 
            candidate_labels,
            multi_label=False,
            hypothesis_template="This sentence is a {}." 
        )
        
        print("Results:")
        for label, score in zip(result['labels'], result['scores']):
            original_label = label_map.get(label, label)
            print(f"  {original_label}: {score:.4f}")
        
        # Display best match
        best_label = label_map.get(result['labels'][0], result['labels'][0])
        print(f"→ Best match: {best_label} ({result['scores'][0]:.4f})")

def predict(sentence):
    result = classifier(
        sentence, 
        candidate_labels,
        multi_label=False,
        hypothesis_template="This sentence is a {}." 
    )
    return label_map.get(result['labels'][0], result['labels'][0])

if __name__ == "__main__":
    # Example usage
    sentence = "The soup is hot."
    prediction = predict(sentence)
    print(f"Prediction for \"{sentence}\": {prediction}")
    