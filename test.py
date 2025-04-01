import torch
import joblib
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_model_and_vectorizer(model_type="advanced"):
    """Load the model and vectorizer from the tmp directory"""
    if model_type == "advanced":
        model_path = 'tmp/advanced_model.pt'
        vectorizer_path = 'tmp/advanced_vectorizer.pkl'
        model_info_path = 'tmp/advanced_model_info.pkl'
    else:
        model_path = 'tmp/multi_class_model.pt'
        vectorizer_path = 'tmp/tfidf_vectorizer.pkl'
        model_info_path = 'tmp/model_info.pkl'
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print(f"Error: {model_type} model or vectorizer not found in tmp folder.")
        print(f"Files needed: {model_path}, {vectorizer_path}")
        print("Please run the appropriate training script first.")
        sys.exit(1)
    
    print(f"Loading {model_type} model and vectorizer...")
    
    # Load the vectorizer
    vectorizer = joblib.load(vectorizer_path)
    
    # Load model info
    model_info = joblib.load(model_info_path)
    input_dim = model_info['input_dim']
    num_classes = model_info['num_classes']
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model with appropriate architecture
    if model_type == "advanced":
        # Updated AdvancedModel class definition to match the saved model
        class AdvancedModel(torch.nn.Module):
            def __init__(self, input_dim, num_classes=3, dropout_rate=0.3):
                super(AdvancedModel, self).__init__()
                
                # Simple architecture to match the saved model
                hidden1 = 64
                
                self.layer1 = torch.nn.Linear(input_dim, hidden1)
                self.dropout1 = torch.nn.Dropout(dropout_rate)
                self.output = torch.nn.Linear(hidden1, num_classes)
                
            def forward(self, x):
                x = self.layer1(x)
                x = torch.nn.functional.relu(x)
                x = self.dropout1(x)
                x = self.output(x)
                return x
        
        model = AdvancedModel(input_dim, num_classes, dropout_rate=0.3).to(device)
    
    elif model_info.get('has_hidden_layer', False):
        # Multi-layer model with regularization
        hidden_dim = model_info.get('hidden_dim', 100)
        
        class MultiClassModelWithRegularization(torch.nn.Module):
            def __init__(self, input_dim, num_classes, hidden_dim):
                super(MultiClassModelWithRegularization, self).__init__()
                self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
                self.dropout = torch.nn.Dropout(0.3)
                self.layer2 = torch.nn.Linear(hidden_dim, num_classes)
                
            def forward(self, x):
                x = self.layer1(x)
                x = torch.nn.functional.relu(x)
                x = self.dropout(x)
                x = self.layer2(x)
                return x
        
        model = MultiClassModelWithRegularization(input_dim, num_classes, hidden_dim).to(device)
    
    else:
        # Simple linear model
        class MultiClassModel(torch.nn.Module):
            def __init__(self, input_dim, num_classes):
                super(MultiClassModel, self).__init__()
                self.linear = torch.nn.Linear(input_dim, num_classes)
                
            def forward(self, x):
                return self.linear(x)
        
        model = MultiClassModel(input_dim, num_classes).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # Set to evaluation mode
    
    print(f"{model_type.capitalize()} model and vectorizer loaded successfully")
    
    return model, vectorizer, device, model_info

def predict_explicitness(text, model, vectorizer, device, model_info):
    """Predict explicitness level for a given text"""
    model.eval()  # Set model to evaluation mode
    
    # Transform text
    text_tfidf = vectorizer.transform([text])
    text_tensor = torch.FloatTensor(text_tfidf.toarray()).to(device)
    
    # Make prediction
    with torch.no_grad():
        logits = model(text_tensor)
        probabilities = torch.softmax(logits, dim=1)
        probs, predicted = torch.max(probabilities, 1)
        predicted_class = predicted.item()
        confidence = probs.item()
    
    # Convert back to original label (add 1)
    original_label = predicted_class + 1
    
    # Label meanings
    label_meanings = {
        1: "Explicit meaning",
        2: "Ambiguous (can be both explicit or implicit)",
        3: "Implicit meaning"
    }
    
    return {
        'text': text,
        'predicted_class': original_label,
        'confidence': confidence,
        'class_meaning': label_meanings.get(original_label, "Unknown"),
        'all_probabilities': {f"Level {i+1}": float(prob) for i, prob in enumerate(probabilities[0].cpu().numpy())}
    }

def interactive_testing(model, vectorizer, device, model_info):
    """Interactive console for testing individual sentences"""
    print("\n=== Interactive Testing Mode ===")
    print("Enter sentences to test (type 'q' to quit)")
    print("Labels: 1=Explicit, 2=Ambiguous, 3=Implicit")
    
    # Keep track of all predictions for analysis at the end
    all_predictions = []
    
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'q':
            break
        elif not text.strip():
            continue
        
        result = predict_explicitness(text, model, vectorizer, device, model_info)
        all_predictions.append(result)
        
        print(f"Prediction: {result['class_meaning']} (Level {result['predicted_class']})")
        print(f"Confidence: {result['confidence']:.4f}")
        print("All class probabilities:")
        for label, prob in result['all_probabilities'].items():
            print(f"  {label}: {prob:.4f}")
    
    # Analyze predictions if there were any
    if len(all_predictions) > 0:
        # Offer to save the results
        save_results = input("\nWould you like to save these results? (y/n): ")
        if save_results.lower() == 'y':
            save_interactive_results(all_predictions)

def save_interactive_results(predictions):
    """Save interactive testing results to a CSV file"""
    results_data = []
    
    for p in predictions:
        results_data.append({
            'sentence': p['text'],
            'predicted_class': p['predicted_class'],
            'confidence': p['confidence'],
            'meaning': p['class_meaning'],
            'prob_level_1': p['all_probabilities']['Level 1'],
            'prob_level_2': p['all_probabilities']['Level 2'],
            'prob_level_3': p['all_probabilities']['Level 3']
        })
    
    results_df = pd.DataFrame(results_data)
    filename = 'interactive_test_results.csv'
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def batch_testing(file_path, model, vectorizer, device, model_info):
    """Test model with a CSV or Excel file containing sentences"""
    try:
        # Load the test data
        if file_path.endswith('.xlsx'):
            test_data = pd.read_excel(file_path)
        else:
            test_data = pd.read_csv(file_path)
        
        # Check if the file has the required column
        if 'sentence' not in test_data.columns:
            print("Error: Input file must have a 'sentence' column")
            return
        
        # Get sentences
        test_sentences = test_data['sentence'].tolist()
        print(f"Processing {len(test_sentences)} sentences...")
        
        # Process each sentence
        results = []
        for sentence in test_sentences:
            result = predict_explicitness(sentence, model, vectorizer, device, model_info)
            results.append({
                'sentence': sentence,
                'explicitness_level': result['predicted_class'],
                'meaning': result['class_meaning'],
                'confidence': result['confidence'],
                'level_1_prob': result['all_probabilities']['Level 1'],
                'level_2_prob': result['all_probabilities']['Level 2'],
                'level_3_prob': result['all_probabilities']['Level 3']
            })
        
        # Create and save results DataFrame
        results_df = pd.DataFrame(results)
        output_path = os.path.splitext(file_path)[0] + '_results.csv'
        results_df.to_csv(output_path, index=False)
        
        print(f"Results saved to {output_path}")
        
        # Print summary statistics
        print("\nPrediction Summary:")
        value_counts = results_df['explicitness_level'].value_counts().sort_index()
        for level, count in value_counts.items():
            print(f"  Level {level}: {count} sentences ({count/len(results_df)*100:.1f}%)")
        
        # Plot distribution
        plt.figure(figsize=(8, 6))
        plt.bar(['Explicit (1)', 'Ambiguous (2)', 'Implicit (3)'], 
                [results_df['explicitness_level'].value_counts().get(1, 0),
                 results_df['explicitness_level'].value_counts().get(2, 0),
                 results_df['explicitness_level'].value_counts().get(3, 0)])
        plt.title('Distribution of Predictions')
        plt.ylabel('Count')
        plt.savefig('prediction_distribution.png')
        print("Distribution chart saved to prediction_distribution.png")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def analyze_model_features(vectorizer, model_info):
    """Analyze and display the most important features used by the model"""
    feature_names = vectorizer.get_feature_names_out()
    
    print("\n=== Model Feature Analysis ===")
    print(f"Total features: {len(feature_names)}")
    
    # Show some sample features
    print("\nSample features used by the model:")
    sample_size = min(20, len(feature_names))
    for i, feature in enumerate(feature_names[:sample_size]):
        print(f"  {i+1}. {feature}")
    
    # If this is a count vectorizer or TF-IDF, show top n-grams
    if hasattr(vectorizer, 'ngram_range'):
        print(f"\nN-gram range: {vectorizer.ngram_range}")
        
        # Count features by length (unigrams vs bigrams)
        unigrams = [f for f in feature_names if ' ' not in f]
        bigrams = [f for f in feature_names if ' ' in f]
        
        print(f"Unigrams: {len(unigrams)} features")
        print(f"Bigrams: {len(bigrams)} features")
        
        if bigrams:
            print("\nSample bigrams:")
            sample_bigrams = bigrams[:10]
            for i, bigram in enumerate(sample_bigrams):
                print(f"  {i+1}. {bigram}")

def main():
    """Main function to run the testing app"""
    print("===== Explicitness Classification Model Tester =====")
    
    # Ask which model to use
    print("\nWhich model would you like to use?")
    print("1. Advanced model (with batch normalization)")
    print("2. Regular model (multi-class classifier)")
    
    model_choice = input("Select an option (1-2): ")
    
    # Load model and vectorizer based on user choice
    if model_choice == '1':
        model, vectorizer, device, model_info = load_model_and_vectorizer("advanced")
    else:
        model, vectorizer, device, model_info = load_model_and_vectorizer("regular")
    
    while True:
        # Display menu
        print("\nOptions:")
        print("1. Test with individual sentences")
        print("2. Test with a file (CSV or Excel)")
        print("3. Analyze model features")
        print("4. Exit")
        
        choice = input("Select an option (1-4): ")
        
        if choice == '1':
            interactive_testing(model, vectorizer, device, model_info)
        elif choice == '2':
            file_path = input("Enter path to test file (CSV or Excel): ")
            batch_testing(file_path, model, vectorizer, device, model_info)
        elif choice == '3':
            analyze_model_features(vectorizer, model_info)
        elif choice == '4':
            print("Exiting program")
            break
        else:
            print("Invalid choice. Please select a valid option (1-4).")

if __name__ == "__main__":
    main()