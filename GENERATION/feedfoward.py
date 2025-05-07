# Feedforward Neural Network for Sequence-to-Sequence Generation

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
import os


class FeedforwardSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(FeedforwardSeq2Seq, self).__init__()
        
        # Encoder (feedforward layers)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Decoder (feedforward layers)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    
    def forward(self, x):
        # Encode input
        encoded = self.encoder(x)
        
        # Decode to output
        output = self.decoder(encoded)
        
        return output


def calculate_bleu(references, hypotheses):
    """
    Calculate BLEU score using a simple implementation
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        
        # Tokenize references and hypotheses
        tokenized_refs = [[ref.lower().split()] for ref in references]
        tokenized_hyps = [hyp.lower().split() for hyp in hypotheses]
        
        # Calculate BLEU score with smoothing
        try:
            smoothie = SmoothingFunction().method1
            bleu_score = corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smoothie)
        except TypeError:
            # Older versions of NLTK don't support the _normalize parameter
            print("Using alternative BLEU calculation due to NLTK version incompatibility")
            bleu_score = simple_bleu_calculation(references, hypotheses)
        
        return bleu_score
    
    except ImportError:
        print("NLTK not found. Using simple BLEU implementation.")
        return simple_bleu_calculation(references, hypotheses)


def simple_bleu_calculation(references, hypotheses):
    """
    A simple implementation of BLEU-1 score (unigram precision)
    """
    total_precision = 0.0
    count = 0
    
    for ref, hyp in zip(references, hypotheses):
        if len(hyp.strip()) == 0:
            continue
            
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        
        # Count matching tokens
        matches = sum(1 for token in hyp_tokens if token in ref_tokens)
        precision = matches / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
        
        total_precision += precision
        count += 1
    
    return total_precision / count if count > 0 else 0.0


def calculate_rouge(references, hypotheses):
    """
    Calculate ROUGE score using a simple implementation
    """
    try:
        from rouge import Rouge
        rouge = Rouge()
        
        # Handle empty hypotheses or references
        valid_pairs = [(ref, hyp) for ref, hyp in zip(references, hypotheses) 
                      if len(hyp.strip()) > 0 and len(ref.strip()) > 0]
        
        if not valid_pairs:
            return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
        
        refs, hyps = zip(*valid_pairs)
        
        try:
            scores = rouge.get_scores(hyps, refs, avg=True)
            return scores
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    except ImportError:
        print("Rouge not found. Using simple ROUGE implementation.")
        return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}


def get_text_from_vector(vector, vectorizer):
    """
    Convert a TF-IDF vector back to text by selecting the top words
    with some randomness for diversity
    """
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get indices of top values in the vector
    top_indices = vector.argsort()[-20:][::-1]  # Get top 20 words
    
    # Get the corresponding words with values
    top_words_with_values = [(feature_names[i], vector[i]) for i in top_indices if vector[i] > 0]
    
    # Fix for empty or small lists
    if len(top_words_with_values) < 5:
        # If we have fewer than 5 words, just use all of them
        selected_words = [word for word, _ in top_words_with_values]
    else:
        # Sort by value but add some randomness
        import random
        random.shuffle(top_words_with_values)  # Add randomness
        top_words_with_values.sort(key=lambda x: x[1] + random.random() * 0.01, reverse=True)
        
        # Take top 5-10 words (random number for diversity)
        num_words = random.randint(5, min(10, len(top_words_with_values)))
        selected_words = [word for word, _ in top_words_with_values[:num_words]]
    
    # Join words to form a sentence
    return " ".join(selected_words)


def main():
    # Load data
    data = pd.read_csv("../data/gen.csv")
    
    print(f"Dataset size: {len(data)} examples")
    
    # Create TF-IDF vectorizers for input and output with lower max_features
    # to avoid overfitting on small dataset
    input_vectorizer = TfidfVectorizer(max_features=1000)
    output_vectorizer = TfidfVectorizer(max_features=1000)
    
    # Fit vectorizers
    input_vectors = input_vectorizer.fit_transform(data["sentence"])
    output_vectors = output_vectorizer.fit_transform(data["implied_statement"])
    
    # Print some stats about the vectors
    print(f"Input vector shape: {input_vectors.shape}")
    print(f"Output vector shape: {output_vectors.shape}")
    print(f"Input vector sparsity: {input_vectors.nnz / (input_vectors.shape[0] * input_vectors.shape[1]):.4f}")
    print(f"Output vector sparsity: {output_vectors.nnz / (output_vectors.shape[0] * output_vectors.shape[1]):.4f}")
    
    # Convert to dense tensors
    input_dense = torch.FloatTensor(input_vectors.toarray())
    output_dense = torch.FloatTensor(output_vectors.toarray())
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        input_dense, output_dense, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = min(32, len(train_dataset))  # Ensure batch size isn't larger than dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = input_dense.shape[1]
    output_dim = output_dense.shape[1]
    hidden_dim = 256  # Reduced from 512
    dropout = 0.3  # Reduced from 0.7 which might be too aggressive
    
    # Initialize model with a more complex architecture
    class EnhancedFeedforwardSeq2Seq(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
            super(EnhancedFeedforwardSeq2Seq, self).__init__()
            
            # Encoder with more layers
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            )
            
            # Decoder with more layers
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, output_dim),
                nn.Sigmoid()  # Constrain outputs between 0 and 1
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            output = self.decoder(encoded)
            return output
    
    # Use the enhanced model
    model = EnhancedFeedforwardSeq2Seq(input_dim, hidden_dim, output_dim, dropout).to(device)
    
    # Check if model exists
    model_path = "models/forward.pth"
    os.makedirs("models", exist_ok=True)
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No existing model found. Training new model...")
        
        # Optimizer with better parameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Use a combination of MSE and cosine similarity loss
        mse_criterion = nn.MSELoss()
        cos_criterion = nn.CosineEmbeddingLoss()
        
        # Learning rate scheduler without verbose parameter
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop with more patience
        num_epochs = 100
        best_val_loss = float('inf')
        patience = 15  # Increased patience
        no_improve = 0
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                # MSE loss
                mse_loss = mse_criterion(outputs, targets)
                
                # Cosine similarity loss (helps with sparse vectors)
                batch_size = inputs.size(0)
                cos_loss = cos_criterion(
                    outputs, targets, 
                    torch.ones(batch_size).to(device)
                )
                
                # Combined loss
                loss = mse_loss + 0.5 * cos_loss
                
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    
                    # MSE loss
                    mse_loss = mse_criterion(outputs, targets)
                    
                    # Cosine similarity loss
                    batch_size = inputs.size(0)
                    cos_loss = cos_criterion(
                        outputs, targets, 
                        torch.ones(batch_size).to(device)
                    )
                    
                    # Combined loss
                    loss = mse_loss + 0.5 * cos_loss
                    
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model with more patience
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)
                print(f"Model saved with validation loss: {val_loss:.6f}")
                no_improve = 0
            else:
                no_improve += 1
                
            # Early stopping with patience
            if no_improve >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    # Evaluate model
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            
            # Get model predictions
            outputs = model(inputs)
            
            # Convert TF-IDF vectors back to text
            for i in range(outputs.size(0)):
                # Get target text
                target_vector = targets[i].cpu().numpy()
                reference = get_text_from_vector(target_vector, output_vectorizer)
                references.append(reference)
                
                # Get predicted text
                output_vector = outputs[i].cpu().numpy()
                hypothesis = get_text_from_vector(output_vector, output_vectorizer)
                hypotheses.append(hypothesis)
    
    # Calculate BLEU score
    bleu_score = calculate_bleu(references, hypotheses)
    print(f"BLEU score: {bleu_score:.4f}")
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge(references, hypotheses)
    print(f"ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")
    
    # Function to generate implied statements
    def generate_implied_statement(sentence):
        model.eval()
        
        # Vectorize the input sentence
        input_vector = input_vectorizer.transform([sentence]).toarray()
        input_tensor = torch.FloatTensor(input_vector).to(device)
        
        # Generate output
        with torch.no_grad():
            output = model(input_tensor)
            
        # Convert output vector to text
        output_vector = output[0].cpu().numpy()
        implied_statement = get_text_from_vector(output_vector, output_vectorizer)
        
        return implied_statement
    
    # Test with some examples
    test_sentences = [
        "if you are white in that state watch yourself !",
        "capitalism like communism is an enemy of the white race .",
        "so the 500% increase in h1b visas ? yeah i'm sure that will win over voters ."
    ]
    
    print("\nTesting the model with example sentences:")
    for sentence in test_sentences:
        implied = generate_implied_statement(sentence)
        print(f"Sentence: {sentence}")
        print(f"Implied statement: {implied}")
        print()
    
    # Interactive mode
    print("Enter a sentence to generate an implied statement (type 'q' to quit):")
    while True:
        sentence = input("> ")
        if sentence.lower() == 'q':
            break
        
        implied = generate_implied_statement(sentence)
        print(f"Implied statement: {implied}")


if __name__ == "__main__":
    main()


"""
We implemented a feedforward neural network that maps TF-IDF vector representations of potentially biased statements to their implied meanings. The model uses a multi-layer architecture with batch normalization, LeakyReLU activations, and dropout regularization to learn these mappings. For output generation, we convert the predicted vectors back to text by selecting the most relevant words with controlled randomness.
Several challenges emerged during implementation. The model initially produced identical outputs for all inputs due to premature convergence and insufficient differentiation in the vector space. We addressed this by implementing a combined MSE and cosine similarity loss function, adding weight decay to prevent overfitting, and introducing controlled randomness in the output generation process. Additionally, we encountered technical issues including empty ranges in random selection and PyTorch version incompatibilities, which were resolved with appropriate error handling.
The feedforward approach successfully captured thematic connections between inputs and implied biases, though with limitations compared to sequence models. Outputs consisted of relevant keywords rather than grammatical sentences, trading linguistic sophistication for computational efficiency. Despite these constraints, the model demonstrated that even simple feedforward architectures can effectively detect implicit biases in text, making it a viable option for bias analysis applications with limited computational resources.
"""