# Seq2Seq-like model using TF-IDF and Word2Vec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import os
import random
import pickle

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TfidfWord2VecSeq2Seq(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_vocab_size, dropout=0.3):
        super(TfidfWord2VecSeq2Seq, self).__init__()
        
        # Encoder (maps combined TF-IDF and Word2Vec to hidden state)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Decoder (maps hidden state to output word probabilities)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_vocab_size)
        )
    
    def forward(self, x):
        # Encode input
        encoded = self.encoder(x)
        
        # Decode to output logits (not probabilities)
        output = self.decoder(encoded)
        
        return output

def preprocess_data(data):
    # Tokenize sentences
    data['tokenized_sentence'] = data['sentence'].apply(lambda x: word_tokenize(x.lower()))
    data['tokenized_implied'] = data['implied_statement'].apply(lambda x: word_tokenize(x.lower()))
    
    return data

def train_word2vec(tokenized_sentences, embedding_dim=100):
    # Train Word2Vec model
    model = Word2Vec(sentences=tokenized_sentences, 
                     vector_size=embedding_dim, 
                     window=5, 
                     min_count=1, 
                     workers=4)
    return model

def create_tfidf_vectors(sentences):
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix, vectorizer

def combine_tfidf_word2vec(tfidf_matrix, tfidf_vectorizer, word2vec_model, sentences, embedding_dim):
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Create combined vectors
    combined_vectors = []
    
    for i, sentence in enumerate(sentences):
        # Get TF-IDF vector for this sentence
        tfidf_vector = tfidf_matrix[i].toarray()[0]
        
        # Create weighted average of Word2Vec vectors
        weighted_vector = np.zeros(embedding_dim)
        total_weight = 0
        
        for word in word_tokenize(sentence.lower()):
            if word in word2vec_model.wv:
                # Get word index in TF-IDF vocabulary
                try:
                    word_idx = list(feature_names).index(word)
                    # Get TF-IDF weight
                    weight = tfidf_vector[word_idx]
                    # Add weighted Word2Vec vector
                    weighted_vector += weight * word2vec_model.wv[word]
                    total_weight += weight
                except ValueError:
                    # Word not in TF-IDF vocabulary
                    pass
        
        # Normalize
        if total_weight > 0:
            weighted_vector /= total_weight
            
        combined_vectors.append(weighted_vector)
    
    return np.array(combined_vectors)

def create_output_vocabulary(tokenized_statements):
    # Create vocabulary for output
    vocab = set()
    for statement in tokenized_statements:
        vocab.update(statement)
    
    # Add special tokens
    vocab.add('<PAD>')
    vocab.add('<UNK>')
    vocab.add('<SOS>')
    vocab.add('<EOS>')
    
    # Create word to index and index to word mappings
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return word2idx, idx2word

def create_output_vectors(tokenized_statements, word2idx, max_length=20):
    # Create one-hot vectors for output
    output_vectors = []
    
    for statement in tokenized_statements:
        # Truncate if too long
        if len(statement) > max_length - 2:  # -2 for SOS and EOS
            statement = statement[:max_length-2]
        
        # Add SOS and EOS
        statement = ['<SOS>'] + statement + ['<EOS>']
        
        # Convert to indices
        indices = [word2idx.get(word, word2idx['<UNK>']) for word in statement]
        
        # Pad to max_length
        while len(indices) < max_length:
            indices.append(word2idx['<PAD>'])
        
        # Create one-hot vector
        output_vectors.append(indices)
    
    return np.array(output_vectors)

def generate_text(model, input_vector, idx2word, max_length=20, temperature=0.8):
    model.eval()
    
    with torch.no_grad():
        # Encode input
        input_tensor = torch.FloatTensor(input_vector).unsqueeze(0)
        output_logits = model(input_tensor).squeeze(0)
        
        # Convert logits to probabilities
        output_probs = torch.softmax(output_logits, dim=0).numpy()
        
        # Apply temperature
        output_probs = np.power(output_probs, 1/temperature)
        output_probs /= output_probs.sum()
        
        # Sample words based on probabilities
        words = []
        sos_idx = list(idx2word.keys())[list(idx2word.values()).index('<SOS>')]
        eos_idx = list(idx2word.keys())[list(idx2word.values()).index('<EOS>')]
        pad_idx = list(idx2word.keys())[list(idx2word.values()).index('<PAD>')]
        
        # Start with most probable words
        top_indices = np.argsort(output_probs)[::-1]
        
        for idx in top_indices:
            if idx != sos_idx and idx != eos_idx and idx != pad_idx:
                words.append(idx2word[idx])
                if len(words) >= max_length:
                    break
        
        return ' '.join(words)

def main():
    # Load data
    data = pd.read_csv("../data/gen.csv")
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Train Word2Vec model
    embedding_dim = 100
    word2vec_model = train_word2vec(data['tokenized_sentence'].tolist() + 
                                    data['tokenized_implied'].tolist(), 
                                    embedding_dim)
    
    # Create TF-IDF vectors
    tfidf_matrix, tfidf_vectorizer = create_tfidf_vectors(data['sentence'])
    
    # Combine TF-IDF and Word2Vec
    combined_vectors = combine_tfidf_word2vec(tfidf_matrix, tfidf_vectorizer, 
                                             word2vec_model, data['sentence'], 
                                             embedding_dim)
    
    # Create output vocabulary
    word2idx, idx2word = create_output_vocabulary(data['tokenized_implied'])
    
    # Create output vectors
    output_vectors = create_output_vectors(data['tokenized_implied'], word2idx)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        combined_vectors, output_vectors, test_size=0.2, random_state=42
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    batch_size = min(32, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model parameters
    hidden_dim = 256
    output_vocab_size = len(word2idx)
    dropout = 0.3
    
    # Initialize model with batch normalization for stability
    class StableTfidfWord2VecSeq2Seq(nn.Module):
        def __init__(self, embedding_dim, hidden_dim, output_vocab_size, dropout=0.3):
            super(StableTfidfWord2VecSeq2Seq, self).__init__()
            
            # Encoder with batch normalization
            self.encoder = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Decoder with batch normalization
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, output_vocab_size)
            )
        
        def forward(self, x):
            # Encode input
            encoded = self.encoder(x)
            
            # Decode to output logits
            output = self.decoder(encoded)
            
            return output
    
    # Use the stable model
    model = StableTfidfWord2VecSeq2Seq(embedding_dim, hidden_dim, output_vocab_size, dropout)
    
    # Check if model exists
    model_path = "models/tfidf_word2vec.pth"
    os.makedirs("models", exist_ok=True)
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No existing model found. Training new model...")
        
        # Optimizer with much lower learning rate and weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
        
        # Use a different loss approach - we'll use a sparse categorical cross-entropy
        # that works with indices directly
        criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
        
        # Training loop
        num_epochs = 100
        best_val_loss = float('inf')
        patience = 10
        no_improve = 0
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                # Calculate loss differently - one position at a time
                batch_size = targets.size(0)
                seq_length = targets.size(1)
                
                # Initialize loss for this batch
                batch_loss = 0
                
                # Process each position in the sequence separately
                for i in range(seq_length):
                    # Get target indices for this position
                    target_indices = targets[:, i]
                    
                    # Skip if all targets are padding
                    if (target_indices == word2idx['<PAD>']).all():
                        continue
                    
                    # Calculate loss for this position
                    pos_loss = criterion(outputs, target_indices)
                    
                    # Add to batch loss
                    batch_loss += pos_loss
                
                # Normalize by sequence length
                if seq_length > 0:
                    batch_loss /= seq_length
                
                # Check for NaN before backprop
                if torch.isnan(batch_loss).item():
                    print("NaN loss detected. Skipping batch.")
                    continue
                
                # Backpropagate
                batch_loss.backward()
                
                # Clip gradients very aggressively
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                
                # Step optimizer
                optimizer.step()
                
                # Add to epoch loss
                train_loss += batch_loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    
                    # Calculate loss for each position in sequence
                    batch_size = targets.size(0)
                    seq_length = targets.size(1)
                    
                    batch_loss = 0
                    for i in range(seq_length):
                        # Get target indices for this position
                        target_indices = targets[:, i]
                        
                        # Skip if all targets are padding
                        if (target_indices == word2idx['<PAD>']).all():
                            continue
                        
                        # Calculate loss for this position
                        try:
                            pos_loss = criterion(outputs, target_indices)
                            batch_loss += pos_loss
                        except Exception as e:
                            print(f"Error in validation: {e}")
                            continue
                    
                    # Normalize by sequence length
                    if seq_length > 0:
                        batch_loss /= seq_length
                    
                    # Check for NaN
                    if not torch.isnan(batch_loss).item():
                        val_loss += batch_loss.item()
                    else:
                        print("NaN validation loss detected. Skipping batch.")
                
                # Ensure we don't divide by zero
                if len(val_loader) > 0:
                    val_loss /= len(val_loader)
                else:
                    val_loss = float('inf')
            
            # Check for NaN in final val_loss
            if np.isnan(val_loss):
                print("NaN in final validation loss. Setting to infinity.")
                val_loss = float('inf')
            
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)
                print(f"Model saved with validation loss: {val_loss:.6f}")
                no_improve = 0
            else:
                no_improve += 1
                
            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    # Test with some examples
    test_sentences = [
        "if you are white in that state watch yourself !",
        "capitalism like communism is an enemy of the white race .",
        "so the 500% increase in h1b visas ? yeah i'm sure that will win over voters ."
    ]
    
    print("\nTesting the model with example sentences:")
    for sentence in test_sentences:
        # Create input vector
        tfidf_vector = tfidf_vectorizer.transform([sentence]).toarray()[0]
        combined_vector = combine_tfidf_word2vec(
            tfidf_vectorizer.transform([sentence]), 
            tfidf_vectorizer, 
            word2vec_model, 
            [sentence], 
            embedding_dim
        )[0]
        
        # Generate text
        implied = generate_text(model, combined_vector, idx2word)
        
        print(f"Sentence: {sentence}")
        print(f"Implied statement: {implied}")
        print()
    
    # Interactive mode
    print("Enter a sentence to generate an implied statement (type 'q' to quit):")
    while True:
        sentence = input("> ")
        if sentence.lower() == 'q':
            break
        
        # Create input vector
        try:
            tfidf_vector = tfidf_vectorizer.transform([sentence]).toarray()[0]
            combined_vector = combine_tfidf_word2vec(
                tfidf_vectorizer.transform([sentence]), 
                tfidf_vectorizer, 
                word2vec_model, 
                [sentence], 
                embedding_dim
            )[0]
            
            # Generate text
            implied = generate_text(model, combined_vector, idx2word)
            
            print(f"Implied statement: {implied}")
        except Exception as e:
            print(f"Error: {e}")
            print("Could not generate implied statement for this input.")

    # Save model and components
    os.makedirs("models", exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), "models/tfidf_word2vec.pth")
    
    # Save components
    components = {
        'tfidf_vectorizer': tfidf_vectorizer,
        'word2vec_model': word2vec_model,
        'word2idx': word2idx,
        'idx2word': idx2word
    }
    
    with open("models/tfidf_word2vec_components.pkl", "wb") as f:
        pickle.dump(components, f)
    
    print("Model and components saved successfully.")

if __name__ == "__main__":
    main()
