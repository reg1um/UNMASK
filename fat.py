import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import seaborn as sns

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

# Create tmp directory if it doesn't exist
os.makedirs('tmp', exist_ok=True)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

# Load data
try:
    data = pd.read_excel(r"C:\Users\leolo\OneDrive\Bureau\UNMASK\data\GOAT.xlsx", engine='openpyxl')
    print("Loaded data using openpyxl")
except:
    try:
        data = pd.read_csv(r'C:\Users\leolo\OneDrive\Bureau\UNMASK\data\sentence_explicitness_dataset_fewer_emojis_GOAT.xlsx')
    except:
        print("Error: Could not load the dataset. Please ensure the file exists.")
        exit(1)

print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns")
print("Distribution of explicitness labels:")
print(data['explicitness'].value_counts().sort_index())

# Preprocess text (apply only if you want to use the preprocessing)
USE_PREPROCESSING = False  # Set to True to use preprocessing
if USE_PREPROCESSING:
    print("Preprocessing text data...")
    data['processed_sentence'] = data['sentence'].apply(preprocess_text)
    X = data['processed_sentence']
    print("Text preprocessing complete")
else:
    X = data['sentence']

Y = data['explicitness']
Y = Y - 1  # Convert 1,2,3 to 0,1,2

# Split with stratification to ensure class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y  # Changed from 200
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Feature extraction - try both TF-IDF and Count vectorization
VECTORIZER_TYPE = "tfidf"  # Options: "tfidf" or "count"

if VECTORIZER_TYPE == "tfidf":
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Increase from 800
        ngram_range=(1, 3),  # Include trigrams to better capture phrases
        min_df=1,  # Lower threshold to include more unique features
        max_df=0.95,  # Increase to include more common terms
        stop_words=None  # Remove stop word filtering - some might be important
    )
else:
    vectorizer = CountVectorizer(
        max_features=800,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        stop_words='english'
    )

# Fit and transform
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Features: {X_train_vec.shape[1]}")

# Feature importance analysis (for TF-IDF)
if VECTORIZER_TYPE == "tfidf":
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate TF-IDF average by class
    class_feature_importance = {}
    for class_idx in range(3):  # For classes 0, 1, 2
        # Get samples for this class
        class_indices = y_train[y_train == class_idx].index
        class_samples = X_train_vec[X_train.index.get_indexer(class_indices)]
        
        # Calculate average TF-IDF for each feature
        avg_tfidf = np.array(class_samples.mean(axis=0)).flatten()
        
        # Get top features
        top_indices = avg_tfidf.argsort()[-20:][::-1]
        top_features = [(feature_names[i], avg_tfidf[i]) for i in top_indices]
        
        class_feature_importance[class_idx] = top_features
    
    # Print top features by class
    print("\nTop features by class:")
    for class_idx, features in class_feature_importance.items():
        original_class = class_idx + 1
        print(f"Class {original_class} (features with highest average TF-IDF):")
        for feature, score in features[:10]:  # Top 10
            print(f"  - {feature}: {score:.4f}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_vec.toarray()).to(device)
X_test_tensor = torch.FloatTensor(X_test_vec.toarray()).to(device)
y_train_tensor = torch.LongTensor(y_train.values).to(device)
y_test_tensor = torch.LongTensor(y_test.values).to(device)

# Define an even more robust model with batch normalization
class AdvancedModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes=3, dropout_rate=0.3):  # Lower dropout rate from 0.5 to 0.3
        super(AdvancedModel, self).__init__()
        
        # Reduce complexity - smaller hidden layers
        hidden1 = 64  # Reduced from 128
        
        # Simplified architecture - remove one hidden layer
        self.layer1 = torch.nn.Linear(input_dim, hidden1)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.output = torch.nn.Linear(hidden1, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.output(x)
        return x

# Initialize model
input_dim = X_train_vec.shape[1]
num_classes = 3
model = AdvancedModel(input_dim, num_classes, dropout_rate=0.3).to(device)
print(f"Advanced model created and moved to {device}")

# Loss function and optimizer with stronger regularization
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)  # Increase learning rate, reduce regularization

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.8, patience=30, verbose=True  # Gentler decay, more patience
)

# Early stopping
patience = 80  # Increased from 50
best_loss = float('inf')
early_stop_counter = 0
best_model_state = None

# Training with validation and learning rate scheduling
num_epochs = 500
batch_size = 32
n_batches = int(np.ceil(len(X_train_tensor) / batch_size))
print("Training advanced model...")

# Track metrics
train_losses = []
val_losses = []
val_accuracies = []
lr_history = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_train_loss = 0
    
    # Shuffle training data
    indices = torch.randperm(len(X_train_tensor))
    X_train_shuffled = X_train_tensor[indices]
    y_train_shuffled = y_train_tensor[indices]
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_train_tensor))
        
        X_batch = X_train_shuffled[start_idx:end_idx]
        y_batch = y_train_shuffled[start_idx:end_idx]
        
        # Forward pass
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / n_batches
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_logits = model(X_test_tensor)
        val_loss = criterion(val_logits, y_test_tensor).item()
        val_losses.append(val_loss)
        
        # Calculate validation accuracy
        _, val_preds = torch.max(val_logits, 1)
        val_accuracy = (val_preds == y_test_tensor).sum().item() / len(y_test_tensor)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Store current learning rate
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
    
    # Print progress
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Early stopping check
    if early_stop_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load the best model
if best_model_state:
    model.load_state_dict(best_model_state)
    print("Loaded best model based on validation loss")

# Plot training curves
plt.figure(figsize=(15, 10))

# Loss plot
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Accuracy plot
plt.subplot(2, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

# Learning rate plot
plt.subplot(2, 2, 3)
plt.plot(lr_history, label='Learning Rate')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()

plt.tight_layout()
plt.savefig('tmp/training_metrics.png')
print("Training metrics saved to tmp/training_metrics.png")

# Evaluation
model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    _, y_pred = torch.max(logits, 1)
    y_pred_np = y_pred.cpu().numpy()
    y_test_np = y_test.values

# Calculate accuracy
accuracy = accuracy_score(y_test_np, y_pred_np)
print(f"Accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test_np, y_pred_np)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 1', 'Class 2', 'Class 3'],
            yticklabels=['Class 1', 'Class 2', 'Class 3'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('tmp/confusion_matrix.png')
print("Confusion matrix saved to tmp/confusion_matrix.png")

# Print classification report
print("Classification Report:")
print(classification_report(y_test_np, y_pred_np))

# Error Analysis - examine misclassified examples
if not np.array_equal(y_pred_np, y_test_np):
    misclassified_indices = np.where(y_pred_np != y_test_np)[0]
    print(f"\nFound {len(misclassified_indices)} misclassified examples")
    
    if len(misclassified_indices) > 0:
        print("\nMisclassified Examples:")
        test_sentences = X_test.reset_index(drop=True)
        
        for idx in misclassified_indices:
            true_label = y_test_np[idx] + 1  # Convert back to original label
            pred_label = y_pred_np[idx] + 1  # Convert back to original label
            sentence = test_sentences.iloc[idx]
            
            print(f"Example: '{sentence}'")
            print(f"True: {true_label}, Predicted: {pred_label}")
            print("-" * 50)
else:
    print("\nNo misclassified examples found")

# Save the model and vectorizer
print("Saving model and vectorizer to tmp folder...")
torch.save(model.state_dict(), 'tmp/advanced_model.pt')
joblib.dump(vectorizer, 'tmp/advanced_vectorizer.pkl')

# Save model metadata
model_info = {
    'input_dim': input_dim,
    'num_classes': num_classes,
    'device': str(device),
    'original_labels': [1, 2, 3],
    'model_labels': [0, 1, 2],
    'model_type': 'advanced',
    'preprocessing_used': USE_PREPROCESSING,
    'vectorizer_type': VECTORIZER_TYPE
}
joblib.dump(model_info, 'tmp/advanced_model_info.pkl')

print("Advanced model and vectorizer saved successfully")