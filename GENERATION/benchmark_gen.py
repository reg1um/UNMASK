# benchmark_genai.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import evaluate
import time
import json
import os
import re
import string
import random
from collections import defaultdict, Counter
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm.auto import tqdm # For progress bars

# --- Ensure NLTK resources are available ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
    print("NLTK 'punkt' downloaded.")

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab', quiet=True)
    print("NLTK 'punkt_tab' downloaded.")

# --- Global Variables ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = "models/"
TRANSFORMER_MODEL_PATH = "gen_transformers/"
DATA_PATH = "../data/gen.csv"
TEMP_EVAL_DIR = "./temp_hf_eval" # For Hugging Face Trainer temp files

# --- Pre-load expensive metrics once ---
print("Pre-loading BERTScore metric...")
try:
    BERTSCORE_METRIC = evaluate.load("bertscore")
    print("BERTScore metric loaded successfully.")
except Exception as e:
    print(f"Warning: Could not pre-load BERTScore metric: {e}")
    BERTSCORE_METRIC = None

# --- Helper Preprocessing Function ---
def ngram_preprocess_function(text):
    text = text.lower()
    text = re.sub("[" + string.punctuation + "]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Metric Calculation Functions ---
def calculate_bleu(references, hypotheses):
    print("  Calculating BLEU score...")
    s_time = time.time()
    hypotheses_tokenized = [hyp.split() for hyp in hypotheses]
    references_tokenized = [[ref.split()] for ref in references]
    try:
        score = corpus_bleu(references_tokenized, hypotheses_tokenized, smoothing_function=SmoothingFunction().method1)
        print(f"  BLEU calculated in {time.time() - s_time:.2f}s")
        return score
    except ZeroDivisionError:
        print("  BLEU calculation resulted in ZeroDivisionError.")
        return 0.0
    except Exception as e:
        print(f"  Error in BLEU calculation: {e}")
        return "Error"


def calculate_rouge(references, hypotheses):
    print("  Calculating ROUGE scores...")
    s_time = time.time()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1_f, r2_f, rL_f = [], [], []
    for ref, hyp in tqdm(zip(references, hypotheses), total=len(references), desc="  ROUGE"):
        if not hyp.strip():
            r1_f.append(0.0); r2_f.append(0.0); rL_f.append(0.0)
            continue
        try:
            scores = scorer.score(ref, hyp)
            r1_f.append(scores['rouge1'].fmeasure)
            r2_f.append(scores['rouge2'].fmeasure)
            rL_f.append(scores['rougeL'].fmeasure)
        except Exception as e:
            # print(f"    Warning: ROUGE scoring failed for pair: Ref: '{ref[:50]}...', Hyp: '{hyp[:50]}...'. Error: {e}")
            r1_f.append(0.0); r2_f.append(0.0); rL_f.append(0.0) # Default on error for a pair
    
    results = {
        "rouge1": np.mean(r1_f) if r1_f else 0.0,
        "rouge2": np.mean(r2_f) if r2_f else 0.0,
        "rougeL": np.mean(rL_f) if rL_f else 0.0,
    }
    print(f"  ROUGE calculated in {time.time() - s_time:.2f}s")
    return results

def calculate_bertscore(references, hypotheses):
    print("  Calculating BERTScore...")
    s_time = time.time()
    if BERTSCORE_METRIC is None:
        print("  BERTScore metric not loaded, skipping BERTScore calculation.")
        return {"precision": "Error", "recall": "Error", "f1": "Error"}

    valid_indices = [i for i, hyp in enumerate(hypotheses) if hyp.strip()]
    if not valid_indices:
        print("  No valid hypotheses for BERTScore.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    filtered_refs = [references[i] for i in valid_indices]
    filtered_hyps = [hypotheses[i] for i in valid_indices]

    if not filtered_refs or not filtered_hyps:
        print("  Not enough valid reference/hypothesis pairs for BERTScore after filtering.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    try:
        results = BERTSCORE_METRIC.compute(predictions=filtered_hyps, references=filtered_refs, lang="en", device=DEVICE, batch_size=16) # Added batch_size
        print(f"  BERTScore calculated in {time.time() - s_time:.2f}s")
        return {
            "precision": np.mean(results["precision"]) if "precision" in results and results["precision"] is not None else 0.0,
            "recall": np.mean(results["recall"]) if "recall" in results and results["recall"] is not None else 0.0,
            "f1": np.mean(results["f1"]) if "f1" in results and results["f1"] is not None else 0.0,
        }
    except Exception as e:
        print(f"  Error in BERTScore calculation: {e}")
        return {"precision": "Error", "recall": "Error", "f1": "Error"}


def calculate_perplexity_hf(model, tokenizer, hf_dataset_test, model_name="Transformer"):
    print(f"  Calculating Perplexity for {model_name}...")
    s_time = time.time()
    # Consider sub-sampling for faster debugging:
    # sample_size = min(100, len(hf_dataset_test))
    # hf_dataset_test = HFDataset.from_dict(hf_dataset_test.select(range(sample_size)).to_dict())
    # print(f"    Using a sample of {len(hf_dataset_test)} for perplexity calculation.")

    def preprocess_for_perplexity(examples):
        inputs = ["generate implication: " + sentence for sentence in examples["sentence"]] # Assuming T5-style prefix
        model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
        labels = tokenizer(text_target=examples["implied_statement"], max_length=64, truncation=True, padding="max_length")
        model_inputs["labels"] = [[l if l != tokenizer.pad_token_id else -100 for l in label] for label in labels["input_ids"]]
        return model_inputs

    print(f"    Tokenizing {len(hf_dataset_test)} examples for perplexity...")
    tokenized_test = hf_dataset_test.map(preprocess_for_perplexity, batched=True, remove_columns=hf_dataset_test.column_names, desc="Tokenizing for Perplexity")
    
    # Ensure temp eval dir exists
    os.makedirs(TEMP_EVAL_DIR, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=TEMP_EVAL_DIR, 
        per_device_eval_batch_size=max(1, 4 // torch.cuda.device_count()) if torch.cuda.is_available() else 4, # Adjust batch size per GPU
        fp16=torch.cuda.is_available(),
        report_to="none" # Disable wandb/other reporting for eval
    )
    trainer = Seq2SeqTrainer(model=model, args=training_args, eval_dataset=tokenized_test, tokenizer=tokenizer)
    
    try:
        print("    Trainer evaluating for perplexity...")
        eval_results = trainer.evaluate()
        ce_loss = eval_results.get("eval_loss", float('nan'))
        print(f"  Perplexity for {model_name} (CE Loss: {ce_loss:.4f}) calculated in {time.time() - s_time:.2f}s")
        return ce_loss
    except Exception as e:
        print(f"  Error during HF perplexity calculation for {model_name}: {e}")
        return float('nan')

# --- N-gram Model Definition ---
class ImpliedStatementGenerator:
    def __init__(self, n=3): self.n = n; self.sentence_vectorizer = None; self.implied_vectorizer = None; self.ngram_associations = defaultdict(list); self.implied_statements = []; self.start_words = []; self.transitions = defaultdict(list)
    def fit(self, df): pass
    def generate(self, sentence):
        if self.sentence_vectorizer is None: raise ValueError("N-gram model not loaded.")
        clean_sentence = ngram_preprocess_function(sentence)
        X = self.sentence_vectorizer.transform([clean_sentence])
        ngram_indices = X.nonzero()[1]
        if len(ngram_indices) == 0: return random.choice(self.implied_statements) if self.implied_statements else "No n-grams/fallback."
        active_ngrams = [self.sentence_vectorizer.get_feature_names_out()[idx] for idx in ngram_indices]
        return self._generate_by_matching(active_ngrams)
    def _generate_by_matching(self, ngrams):
        all_implications = [imp for gram in ngrams if gram in self.ngram_associations for imp in self.ngram_associations[gram]]
        if all_implications: return Counter(all_implications).most_common(1)[0][0]
        return random.choice(self.implied_statements) if self.implied_statements else "No matching implications."

def predict_ngram(model, test_sentences):
    print("N-gram: Preprocessing all test sentences...")
    s_time = time.time()
    clean_test_sentences = [ngram_preprocess_function(s) for s in tqdm(test_sentences, desc="N-gram Preprocessing")]
    print(f"N-gram preprocessing took {time.time() - s_time:.2f}s")

    print("N-gram: Transforming all test sentences (batch)...")
    s_time = time.time()
    X_all_test = model.sentence_vectorizer.transform(clean_test_sentences)
    print(f"N-gram batch transform took {time.time() - s_time:.2f}s")
    
    print("N-gram: Generating predictions sentence by sentence...")
    s_time = time.time()
    predictions = []
    feature_names = model.sentence_vectorizer.get_feature_names_out()

    for i in tqdm(range(X_all_test.shape[0]), desc="N-gram Generating"):
        ngram_indices = X_all_test[i].nonzero()[1]
        if len(ngram_indices) == 0:
            predictions.append(random.choice(model.implied_statements) if model.implied_statements else "No n-grams/fallback.")
            continue
        active_ngrams = [feature_names[idx] for idx in ngram_indices]
        predictions.append(model._generate_by_matching(active_ngrams))
    print(f"N-gram generation loop took {time.time() - s_time:.2f}s")
    return predictions

# --- TF-IDF + Word2Vec Model ---
class TfidfWord2VecSeq2Seq(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_vocab_size, dropout=0.3):
        super(TfidfWord2VecSeq2Seq, self).__init__(); self.encoder = nn.Sequential(nn.Linear(embedding_dim, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)); self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, output_vocab_size))
    def forward(self, x): return self.decoder(self.encoder(x))

def tfidf_w2v_prepare_input_vector(sentence_text, tfidf_vectorizer, word2vec_model, embedding_dim):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_vector_sparse = tfidf_vectorizer.transform([sentence_text.lower()])
    tfidf_vector_dense = tfidf_vector_sparse.toarray()[0]
    weighted_vector = np.zeros(embedding_dim); total_weight = 0
    for word in word_tokenize(sentence_text.lower()):
        if word in word2vec_model.wv:
            try:
                word_idx = list(feature_names).index(word)
                weight = tfidf_vector_dense[word_idx]
                weighted_vector += weight * word2vec_model.wv[word]; total_weight += weight
            except ValueError: pass
    if total_weight > 0: weighted_vector /= total_weight
    return weighted_vector

def tfidf_w2v_generate_text(model, input_vector, idx2word, max_length=20, temperature=0.8):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_vector).unsqueeze(0).to(DEVICE)
        output_logits = model(input_tensor).squeeze(0)
        output_probs = torch.softmax(output_logits / temperature, dim=0).cpu().numpy()
        words = []; sorted_indices = np.argsort(output_probs)[::-1]
        for idx in sorted_indices:
            word = idx2word.get(idx)
            if word and word not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']: words.append(word)
            if len(words) >= max_length: break
        return ' '.join(words[:max_length])

# --- Feedforward Model ---
class FeedforwardSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(FeedforwardSeq2Seq, self).__init__(); self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim * 2), nn.LeakyReLU(0.2), nn.BatchNorm1d(hidden_dim * 2), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2), nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout)); self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.LeakyReLU(0.2), nn.BatchNorm1d(hidden_dim * 2), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, output_dim), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x))

def ffnn_get_text_from_vector(vector, vectorizer, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    top_indices = vector.argsort()[-top_n:][::-1]
    selected_words = [feature_names[i] for i in top_indices if vector[i] > 0.1]
    return " ".join(selected_words)

# --- Vocabulary class for RNN model ---
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_count = {}
        self.n_words = 4  # Count default tokens
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1
            
    def add_sentence(self, sentence):
        for word in word_tokenize(sentence.lower()):
            self.add_word(word)
            
    def __len__(self):
        return self.n_words

# --- Recurrent NN Model ---
class RNNEncoder(nn.Module):
    def __init__(self,vs,ed,hd,nl=1,dr=0.5): super(RNNEncoder,self).__init__();self.emb=nn.Embedding(vs,ed);self.rnn=nn.LSTM(ed,hd,nl,dropout=dr if nl>1 else 0,bidirectional=True,batch_first=True);self.fc=nn.Linear(hd*2,hd);self.drp=nn.Dropout(dr)
    def forward(self,x,l): embd=self.drp(self.emb(x));pck=pack_padded_sequence(embd,l.cpu(),batch_first=True,enforce_sorted=False);o,(h,c)=self.rnn(pck);o,_=pad_packed_sequence(o,batch_first=True);h=torch.cat((h[-2,:,:],h[-1,:,:]),dim=1);c=torch.cat((c[-2,:,:],c[-1,:,:]),dim=1);h=torch.tanh(self.fc(h));c=torch.tanh(self.fc(c));h=h.unsqueeze(0).repeat(self.rnn.num_layers,1,1);c=c.unsqueeze(0).repeat(self.rnn.num_layers,1,1);return o,(h,c)

class RNNAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(RNNAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim + hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, h, eo):
        # h is shape [batch_size, 1, hidden_dim] or [num_layers, batch_size, hidden_dim]
        # Extract the hidden state correctly
        if isinstance(h, tuple):  # If h is a tuple (from LSTM)
            h = h[0]  # Get the hidden state, not the cell state
        
        # Make sure h has the right shape
        if h.dim() == 3 and h.size(0) > 1:  # [num_layers, batch_size, hidden_dim]
            h = h[-1].unsqueeze(0)  # Take the last layer and add a dim back
            
        bs, sl = eo.shape[0], eo.shape[1]
        h = h.permute(1, 0, 2)  # [batch_size, 1, hidden_dim]
        h = h.repeat(1, sl, 1)  # [batch_size, seq_len, hidden_dim]
        
        # Concatenate and compute attention
        ei = torch.cat((h, eo), dim=2)
        en = torch.tanh(self.attn(ei))
        a = self.v(en).squeeze(2)
        return torch.softmax(a, dim=1)

class RNNDecoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers=1, dropout=0.5):
        super(RNNDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.emb = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim + hidden_dim * 2, hidden_dim, n_layers, 
                          dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fco = nn.Linear(hidden_dim * 3 + embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.att = RNNAttention(hidden_dim)
        
    def forward(self, input, hidden, eo):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.emb(input))
        
        # Pass the hidden state directly to the attention mechanism
        a = self.att(hidden, eo)
        
        # Multiply attention weights by encoder outputs to get weighted context
        weighted = torch.bmm(a.unsqueeze(1), eo)
        weighted = weighted.squeeze(1)
        
        # Concatenate with embedded input for RNN input
        rnn_input = torch.cat((embedded, weighted.unsqueeze(1)), dim=2)
        
        # Pass through LSTM
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Combine output with weighted context vector and embedded input
        output = torch.cat((output.squeeze(1), weighted, embedded.squeeze(1)), dim=1)
        
        # Pass through linear layer to get prediction
        prediction = self.fco(output)
        
        return prediction, hidden

class RNNSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # Forward pass implementation
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_len)
        
        # First input to the decoder is the <SOS> token
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs
    
    def generate(self, src, src_len, trg_vocab, max_len=50):
        """Generate text from source sentence"""
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src, src_len)
            
            # Start with <SOS> token
            input = torch.tensor([trg_vocab.word2idx["<SOS>"]]).to(self.device)
            
            output_tokens = []
            
            # Generate tokens one by one
            for _ in range(max_len):
                output, hidden = self.decoder(input, hidden, encoder_outputs)
                pred_token = output.argmax(1).item()
                
                # Break if we predict <EOS>
                if pred_token == trg_vocab.word2idx["<EOS>"]:
                    break
                    
                # Add token to output if it's not a special token
                if pred_token not in [trg_vocab.word2idx["<PAD>"], trg_vocab.word2idx["<SOS>"], trg_vocab.word2idx["<UNK>"]]:
                    output_tokens.append(pred_token)
                
                # Use predicted token as next input
                input = torch.tensor([pred_token]).to(self.device)
            
            # Convert token IDs to words
            output_text = " ".join([trg_vocab.idx2word[token] for token in output_tokens])
            return output_text

def calculate_perplexity_rnn(model, dataloader_test, criterion, device, trg_vocab_size, pad_idx, model_name="RNN"):
    print(f"  Calculating Perplexity for {model_name}...")
    s_time = time.time()
    model.eval(); total_loss=0; total_tokens=0
    with torch.no_grad():
        for src, src_len, trg in tqdm(dataloader_test, desc=f"Perplexity {model_name}"):
            src,trg,src_len = src.to(device),trg.to(device),src_len.to(device) # src_len to device
            output_logits = model(src,src_len,trg,teacher_forcing_ratio=0.0)
            output_for_loss = output_logits[:,1:].reshape(-1,trg_vocab_size)
            trg_for_loss = trg[:,1:].reshape(-1)
            loss = criterion(output_for_loss,trg_for_loss)
            mask = (trg_for_loss != pad_idx)
            num_actual_tokens = mask.sum().item()
            if num_actual_tokens > 0: total_loss += loss.item()*num_actual_tokens; total_tokens += num_actual_tokens
    if total_tokens == 0: print("  No tokens for RNN perplexity."); return float('nan')
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    print(f"  Perplexity for {model_name} (CE Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}) calculated in {time.time() - s_time:.2f}s")
    return perplexity

# Add the rnn_tokenize function
def rnn_tokenize(text, vocab):
    """Tokenize text for RNN models"""
    tokens = []
    for word in word_tokenize(text.lower()):
        if word in vocab.word2idx:
            tokens.append(vocab.word2idx[word])
        else:
            tokens.append(vocab.word2idx["<UNK>"])
    return tokens

# --- Main Benchmarking Logic ---
def main_benchmark():
    print(f"Using device: {DEVICE}")
    print(f"Loading data from {DATA_PATH}...")
    df_full = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df_full)} examples.")
    
    # Use a consistent test split for all models
    # _, df_test = train_test_split(df_full, test_size=0.2, random_state=42) # Use all data if small
    df_test = df_full # If dataset is small, can use all for testing, or a dedicated test file
    if len(df_full) > 1000: # Example threshold for splitting
         _, df_test = train_test_split(df_full, test_size=0.2, random_state=42)
    print(f"Using {len(df_test)} examples for testing.")
    
    test_sentences = df_test['sentence'].tolist()
    true_implications = df_test['implied_statement'].tolist()

    results = {}
    metrics_log = {} # To store detailed metrics objects if needed

    # --- 1. Recurrent NN Model ---
    print("\n--- Benchmarking Recurrent NN Model ---")
    model_name = "RecurrentNN"
    rnn_model_weights_path = os.path.join(MODEL_SAVE_PATH, "recurrent_nn_model.pth")
    rnn_components_path = os.path.join(MODEL_SAVE_PATH, "recurrent_nn_components.pkl")

    if os.path.exists(rnn_model_weights_path) and os.path.exists(rnn_components_path):
        print("Loading RecurrentNN model and components...")
        s_time = time.time()
        with open(rnn_components_path, "rb") as f: components = pickle.load(f)
        src_vocab_loaded = components['input_vocab']; trg_vocab_loaded = components['output_vocab']
        embedding_dim, hidden_dim, n_layers, dropout = 256, 512, 2, 0.5
        
        # Fix: Handle different vocabulary implementations
        # Check if n_words attribute exists, if not, set it to length of word2idx dictionary
        if not hasattr(src_vocab_loaded, 'n_words'):
            src_vocab_loaded.n_words = len(src_vocab_loaded.word2idx)
        if not hasattr(trg_vocab_loaded, 'n_words'):
            trg_vocab_loaded.n_words = len(trg_vocab_loaded.word2idx)
        
        input_dim_rnn = len(src_vocab_loaded)
        output_dim_rnn = len(trg_vocab_loaded)
        
        print(f"RNN vocabulary sizes: input={input_dim_rnn}, output={output_dim_rnn}")
        
        # Create a new checkpoint with renamed keys to match the expected architecture
        checkpoint = torch.load(rnn_model_weights_path, map_location=DEVICE)
        
        # Create new state dicts with renamed keys
        if 'encoder' in checkpoint and 'decoder' in checkpoint:
            encoder_state_dict = {}
            for k, v in checkpoint['encoder'].items():
                # Map the keys from the saved model to the expected keys
                if k.startswith('embedding'):
                    encoder_state_dict[k.replace('embedding', 'emb')] = v
                elif k.startswith('dropout'):
                    encoder_state_dict[k.replace('dropout', 'drp')] = v
                else:
                    encoder_state_dict[k] = v
                
            decoder_state_dict = {}
            for k, v in checkpoint['decoder'].items():
                # Map the keys from the saved model to the expected keys
                if k.startswith('embedding'):
                    decoder_state_dict[k.replace('embedding', 'emb')] = v
                elif k.startswith('dropout'):
                    decoder_state_dict[k.replace('dropout', 'drp')] = v
                elif k.startswith('fc_out'):
                    decoder_state_dict[k.replace('fc_out', 'fco')] = v
                else:
                    decoder_state_dict[k] = v
            
            # Add this fix - rename the keys to match the current model architecture
            new_decoder_state_dict = {}
            for k, v in decoder_state_dict.items():
                if k.startswith('attention.'):
                    # Replace 'attention.' with 'att.'
                    new_key = 'att.' + k[10:]
                    new_decoder_state_dict[new_key] = v
                else:
                    new_decoder_state_dict[k] = v
            
            # Create the models with the expected architecture
            encoder = RNNEncoder(input_dim_rnn, embedding_dim, hidden_dim, n_layers, dropout)
            decoder = RNNDecoder(output_dim_rnn, embedding_dim, hidden_dim, n_layers, dropout)
            
            # Load the renamed state dicts
            encoder.load_state_dict(encoder_state_dict)
            decoder.load_state_dict(new_decoder_state_dict)
            
            model_rnn = RNNSeq2Seq(encoder, decoder, DEVICE).to(DEVICE)
        else:
            print("Error: Unexpected checkpoint format. Expected 'encoder' and 'decoder' keys.")
            model_rnn = None
        
        print(f"RecurrentNN loaded in {time.time() - s_time:.2f}s.")

        print("RecurrentNN: Generating predictions...")
        s_time = time.time()
        rnn_predictions = []
        for sentence in tqdm(test_sentences, desc="RNN Generating"):
            tokens = rnn_tokenize(sentence, src_vocab_loaded)
            tokens = [src_vocab_loaded.word2idx["<SOS>"]] + tokens + [src_vocab_loaded.word2idx["<EOS>"]]
            src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)
            src_len = torch.LongTensor([len(tokens)]).to(DEVICE) # Not .cpu() here
            rnn_predictions.append(model_rnn.generate(src_tensor, src_len, trg_vocab_loaded))
        pred_time = time.time() - s_time
        print(f"RecurrentNN predictions generated in {pred_time:.2f}s.")
        
        metrics_log[model_name] = {}
        try: metrics_log[model_name]["bleu"] = calculate_bleu(true_implications, rnn_predictions)
        except Exception as e: print(f"Error calc RNN BLEU: {e}"); metrics_log[model_name]["bleu"] = "Error"
        try: metrics_log[model_name]["rouge"] = calculate_rouge(true_implications, rnn_predictions)
        except Exception as e: print(f"Error calc RNN ROUGE: {e}"); metrics_log[model_name]["rouge"] = "Error"
        try: metrics_log[model_name]["bertscore"] = calculate_bertscore(true_implications, rnn_predictions)
        except Exception as e: print(f"Error calc RNN BERTScore: {e}"); metrics_log[model_name]["bertscore"] = "Error"

        print("RecurrentNN: Preparing data for perplexity...")
        test_src_token_ids = [[src_vocab_loaded.word2idx["<SOS>"]] + rnn_tokenize(s, src_vocab_loaded) + [src_vocab_loaded.word2idx["<EOS>"]] for s in df_test["sentence"]]
        test_trg_token_ids = [[trg_vocab_loaded.word2idx["<SOS>"]] + rnn_tokenize(s, trg_vocab_loaded) + [trg_vocab_loaded.word2idx["<EOS>"]] for s in df_test["implied_statement"]]
        test_src_lengths = torch.tensor([len(s) for s in test_src_token_ids])
        test_src_padded = pad_sequence([torch.tensor(s) for s in test_src_token_ids], batch_first=True, padding_value=src_vocab_loaded.word2idx["<PAD>"])
        test_trg_padded = pad_sequence([torch.tensor(s) for s in test_trg_token_ids], batch_first=True, padding_value=trg_vocab_loaded.word2idx["<PAD>"])
        rnn_test_dataset = TensorDataset(test_src_padded, test_src_lengths, test_trg_padded)
        rnn_test_loader = DataLoader(rnn_test_dataset, batch_size=min(16, len(df_test))) # Smaller batch for perplexity if needed
        criterion_rnn = nn.CrossEntropyLoss(ignore_index=trg_vocab_loaded.word2idx["<PAD>"])
        perplexity_rnn = calculate_perplexity_rnn(model_rnn, rnn_test_loader, criterion_rnn, DEVICE, len(trg_vocab_loaded), trg_vocab_loaded.word2idx["<PAD>"], model_name)
        
        results[model_name] = {
            "prediction_time_seconds": pred_time,
            "bleu": metrics_log[model_name]["bleu"],
            "rouge": metrics_log[model_name]["rouge"],
            "bertscore_f1": metrics_log[model_name]["bertscore"].get('f1', "Error") if isinstance(metrics_log[model_name]["bertscore"], dict) else "Error",
            "perplexity": perplexity_rnn
        }
        print(f"RecurrentNN results: {results[model_name]}")
    else:
        print(f"Recurrent NN model or components not found. Skipping.")
        results[model_name] = "Model or components not found"
        
    # --- 2. N-gram Model ---
    print("\n--- Benchmarking N-gram Model ---")
    model_name = "N-gram"
    ngram_model_path = os.path.join(MODEL_SAVE_PATH, "ngram_model.pkl")
    if os.path.exists(ngram_model_path):
        print(f"Loading N-gram model from {ngram_model_path}...")
        s_time = time.time()
        with open(ngram_model_path, "rb") as f:
            ngram_model = pickle.load(f)
        print(f"N-gram model loaded in {time.time() - s_time:.2f}s.")
        
        print("N-gram: Generating predictions...")
        s_time = time.time()
        ngram_predictions = predict_ngram(ngram_model, test_sentences)
        pred_time = time.time() - s_time
        print(f"N-gram predictions generated in {pred_time:.2f}s.")
        
        metrics_log[model_name] = {}
        try: metrics_log[model_name]["bleu"] = calculate_bleu(true_implications, ngram_predictions)
        except Exception as e: print(f"Error calc N-gram BLEU: {e}"); metrics_log[model_name]["bleu"] = "Error"
        try: metrics_log[model_name]["rouge"] = calculate_rouge(true_implications, ngram_predictions)
        except Exception as e: print(f"Error calc N-gram ROUGE: {e}"); metrics_log[model_name]["rouge"] = "Error"
        try: metrics_log[model_name]["bertscore"] = calculate_bertscore(true_implications, ngram_predictions)
        except Exception as e: print(f"Error calc N-gram BERTScore: {e}"); metrics_log[model_name]["bertscore"] = "Error"
        
        results[model_name] = {
            "prediction_time_seconds": pred_time,
            "bleu": metrics_log[model_name]["bleu"],
            "rouge": metrics_log[model_name]["rouge"],
            "bertscore_f1": metrics_log[model_name]["bertscore"].get('f1', "Error") if isinstance(metrics_log[model_name]["bertscore"], dict) else "Error",
            "perplexity": "N/A"
        }
        print(f"N-gram results: {results[model_name]}")
    else:
        print(f"N-gram model not found at {ngram_model_path}. Skipping.")
        results[model_name] = "Model not found"

    # --- 3. Transformer Model ---
    print("\n--- Benchmarking Transformer Model ---")
    model_name = "Transformer"
    if os.path.exists(TRANSFORMER_MODEL_PATH) and len(os.listdir(TRANSFORMER_MODEL_PATH)) > 3:
        print("Loading Transformer tokenizer and model...")
        s_time = time.time()
        transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_PATH)
        transformer_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSFORMER_MODEL_PATH).to(DEVICE)
        transformer_model.eval()
        print(f"Transformer model loaded in {time.time() - s_time:.2f}s.")
        
        print("Transformer: Generating predictions (batching where possible)...")
        s_time = time.time()
        transformer_predictions = []
        # Batching generation (tokenize all, then generate in batches if model supports)
        # For seq2seq .generate(), it's usually one by one, but tokenization can be batched.
        # Simple loop for now, can be optimized with DataLoader if very slow and VRAM allows.
        for sentence in tqdm(test_sentences, desc="Transformer Generating"):
            inputs = transformer_tokenizer("generate implication: " + sentence, return_tensors="pt", max_length=256, truncation=True, padding=True).to(DEVICE)
            outputs = transformer_model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True, no_repeat_ngram_size=2)
            transformer_predictions.append(transformer_tokenizer.decode(outputs[0], skip_special_tokens=True))
        pred_time = time.time() - s_time
        print(f"Transformer predictions generated in {pred_time:.2f}s.")

        metrics_log[model_name] = {}
        try: metrics_log[model_name]["bleu"] = calculate_bleu(true_implications, transformer_predictions)
        except Exception as e: print(f"Error calc Transformer BLEU: {e}"); metrics_log[model_name]["bleu"] = "Error"
        try: metrics_log[model_name]["rouge"] = calculate_rouge(true_implications, transformer_predictions)
        except Exception as e: print(f"Error calc Transformer ROUGE: {e}"); metrics_log[model_name]["rouge"] = "Error"
        try: metrics_log[model_name]["bertscore"] = calculate_bertscore(true_implications, transformer_predictions)
        except Exception as e: print(f"Error calc Transformer BERTScore: {e}"); metrics_log[model_name]["bertscore"] = "Error"
        
        hf_test_dataset = HFDataset.from_pandas(df_test)
        ce_loss = calculate_perplexity_hf(transformer_model, transformer_tokenizer, hf_test_dataset, model_name)
        perplexity_transformer = np.exp(ce_loss) if not np.isnan(ce_loss) else "Error"

        results[model_name] = {
            "prediction_time_seconds": pred_time,
            "bleu": metrics_log[model_name]["bleu"],
            "rouge": metrics_log[model_name]["rouge"],
            "bertscore_f1": metrics_log[model_name]["bertscore"].get('f1', "Error") if isinstance(metrics_log[model_name]["bertscore"], dict) else "Error",
            "perplexity": perplexity_transformer
        }
        print(f"Transformer results: {results[model_name]}")
    else:
        print(f"Transformer model not found at {TRANSFORMER_MODEL_PATH}. Skipping.")
        results[model_name] = "Model not found"

    # --- 4. TF-IDF + Word2Vec ---
    print("\n--- Benchmarking TF-IDF + Word2Vec Model ---")
    model_name = "TF-IDF_Word2Vec"
    tfidf_w2v_model_weights_path = os.path.join(MODEL_SAVE_PATH, "tfidf_word2vec.pth")
    tfidf_w2v_components_path = os.path.join(MODEL_SAVE_PATH, "tfidf_word2vec_components.pkl")

    if os.path.exists(tfidf_w2v_model_weights_path) and os.path.exists(tfidf_w2v_components_path):
        print("Loading TF-IDF+Word2Vec model and components...")
        s_time = time.time()
        with open(tfidf_w2v_components_path, "rb") as f: components = pickle.load(f)
        tfidf_vec_loaded = components['tfidf_vectorizer']; w2v_model_loaded = components['word2vec_model']
        word2idx_loaded = components['word2idx']; idx2word_loaded = components['idx2word']
        embedding_dim = w2v_model_loaded.vector_size; output_vocab_size = len(word2idx_loaded); hidden_dim = 256
        model_tfidf_w2v = TfidfWord2VecSeq2Seq(embedding_dim, hidden_dim, output_vocab_size).to(DEVICE)
        model_tfidf_w2v.load_state_dict(torch.load(tfidf_w2v_model_weights_path, map_location=DEVICE))
        model_tfidf_w2v.eval()
        print(f"TF-IDF+Word2Vec loaded in {time.time() - s_time:.2f}s.")

        print("TF-IDF+Word2Vec: Generating predictions...")
        s_time = time.time()
        tfidf_w2v_predictions = []
        for sentence in tqdm(test_sentences, desc="TF-IDF+W2V Generating"):
            input_vec = tfidf_w2v_prepare_input_vector(sentence, tfidf_vec_loaded, w2v_model_loaded, embedding_dim)
            tfidf_w2v_predictions.append(tfidf_w2v_generate_text(model_tfidf_w2v, input_vec, idx2word_loaded))
        pred_time = time.time() - s_time
        print(f"TF-IDF+Word2Vec predictions generated in {pred_time:.2f}s.")
        
        metrics_log[model_name] = {}
        try: metrics_log[model_name]["bleu"] = calculate_bleu(true_implications, tfidf_w2v_predictions)
        except Exception as e: print(f"Error calc TFIDF W2V BLEU: {e}"); metrics_log[model_name]["bleu"] = "Error"
        try: metrics_log[model_name]["rouge"] = calculate_rouge(true_implications, tfidf_w2v_predictions)
        except Exception as e: print(f"Error calc TFIDF W2V ROUGE: {e}"); metrics_log[model_name]["rouge"] = "Error"
        try: metrics_log[model_name]["bertscore"] = calculate_bertscore(true_implications, tfidf_w2v_predictions)
        except Exception as e: print(f"Error calc TFIDF W2V BERTScore: {e}"); metrics_log[model_name]["bertscore"] = "Error"

        results[model_name] = {
            "prediction_time_seconds": pred_time,
            "bleu": metrics_log[model_name]["bleu"],
            "rouge": metrics_log[model_name]["rouge"],
            "bertscore_f1": metrics_log[model_name]["bertscore"].get('f1', "Error") if isinstance(metrics_log[model_name]["bertscore"], dict) else "Error",
            "perplexity": "N/A"
        }
        print(f"TF-IDF_Word2Vec results: {results[model_name]}")
    else:
        print(f"TF-IDF+Word2Vec model or components not found. Skipping.")
        results[model_name] = "Model or components not found"

    # --- 5. Feedforward NN ---
    print("\n--- Benchmarking Feedforward NN Model ---")
    model_name = "FeedforwardNN"
    ffnn_model_weights_path = os.path.join(MODEL_SAVE_PATH, "forward.pth")
    ffnn_components_path = os.path.join(MODEL_SAVE_PATH, "forward_components.pkl")

    if os.path.exists(ffnn_model_weights_path) and os.path.exists(ffnn_components_path):
        print("Loading FeedforwardNN model and components...")
        s_time = time.time()
        with open(ffnn_components_path, "rb") as f: components = pickle.load(f)
        input_vectorizer_loaded = components['input_vectorizer']; output_vectorizer_loaded = components['output_vectorizer']
        input_dim = len(input_vectorizer_loaded.get_feature_names_out()); output_dim = len(output_vectorizer_loaded.get_feature_names_out()); hidden_dim = 256
        model_ffnn = FeedforwardSeq2Seq(input_dim, hidden_dim, output_dim).to(DEVICE)
        model_ffnn.load_state_dict(torch.load(ffnn_model_weights_path, map_location=DEVICE))
        model_ffnn.eval()
        print(f"FeedforwardNN loaded in {time.time() - s_time:.2f}s.")

        print("FeedforwardNN: Generating predictions...")
        s_time = time.time()
        ffnn_predictions = []
        print("  FeedforwardNN: Vectorizing all test inputs...")
        s_vec_time = time.time()
        test_input_vecs_sparse = input_vectorizer_loaded.transform(test_sentences)
        test_input_vecs_dense = torch.FloatTensor(test_input_vecs_sparse.toarray()).to(DEVICE)
        print(f"  FeedforwardNN: Vectorization took {time.time() - s_vec_time:.2f}s")
        
        print("  FeedforwardNN: Model prediction (batch)...")
        s_pred_batch_time = time.time()
        with torch.no_grad(): test_output_vecs = model_ffnn(test_input_vecs_dense).cpu().numpy()
        print(f"  FeedforwardNN: Model prediction batch took {time.time() - s_pred_batch_time:.2f}s")

        print("  FeedforwardNN: Converting output vectors to text...")
        s_decode_time = time.time()
        for vec in tqdm(test_output_vecs, desc="FFNN Decoding"): ffnn_predictions.append(ffnn_get_text_from_vector(vec, output_vectorizer_loaded))
        print(f"  FeedforwardNN: Decoding took {time.time() - s_decode_time:.2f}s")
        pred_time = time.time() - s_time
        print(f"FeedforwardNN predictions generated in {pred_time:.2f}s (Total).")
        
        metrics_log[model_name] = {}
        try: metrics_log[model_name]["bleu"] = calculate_bleu(true_implications, ffnn_predictions)
        except Exception as e: print(f"Error calc FFNN BLEU: {e}"); metrics_log[model_name]["bleu"] = "Error"
        try: metrics_log[model_name]["rouge"] = calculate_rouge(true_implications, ffnn_predictions)
        except Exception as e: print(f"Error calc FFNN ROUGE: {e}"); metrics_log[model_name]["rouge"] = "Error"
        try: metrics_log[model_name]["bertscore"] = calculate_bertscore(true_implications, ffnn_predictions)
        except Exception as e: print(f"Error calc FFNN BERTScore: {e}"); metrics_log[model_name]["bertscore"] = "Error"

        results[model_name] = {
            "prediction_time_seconds": pred_time,
            "bleu": metrics_log[model_name]["bleu"],
            "rouge": metrics_log[model_name]["rouge"],
            "bertscore_f1": metrics_log[model_name]["bertscore"].get('f1', "Error") if isinstance(metrics_log[model_name]["bertscore"], dict) else "Error",
            "perplexity": "N/A"
        }
        print(f"FeedforwardNN results: {results[model_name]}")
    else:
        print(f"Feedforward NN model or components not found. Skipping.")
        results[model_name] = "Model or components not found"
        
    # --- Save results ---
    results_path = "benchmark_results.json"
    print(f"\nSaving benchmark results to {results_path}...")
    with open(results_path, 'w') as f:
        # Custom JSON encoder to handle numpy types if they sneak in (e.g. np.float32)
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        json.dump(results, f, indent=4, cls=NpEncoder)
    print(f"Benchmark results saved to {results_path}")

    # --- Print summary table ---
    print("\n--- Benchmark Summary ---")
    summary_data = []
    model_names_order = ['RecurrentNN', 'N-gram', 'Transformer', 'TF-IDF_Word2Vec', 'FeedforwardNN']
    for name in model_names_order:
        if name in results:
            if isinstance(results[name], str): row = {"model_name": name, "status": results[name]}
            else: row = {"model_name": name, "status": "Success", **results[name]}
        else: row = {"model_name": name, "status": "Not run"}
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data).set_index("model_name")
    all_cols_display = ["prediction_time_seconds", "bleu", "rouge1", "rougeL", "bertscore_f1", "perplexity", "status"]
    
    if 'rouge' in summary_df.columns:
        def extract_rouge(x, key):
            if isinstance(x, dict): return x.get(key, np.nan)
            return np.nan
        summary_df['rouge1'] = summary_df['rouge'].apply(lambda x: extract_rouge(x, 'rouge1'))
        summary_df['rougeL'] = summary_df['rouge'].apply(lambda x: extract_rouge(x, 'rougeL'))
        summary_df = summary_df.drop(columns=['rouge'], errors='ignore')
    else: summary_df['rouge1'] = np.nan; summary_df['rougeL'] = np.nan

    final_summary_df = pd.DataFrame(index=summary_df.index)
    for col in all_cols_display:
        final_summary_df[col] = summary_df.get(col, pd.Series(np.nan, index=summary_df.index))

    # Fix: Use a formatter function instead of float_format string
    print(final_summary_df.to_string(formatters={col: lambda x: f"{float(x):.4f}" if isinstance(x, (int, float)) else x 
                                               for col in final_summary_df.columns if col != "status"}))

    # Clean up temp Hugging Face Trainer directory
    if os.path.exists(TEMP_EVAL_DIR):
        import shutil
        try:
            shutil.rmtree(TEMP_EVAL_DIR)
            print(f"Cleaned up temporary directory: {TEMP_EVAL_DIR}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {TEMP_EVAL_DIR}: {e}")

if __name__ == "__main__":
    main_benchmark()