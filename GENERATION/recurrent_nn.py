import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from rouge import Rouge
import os
import pickle
import random
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                          dropout=dropout if n_layers > 1 else 0,
                          bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        outputs, (hidden, cell) = self.rnn(packed)
        
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        
        hidden = torch.tanh(self.fc(hidden))
        cell = torch.tanh(self.fc(cell))
        
        hidden = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(self.n_layers, 1, 1)
        
        return outputs, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear((hidden_dim * 2) + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.permute(1, 0, 2)
        
        hidden = hidden.repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        attention = self.v(energy).squeeze(2)
        
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.5):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_dim)
        self.rnn = nn.LSTM(embedding_dim + hidden_dim * 2, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 3 + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden[0:1], encoder_outputs)
        a = a.unsqueeze(1)
        
        weighted = torch.bmm(a, encoder_outputs)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        output = torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1)
        
        prediction = self.fc_out(output)
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            
            outputs[:, t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.argmax(1)
            
            input = trg[:, t] if teacher_force else top1
            
        return outputs


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_count = 4
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.word_count
            self.idx2word[self.word_count] = word
            self.word_count += 1
    
    def __len__(self):
        return self.word_count


def tokenize(text, vocab, is_adding=False):
    tokens = text.lower().split()
    if is_adding:
        for token in tokens:
            vocab.add_word(token)
    return [vocab.word2idx.get(token, vocab.word2idx["<UNK>"]) for token in tokens]


def calculate_bleu(references, hypotheses):
    try:
        
        tokenized_refs = [[ref.lower().split()] for ref in references]
        tokenized_hyps = [hyp.lower().split() for hyp in hypotheses]
        
        try:
            smoothie = SmoothingFunction().method1
            bleu_score = corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smoothie)
        except TypeError:
            print("Using alternative BLEU calculation due to NLTK version incompatibility")
            bleu_score = simple_bleu_calculation(references, hypotheses)
        
        return bleu_score
    
    except ImportError:
        print("NLTK not found. Using simple BLEU implementation.")
        return simple_bleu_calculation(references, hypotheses)


def simple_bleu_calculation(references, hypotheses):
    total_precision = 0.0
    count = 0
    
    for ref, hyp in zip(references, hypotheses):
        if len(hyp.strip()) == 0:
            continue
            
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        
        matches = sum(1 for token in hyp_tokens if token in ref_tokens)
        precision = matches / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
        
        total_precision += precision
        count += 1
    
    return total_precision / count if count > 0 else 0.0


def calculate_rouge(references, hypotheses):
    rouge = Rouge()
    
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


def main():
    data = pd.read_csv("../data/gen.csv")
    
    src_vocab = Vocabulary()
    trg_vocab = Vocabulary()
    
    for sentence in data["sentence"]:
        tokenize(sentence, src_vocab, is_adding=True)
    
    for statement in data["implied_statement"]:
        tokenize(statement, trg_vocab, is_adding=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = len(src_vocab)
    output_dim = len(trg_vocab)
    embedding_dim = 256
    hidden_dim = 512
    n_layers = 2
    dropout = 0.5
    
    encoder = Encoder(input_dim, embedding_dim, hidden_dim, n_layers, dropout)
    decoder = Decoder(output_dim, embedding_dim, hidden_dim, n_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    model_path = "models/recurrent_nn_model.pth"
    os.makedirs("models", exist_ok=True)
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No existing model found. Training new model...")
        
        src_tokens = []
        trg_tokens = []
        
        for sentence in data["sentence"]:
            tokens = tokenize(sentence, src_vocab)
            src_tokens.append([src_vocab.word2idx["<SOS>"]] + tokens + [src_vocab.word2idx["<EOS>"]])
        
        for statement in data["implied_statement"]:
            tokens = tokenize(statement, trg_vocab)
            trg_tokens.append([trg_vocab.word2idx["<SOS>"]] + tokens + [trg_vocab.word2idx["<EOS>"]])
        
        train_src, val_src, train_trg, val_trg = train_test_split(src_tokens, trg_tokens, test_size=0.2, random_state=42)
        
        train_src_lengths = [len(s) for s in train_src]
        val_src_lengths = [len(s) for s in val_src]
        
        train_src_padded = pad_sequence([torch.tensor(s) for s in train_src], batch_first=True, padding_value=src_vocab.word2idx["<PAD>"])
        val_src_padded = pad_sequence([torch.tensor(s) for s in val_src], batch_first=True, padding_value=src_vocab.word2idx["<PAD>"])
        
        train_trg_padded = pad_sequence([torch.tensor(s) for s in train_trg], batch_first=True, padding_value=trg_vocab.word2idx["<PAD>"])
        val_trg_padded = pad_sequence([torch.tensor(s) for s in val_trg], batch_first=True, padding_value=trg_vocab.word2idx["<PAD>"])
        
        train_dataset = TensorDataset(train_src_padded, torch.tensor(train_src_lengths), train_trg_padded)
        val_dataset = TensorDataset(val_src_padded, torch.tensor(val_src_lengths), val_trg_padded)
        
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.word2idx["<PAD>"])
        
        num_epochs = 20
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            
            for src, src_len, trg in train_loader:
                src = src.to(device)
                src_len = src_len.to('cpu')
                trg = trg.to(device)
                
                optimizer.zero_grad()
                
                output = model(src, src_len, trg)
                
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)
                
                loss = criterion(output, trg)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for src, src_len, trg in val_loader:
                    src = src.to(device)
                    src_len = src_len.to('cpu')
                    trg = trg.to(device)
                    
                    output = model(src, src_len, trg, 0)
                    
                    output_dim = output.shape[-1]
                    output = output[:, 1:].reshape(-1, output_dim)
                    trg = trg[:, 1:].reshape(-1)
                    
                    loss = criterion(output, trg)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)
                print(f"Model saved with validation loss: {val_loss:.4f}")
    
    def generate_implied_statement(sentence, max_length=50):
        model.eval()
        
        tokens = tokenize(sentence, src_vocab)
        tokens = [src_vocab.word2idx["<SOS>"]] + tokens + [src_vocab.word2idx["<EOS>"]]
        
        src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        src_len = torch.tensor([len(tokens)]).to('cpu')
        
        with torch.no_grad():
            encoder_outputs, (hidden, cell) = model.encoder(src_tensor, src_len)
            
            trg_idx = [trg_vocab.word2idx["<SOS>"]]
            
            for _ in range(max_length):
                trg_tensor = torch.tensor([trg_idx[-1]]).to(device)
                
                with torch.no_grad():
                    output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
                
                pred_token = output.argmax(1).item()
                
                if pred_token == trg_vocab.word2idx["<EOS>"]:
                    break
                
                trg_idx.append(pred_token)
            
        implied_statement = [trg_vocab.idx2word[i] for i in trg_idx[1:]]
        
        return " ".join(implied_statement)
    
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
    
    print("Enter a sentence to generate an implied statement (type 'q' to quit):")
    while True:
        sentence = input("> ")
        if sentence.lower() == 'q':
            break
        
        implied = generate_implied_statement(sentence)
        print(f"Implied statement: {implied}")

    os.makedirs("models", exist_ok=True)
    
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict()
    }, "models/recurrent_nn_model.pth")
    
    components = {
        'input_vocab': src_vocab,
        'output_vocab': trg_vocab
    }
    
    with open("models/recurrent_nn_components.pkl", "wb") as f:
        pickle.dump(components, f)
    
    print("Model and components saved successfully.")


def load_and_use_model(input_sentence):
    src_vocab = Vocabulary()
    trg_vocab = Vocabulary()
    
    data = pd.read_csv("../data/gen.csv")
    for sentence in data["sentence"]:
        tokenize(sentence, src_vocab, is_adding=True)
    for statement in data["implied_statement"]:
        tokenize(statement, trg_vocab, is_adding=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = len(src_vocab)
    output_dim = len(trg_vocab)
    embedding_dim = 256
    hidden_dim = 512
    n_layers = 2
    dropout = 0.5
    
    encoder = Encoder(input_dim, embedding_dim, hidden_dim, n_layers, dropout)
    decoder = Decoder(output_dim, embedding_dim, hidden_dim, n_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    model.load_state_dict(torch.load("models/recurrent_nn_model.pth"))
    model.eval()
    
    tokens = tokenize(input_sentence, src_vocab)
    tokens = [src_vocab.word2idx["<SOS>"]] + tokens + [src_vocab.word2idx["<EOS>"]]
    
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    src_len = torch.tensor([len(tokens)]).to('cpu')
    
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor, src_len)
        
        trg_idx = [trg_vocab.word2idx["<SOS>"]]
        
        for _ in range(50):
            trg_tensor = torch.tensor([trg_idx[-1]]).to(device)
            
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
            
            pred_token = output.argmax(1).item()
            
            if pred_token == trg_vocab.word2idx["<EOS>"]:
                break
            
            trg_idx.append(pred_token)
        
        implied_statement = [trg_vocab.idx2word[i] for i in trg_idx[1:]]
        
        return " ".join(implied_statement)


if __name__ == "__main__":
    main()