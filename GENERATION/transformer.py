import pandas as pd
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import evaluate
import torch
import torch.cuda as cuda
import warnings


print("✅ CUDA is available:", cuda.is_available(), flush=True)
if cuda.is_available():
    print("🖥️  GPU Name:", cuda.get_device_name(0), flush=True)
else:
    print("🖥️  CPU Name:", "CPU", flush=True)

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
rouge = evaluate.load("rouge")

output_dir = "gen_transformers"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    eval_steps=500,
    save_strategy="epoch",
    save_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    predict_with_generate=True,
    generation_max_length=64,
    fp16=True,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
)

def preprocess_function(examples):
    inputs = ["generate implication: " + sentence for sentence in examples["sentence"]]
    
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    
    labels = tokenizer(text_target=examples["implied_statement"], max_length=64, truncation=True, padding="max_length")
    
    model_inputs["labels"] = [[l if l != tokenizer.pad_token_id else -100 for l in label] for label in labels["input_ids"]]
    
    return model_inputs

def compute_metrics(pred):
    try:
        labels_ids = pred.label_ids.copy()
        
        pred_ids = pred.predictions
        
        vocab_size = tokenizer.vocab_size
        
        if len(pred_ids.shape) > 2:
            pred_ids = np.argmax(pred_ids, axis=-1)
            
        pred_ids = np.clip(pred_ids, 0, vocab_size-1)
        
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        
        try:
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        except Exception as e:
            warnings.warn(f"Error decoding predictions: {str(e)}")
            pred_str = ["" for _ in range(len(labels_ids))]
            
        try:
            label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        except Exception as e:
            warnings.warn(f"Error decoding labels: {str(e)}")
            label_str = ["" for _ in range(len(labels_ids))]
        
        try:
            rouge_output = rouge.compute(predictions=pred_str, references=label_str, use_stemmer=True)
            
            results = {
                "rouge1": rouge_output["rouge1"] if isinstance(rouge_output["rouge1"], (float, int)) else rouge_output["rouge1"].mid.fmeasure,
                "rouge2": rouge_output["rouge2"] if isinstance(rouge_output["rouge2"], (float, int)) else rouge_output["rouge2"].mid.fmeasure,
                "rougeL": rouge_output["rougeL"] if isinstance(rouge_output["rougeL"], (float, int)) else rouge_output["rougeL"].mid.fmeasure
            }
            return results
            
        except Exception as e:
            warnings.warn(f"Error calculating ROUGE: {str(e)}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            
    except Exception as e:
        warnings.warn(f"Error in compute_metrics: {str(e)}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

def test(tokenizer, model):
    test_sentences = [
        "capitalism like communism is an enemy of the white race",
        "The visa policy is allowing more immigrants",
        "Pakistani people are moving into our area",
        "Democrats have invented new conspiracy theories",
        "Capitalism is destroying our society",
        "if you are white in that state watch yourself !"
    ]

    print("\n=== Testing with sample sentences ===\n")
    
    for sentence in test_sentences:
        try:
            input_text = "generate implication: " + sentence
            inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda" if torch.cuda.is_available() else "cpu")
            
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            
            output = model.generate(
                **inputs, 
                max_length=64,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Original: \"{sentence}\"")
            print(f"Implied: \"{decoded}\"")
            print("-" * 60)
        except Exception as e:
            print(f"Error testing sentence: \"{sentence}\"")
            print(f"Error: {str(e)}")
            print("-" * 60)

def main(path="../data/gen.csv"):
    df = pd.read_csv(path)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["test"]
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("Training the model...")
    trainer.train()
    
    print("Evaluating the model...")
    try:
        res = trainer.evaluate()
        print(res)
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        print("Skipping full evaluation due to error.")
    
    print("Saving the model...")
    trainer.save_model(output_dir) 
    tokenizer.save_pretrained(output_dir)
    
    print("Testing the model...")
    try:
        trained_model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
        trained_model.to("cuda" if torch.cuda.is_available() else "cpu")
        test(tokenizer, trained_model)
    except Exception as e:
        print(f"Error loading trained model: {str(e)}")
        print("Testing with the original model instead.")
        test(tokenizer, model)
    
def load_and_test(model_path="gen_transformers"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"🖥️  Using device: {device}")
        
        test(tokenizer, model)
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("Trying to load the base T5 model instead...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
            model.to(device)
            test(tokenizer, model)
        except Exception as e:
            print(f"❌ Error loading base model: {str(e)}")

if __name__ == "__main__":
    main()
