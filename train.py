import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

with open("personality_dataset.json", "r", encoding="utf-8") as f:
    dataset=json.load(f)
dataset=Dataset.from_list(dataset)

model_name="EleutherAI/gpt-neo-125M"

model=AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token  

def preprocess_function(example):
    encoded=tokenizer(
        f"Input: {example['input']} Output: {example['output']}",
        truncation=True,
        padding="max_length",
        max_length=256, 
        return_tensors="pt"
    )

    input_ids=encoded["input_ids"].squeeze(0)
    attention_mask=encoded["attention_mask"].squeeze(0)
    labels=input_ids.clone()

    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_dataset=dataset.map(preprocess_function, remove_columns=dataset.column_names)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

split_dataset=tokenized_dataset.train_test_split(train_size=0.95)
train_dataset=split_dataset["train"]
eval_dataset=split_dataset["test"]

training_args=TrainingArguments(
    output_dir="./llm-personality-model",
    evaluation_strategy="epoch",
    learning_rate=3e-5,  
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,
    num_train_epochs=5, 
    save_strategy="epoch",
    weight_decay=0.01,
    logging_dir='./logs',
    push_to_hub=False
)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)
trainer.train()

trainer.save_model("./llm-personality-model")
tokenizer.save_pretrained("./llm-personality-model")