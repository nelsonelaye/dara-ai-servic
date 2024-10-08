from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

model_name = "facebook/blenderbot-400M-distill"
tokenizer= BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# load dataset
data = load_dataset("json", data_files={'train': '../dataset/dataset.json'}, field='data')




# tokenize data
def tokenize_function(value):
    input_text = value['input']
    response_text = value['response']
    
    # tokenize both input and output
    model_inputs = tokenizer(input_text, max_length=128, truncation=True, padding="max_length")
    
    # tokenize response
    labels = tokenizer(response_text, max_length=128, truncation=True, padding="max_length").input_ids
    
    model_inputs['labels'] = labels
    
    return model_inputs


tokenized_data = data.map(tokenize_function, batched=True)

tokenized_data = tokenized_data.remove_columns(["input", "response"])

split_data = tokenized_data['train'].train_test_split(test_size=0.1)  # 10% for evaluation

train_dataset = split_data['train']
eval_dataset = split_data['test']

# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="../blenderbot_finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,  # Enable mixed precision training if using a GPU
    remove_unused_columns=False
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Training dataset
    eval_dataset=eval_dataset, 
    tokenizer=tokenizer
)

# Start fine-tuning the model
trainer.train()

# save fine-tuned model and tokenizer
model.save_pretrained("../blenderbot_finetuned")
tokenizer.save_pretrained("../blenderbot_finetuned")



# Non-default generation parameters: 