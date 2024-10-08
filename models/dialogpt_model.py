from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

for step in range(5):
    new_user_input = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors="pt")
    
    bot_input = torch.cat([chat_history, new_user_input], dim=1) if step > 0 else new_user_input
    
    attention_mask = bot_input.new_ones(bot_input.shape)
    
    chat_history = model.generate(bot_input, max_length=1000, pad_token_id=tokenizer.eos_token_id,  attention_mask=attention_mask)
    
    print("DialoGPT: {}".format(tokenizer.decode(chat_history[:, bot_input.shape[-1]:][0], skip_special_tokens=True)))