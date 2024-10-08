from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "Replace me by any text you'd like."
encoded_input = tokenizer.encode(text, return_tensors='pt')
output = model.generate(encoded_input, max_length=100, do_sample=True, temperature=0.7, top_k=50)

print(tokenizer.decode(output[0], skip_special_tokens=True))