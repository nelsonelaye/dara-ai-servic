from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

model_name = "facebook/blenderbot-400M-distill"
# fine_tuned_model_name = "blenderbot_finetuned"

model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)

user_input = "How can I save money in Nigeria?"

def chat(query):
    inputs = tokenizer(query, return_tensors="pt")
    # replys = model.generate(**inputs)
    # response = tokenizer.batch_decode(replys)
    generated_ids =  model.generate(inputs["input_ids"])
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # print(response)
    return response
    

# print(chat(user_input))