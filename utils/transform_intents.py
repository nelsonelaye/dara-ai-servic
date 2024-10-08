import json

file_data = open('../dataset/finance_intents.json')

file_data = json.load(file_data)

intents = file_data['intents']


data = []

for i in intents:
    for q, r in zip(i['patterns'], i['responses']):
        data.append({"input": q, "response": r})

        json_data = json.dumps({"data": data}, indent=4)
        new_file = open("dataset.json", "w")
        
        new_file.write(json_data)
        # print(json_data)
    

