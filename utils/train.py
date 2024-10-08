import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.model import NeuralNet

with open('../dataset/finance_intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags =[] # different patterns along with their text
xy =[] # contain patters and text

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w) # extend here spreads the array of tokens into single words 
        xy.append((w, tag)) # add the pattern and it corresponding tag


ignore_words = ['?', '!', ',', '.']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words)) # return uniqe words
tags = sorted(set(tags)) # return uniqe tags
# print(tags)



#  train data
x_train = [] # x usually represents feature or input data
y_train = [] # y usually represents output data or prediction

for (pattarn_sentence, tag) in xy:
    bag = bag_of_words(pattarn_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


# create pytorch dataset from training data

# create dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples  = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

# hyperparameters
batch_size = 8 
hidden_size = 8
output_size = len(tags)
input_size= len(x_train[0]) # len of all words
learning_rate = 0.001
num_epochs = 1000

# print(input_size, len(all_words))
# print(output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = NeuralNet(input_size, hidden_size, output_size).to(device) # why puh model to device?

# loss and optmizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optiizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. File saved to {FILE}')