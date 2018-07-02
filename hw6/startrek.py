import torch
import torch.nn as nn
import torch.optim as optim
import string
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pdb

def getData(file='./star_trek_transcripts_all_episodes_f.csv'):
    all_lines = []
    with open(file, 'r') as f:
        for line in f:
            v = line.strip().replace('=','').replace('/',' ').replace('+',' ').replace('(',' ') \
              .replace('[',' ').replace(')',' ').replace(']',' ').replace(', ','<>').split(',')
            for w in v:
                if len(w)>1:
                    # commas followed by spaces are legit parts of sentences
                    w = w.replace('<>', ', ')
                    all_lines.append(w)
    return all_lines

# helper functions

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    idx = all_letters.find(letter)
    if (idx==-1):
        print('cannot find letter')
        print(letter)
    return idx

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_letters):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_letters = num_letters
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.1)
        self.fc = nn.Linear(hidden_size, num_letters)
        self.softmax = nn.LogSoftmax(dim=1)
        self.temperature = 1
    
    def setTemperature(self, temp):
        self.temperature = temp
    
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out.squeeze(1))
        out = self.softmax(out/self.temperature)
        return out, hidden
    
    def initHidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_size),
                torch.zeros(self.num_layers, 1, self.hidden_size))

def train(model, data, criterion, optimizer):
    running_loss = 0
    running_correct = 0
    running_count = 0
    np.random.shuffle(train_lines)
    
    model.train()
    hidden = model.initHidden()

    for idx, line in enumerate(data, start=1):
        model.zero_grad()
        input_line = lineToTensor(line)
        target = targetTensor(line)

        output, _ = model(input_line, hidden)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss
        _, pred = output.max(1)
        running_correct += torch.eq(pred, target).sum().item()
        running_count += len(line)
    
    epoch_loss = running_loss/len(data)
    epoch_acc = running_correct/running_count
    
    print('Train: {}/{} (acc: {:.4f}), loss: {:.4f}'.format(running_correct, running_count, epoch_acc, epoch_loss))
    
    return epoch_loss, epoch_acc
    

def test(model, data, criterion):
    running_loss = 0
    running_correct = 0
    running_count = 0
    
    model.eval()
    hidden = model.initHidden()
    
    for idx, line in enumerate(data, start=1):
        input_line = lineToTensor(line)
        target = targetTensor(line)

        output, _ = model(input_line, hidden)
        loss = criterion(output, target)

        running_loss += loss
        _, pred = output.max(1)
        running_correct += torch.eq(pred, target).sum().item()
        running_count += len(line)
    
    epoch_loss = running_loss/len(data)
    epoch_acc = running_correct/running_count
    
    print('Test: {}/{} (acc: {:.4f}), loss: {:.4f}'.format(running_correct, running_count, epoch_acc, epoch_loss))
    
    return epoch_loss, epoch_acc

def sample(model, startchar='A'):
    model.eval()
    in_line = lineToTensor(startchar)
    out_line = startchar
    hidden = model.initHidden()
    
    for i in range(100):
        out, hidden = model(in_line, hidden)
        c = torch.distributions.categorical.Categorical(logits=out)
        pred = c.sample()
        pred = pred.item()
        if pred == n_letters-1:  # EOS
            break
        else:
            letter = all_letters[pred]
            out_line += letter
            in_line = lineToTensor(letter)
    return out_line

def sampleMultiple(model, n=15, temperature=0.5):
    model.eval()
    model.setTemperature(0.5)
    startChars = "ABCDEFGHIJKLMNOPRSTUVWZ"
    n_chars = len(startChars)
    lines = []
    for i in range(n):
        start = startChars[np.random.randint(n_chars)]
        line = sample(model, start)
        lines.append(line)
        print(line)
    model.setTemperature(1) # to continue training
    return lines

all_letters = string.ascii_letters + string.digits + string.punctuation + " "
n_letters = len(all_letters) + 1  # +1 for end of sentence marker

all_lines = getData()
# all_lines = all_lines[0:10] ### for testing
n_lines = len(all_lines)
np.random.shuffle(all_lines)
train_lines = all_lines[0:int(0.8*n_lines)]
test_lines = all_lines[int(0.8*n_lines):]

model = LSTM(n_letters, 100, 2, n_letters)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1)
scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.2)

start = time.time()
train_losses = []
test_losses = []
best_loss = 100

with open('startrek_generated.txt', 'w') as f:
    for epoch in range(10):
        print('{} Epoch {}'.format(timeSince(start), epoch))

        train_loss, train_acc = train(model, train_lines, criterion, optimizer)
        test_loss, test_acc = test(model, test_lines, criterion)

        if (test_loss < best_loss):
            torch.save(model.state_dict(), 'model.pt')

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        f.write('Epoch {}\n'.format(epoch))
        f.write('\n'.join(sampleMultiple(model))+'\n\n')

plt.figure()
plt.plot(train_losses)
plt.plot(test_losses)
plt.savefig('losses.png')