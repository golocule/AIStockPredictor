# from tradingview_ta import TA_Handler, Interval, Exchange
import time
from datetime import datetime
import ccxt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

exchanges = ccxt.exchanges
# print(exchanges)

exchange = ccxt.coinbaseadvanced()

# code to gather candlestick data and store in a file (candles.npy)
'''
symbol = 'BTC/USD'
timeframe = '5m'
limit = 1000

# date to stop gathering data
stopDate = int(datetime(2024, 10, 1).timestamp() * 1000)

#date to start gathering data
currDate = int(datetime(2022, 2, 1).timestamp() * 1000)

candleData = []

while currDate < stopDate:
    candlesticks = exchange.fetch_ohlcv(symbol, timeframe, since=currDate, limit=1000)
    # print(len(candlesticks))
    if len(candlesticks) > 0:
        lastCandle = candlesticks[len(candlesticks) - 1]
        # print(lastCandle)
        for i in range(len(candlesticks)):
            candleData.append(candlesticks[i])

        currDate = candlesticks[-1][0] + 50000

print(len(candleData))

np.save('candles.npy', candleData)
'''
# ohlcv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

historicalCandles = np.load('candles.npy').tolist()
print(len(historicalCandles))

# for normalizing try logarithmic or sections (couple days probs)

def sectionNormalize(candleList, sectionSize):
    candleList = np.array(candleList)
    normalizedList = np.copy(candleList)

    for i in range(len(candleList) - sectionSize):
        section = candleList[i:i + sectionSize]

        for j in range(section.shape[1]):
            sectMin = np.min(section[:, j])
            sectMax = np.max(section[:, j])

            if sectMax - sectMin == 0:
                normalizedList[i, j] = 1
            else:
                normalizedList[i, j] = (candleList[i, j] - sectMin) / (sectMax - sectMin)

    normalizedList = normalizedList[0:(len(candleList - 1) - sectionSize)]
    return normalizedList

def maxNormalize(candleList, max):
    candleList = np.array(candleList)
    normalizedList = np.copy(candleList)
    min = 0
    # change this for when bitcoin goes to crazy heights

    for i in range(len(candleList)):

        section = candleList[i]
        for j in range(6):
            # sectMin = np.min(section[:, j])
            # sectMax = np.max(section[:, j])

            normalizedList[i, j] = (candleList[i, j] - min) / (max - min)

    normalizedList = normalizedList[0:(len(candleList - 1))]
    return normalizedList


normalizedCandles = maxNormalize(historicalCandles, 130000)

print("Normalized Candles...")


def dataPrep(candleData, input_length):
    sequences = []
    targets = []
    for i in range(len(candleData) - input_length):
        currSequence = candleData[i:i + input_length]
        sequences.append(currSequence)

        target = candleData[i + input_length][3]
        targets.append(target)
    print("Data Prepped...")
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        prev_hidden = hn[-1]
        prediction = self.fc(prev_hidden)
        return prediction


class LSTMDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]


# to preserve patterns for the AI to recognize, use batches to keep sequences together while still shuffling
def createSequenceBatches(sequences, targets, batch_size):
    batches = len(sequences) // batch_size
    for i in range(batches):
        start = i * batch_size
        end = start + batch_size
        yield sequences[start:end], targets[start:end]


model = LSTMModel(input_size=6, hidden_size=64, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# mess with epochs (iterations through data) 
# mess with batch_size (length segments of sequences)(better pattern recognition)
epochs = 10
batch_size = 32

# mess with input_length in this step:
sequences, targets = dataPrep(historicalCandles, 50)

print("Training model...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for sequenceBatch, targetBatch in createSequenceBatches(sequences, targets, batch_size):
        sequenceBatch, targetBatch = sequenceBatch.clone(), targetBatch.clone()
        optimizer.zero_grad()
        outputs = model(sequenceBatch)
        loss = criterion(outputs.squeeze(), targetBatch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f'Epoch {epoch} Loss {epoch_loss / len(sequences)}')