# from tradingview_ta import TA_Handler, Interval, Exchange
import time
from datetime import datetime
import random
import ccxt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


exchanges = ccxt.exchanges
# print(exchanges)

exchange = ccxt.coinbaseadvanced()

# code to gather candlestick data and store in a file (candles.npy)

symbol = 'BTC/USD'
timeframe = '5m'
limit = 1000

# date to stop gathering data
stopDate = int(datetime(2024, 12, 25).timestamp() * 1000)

#date to start gathering data
currDate = int(datetime(2024, 10, 1).timestamp() * 1000)

# candlestick data is put into candles.npy to be used as training data
'''
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

#validation data
'''
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

np.save('val_candles.npy', candleData)
'''

# Tohlcv (structure of each candle in list form)

#device for operations is picked
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# candle dataset and validation set are loaded
npCandles = np.load('candles.npy')
historicalCandles = np.load('candles.npy').tolist()
validationCandles = np.load('val_candles.npy').tolist()
print(len(historicalCandles))
print(len(validationCandles))

# matplot of candle history data (closing prices)
'''
plotPrices = npCandles[:, 4]
plt.plot(plotPrices)
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.title('Historical Candles Dataset')
plt.show()
'''
# for normalizing try logarithmic or sections (couple days probs)
minPercent = 0.0
maxPercent = 0.0

historicalCandles[0][0] = 0
validationCandles[0][0] = 0

# this is what the model will predict, it is the index in the candle (tohlcv and price change val)
predictIndex = 6

# getting percentage changes in closing prices
for i in range(len(historicalCandles)):
    if i != 0:
        currPercent = (historicalCandles[i][4] - historicalCandles[i-1][4])/(historicalCandles[i-1][4])
        historicalCandles[i][0] = currPercent
        if currPercent and currPercent < minPercent:
            minPercent = historicalCandles[i][0]
        elif currPercent > maxPercent:
            maxPercent = historicalCandles[i][0]

# inputting percent changes (into index 0) for validation candles aswell
for i in range(len(validationCandles)):
    if i != 0:
        currPercent = (validationCandles[i][4] - validationCandles[i-1][4])/(validationCandles[i-1][4])
        validationCandles[i][0] = currPercent
        if currPercent and currPercent < minPercent:
            minPercent = validationCandles[i][0]
        elif currPercent > maxPercent:
            maxPercent = validationCandles[i][0]

# adding value for if the price increased or decreased over the last candle, this is what the model will predict

for i in range(len(historicalCandles)):
    if i > 0 and historicalCandles[i][4] > historicalCandles[i-1][4]:
        historicalCandles[i].append(1)
    else:
        historicalCandles[i].append(0)
print("Candle price change data added")

# same thing done for the validation candles
for i in range(len(validationCandles)):
    if i > 0 and validationCandles[i][4] > validationCandles[i-1][4]:
        validationCandles[i].append(1)
    else:
        validationCandles[i].append(0)
print("Validation candle price change data added")

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

# this method is currently in use to normalize the candle data
def maxNormalize(candleList, max):
    candleList = np.array(candleList)
    normalizedList = np.copy(candleList)
    min = 0
    # change this if bitcoin goes to crazy heights

    for i in range(len(candleList)):

        # section = candleList[i]
        for j in range(6):
            if j == 0:
                normalizedList[i, j] = (normalizedList[i, j] - minPercent)/(maxPercent - minPercent)
            else:
                normalizedList[i, j] = (candleList[i, j] - min) / (max - min)

    normalizedList = normalizedList[0:(len(candleList - 1))]
    return normalizedList


normalizedCandles = maxNormalize(historicalCandles, 130000)
normValCandles = maxNormalize(validationCandles, 130000)

normCandle = normalizedCandles[163400]
rawCandle = historicalCandles[163400]
print("norm candle: ", normCandle, "raw candle: ", rawCandle)

print("Normalized candles...")
def dataPrep(candleData, input_length):
    sequences = []
    targets = []
    for i in range(len(candleData) - input_length):
        currSequence = candleData[i:i + input_length]
        sequences.append(currSequence)
        # index 4 is closing price
        target = candleData[i + input_length][predictIndex]
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

        # a few debug prints; first are for seeing if weights and bias are changing, second is for hidden layers
        # print("FC weights:", self.fc.weight)
        # print("FC bias:", self.fc.bias)

        # print("hn shape:", hn.shape)
        # print("hn[-1]:", hn[-1])

        prev_hidden = hn[-1]
        # prediction = self.fc(prev_hidden)
        prediction = torch.sigmoid(self.fc(prev_hidden))
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
def createSequenceBatches(sequences, targets, batch_size, shuffle):
    total_batches = len(sequences) // batch_size
    indices = list(range(total_batches))

    if shuffle:
        random.shuffle(indices)  # shuffle the batch order

    for i in indices:
        start = i * batch_size
        end = start + batch_size
        yield sequences[start:end], targets[start:end]

        # input_batch = sequences[start:end, :, :6]
        # target_batch = targets[start:end]

        # yield torch.tensor(input_batch, dtype=torch.float32), torch.tensor(target_batch, dtype=torch.float32)


print(f'Using device: {device}')

model = LSTMModel(input_size=7, hidden_size=128, output_size=1).to(device)
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mess with epochs (iterations through data)
# mess with batch_size (length segments of sequences)(better pattern recognition)
epochs = 10
batchSize = 4

# data is finished preparing for training here and saved for future use, uncomment and run to change the saved data
'''
# mess with input_length in this step:
sequences, targets = dataPrep(normalizedCandles, 50)
np.save('sequences.npy', sequences)
np.save('targets.npy', targets)

valSequences, valTargets = dataPrep(normValCandles, 50)
np.save('valSequences.npy', valSequences)
np.save('valTargets.npy', valTargets)
'''


# defining sequences and their targets for the model to predict
sequences = np.load('sequences.npy')
targets = np.load('targets.npy')

valSequences = np.load('valSequences.npy')
valTargets = np.load('valTargets.npy')


'''
# data is shuffled to help with training
print("Shuffling data...")
# targets and sequences in pairs:
pairedList = list(zip(sequences, targets))
print("it made the pair list")
# targets and sequences are shuffled together, and put into shuffledSequences and shuffledTargets
random.shuffle(pairedList)
print("it shuffled the pairs")
shuffledSequences, shuffledTargets = zip(*pairedList)
print("it separated the shuffled pairs into separate lists")
# convert shuffledSequences and targets back into tensors
print(type(shuffledSequences), type(shuffledTargets))
shuffledSequences = torch.tensor(np.array(shuffledSequences))
shuffledTargets = torch.tensor(np.array(shuffledTargets))
print("it converted them into tensors")
print(type(shuffledSequences), type(shuffledTargets))
print(shuffledSequences.shape, shuffledTargets.shape)
'''

targets = torch.tensor(targets)
sequences = torch.tensor(sequences)
valSequences = torch.tensor(valSequences)
valTargets = torch.tensor(valTargets)

print("Training model...")
doPrint = True
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for sequenceBatch, targetBatch in createSequenceBatches(sequences, targets, batchSize, shuffle=True):
        sequenceBatch, targetBatch = sequenceBatch.clone().to(device), targetBatch.clone().to(device)
        optimizer.zero_grad()
        outputs = model(sequenceBatch)

        #print(f'training predicted changes: {outputs.squeeze(-1)}')
        #print(f'training target changes: {targetBatch}')

        loss = criterion(outputs.squeeze(), targetBatch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    model.eval()

    val_epoch_loss = 0.0
    totalCorrect = 0
    totalPredictions = 0

    with torch.no_grad():
        for valSequenceBatch, valTargetBatch in createSequenceBatches(valSequences, valTargets, batchSize, shuffle=False):
            valSequenceBatch, valTargetBatch = valSequenceBatch.clone().to(device), valTargetBatch.clone().to(device)

            valOutputs = model(valSequenceBatch)
            # valOutputs and valTargetBatch have same length :)

            # predictedChanges = torch.sign(valOutputs).squeeze(-1)
            predictedChanges = (valOutputs > 0.5).float().squeeze(-1)

            targetChanges = torch.sign(valTargetBatch)

            #print(f'validation predicted changes: {valOutputs.squeeze(-1)}')
            #print(f'target changes: {valTargetBatch}')

            correctPredictions = (predictedChanges == targetChanges).sum().item()

            totalCorrect += correctPredictions
            totalPredictions += len(valTargetBatch)
            # print(totalCorrect, totalPredictions)

            valLoss = criterion(valOutputs.squeeze(), valTargetBatch)

            val_epoch_loss += valLoss.item()

    percentAccuracy = (totalCorrect / totalPredictions) * 100

    print(f'Epoch {epoch + 1} percent accuracy in direction: {percentAccuracy:.2f}%')

    print(f'Epoch {epoch+1} Loss {epoch_loss / len(sequences)} Validation Loss {val_epoch_loss / len(sequences)}')


newDate = int(datetime(2024, 12, 20).timestamp() * 1000)
newCandles = exchange.fetch_ohlcv(symbol, timeframe, limit=50)
newNormCandles = torch.tensor(maxNormalize(newCandles, 130000), dtype=torch.float32).unsqueeze(0).to(device)

# predicting with real time data!
'''
print(f'Last Candle Date: {datetime.utcfromtimestamp(newCandles[-1][0]/1000)} Last Candle Price: {newCandles[-1][4]}')

model.eval()

with torch.no_grad():
    prediction = model(newNormCandles)

prediction = prediction.squeeze().cpu()
if predictIndex == 4:
    print("prediction:", prediction*130000)
elif predictIndex == 0:
    print("prediction:", (prediction*(maxPercent - minPercent)) + minPercent)
    print("predicted closing price:", (1 + ((prediction*(maxPercent - minPercent)) + minPercent))*newCandles[-1][4])
'''