import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset

class GestureDataset(Dataset):
    def __init__(self,train=True):
        if train:
            df = pd.read_csv('./traindata.csv')
        else:
            df=pd.read_csv('./testdata.csv')
        self.labels = df['class'].values
        self.data = df.drop(columns=['class']).values
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.data = torch.tensor(self.data, dtype=torch.float32)


    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
batch_size = 4
train_dataset = GestureDataset(train=True)
test_dataset=GestureDataset(train=False)
train_dataloader = DataLoader(train_dataset, shuffle=True,batch_size=batch_size)
test_dataloader = DataLoader(train_dataset, batch_size=batch_size)
def train(dataloader, model, loss_fn, optimizer,scheduler):
    size = len(dataloader.dataset)
    model.train()
    total_correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        _, predicted = torch.max(pred, 1)
        total_correct += (predicted == y).sum().item()
        if batch % 100== 0:
            loss_value = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
    accuracy = total_correct / size
    scheduler.step()
    print(f"Accuracy: {accuracy * 100:.2f}%")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
epochs = 5
from ANN import ANN
model=ANN()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ExponentialLR(optimizer, gamma=0.9)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer,scheduler)
    test(test_dataloader, model, loss_fn)
print("Done!")
torch.save(model.state_dict(), "model.pth")