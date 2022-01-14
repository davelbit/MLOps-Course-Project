import torch
import torch.nn.functional as F
from dataset_fetcher import Dataset_fetcher
from model_architecture import XrayClassifier
from torch import nn, optim
from tqdm.notebook import tqdm_notebook


class Training_loop:
    def __init__(self, path="data/raw/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset"):

<<<<<<< HEAD
=======
<<<<<<< HEAD
class Training_loop():
    def __init__(self, path = 'data/raw/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset'):
        
>>>>>>> 92ffdc16b57eb91eb776a67aae73c7cf51f81d33
        self.path = path
        self.model = XrayClassifier(3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        self.criterion = nn.NLLLoss()
        self.DS = Dataset_fetcher(self.path)
        self.loader = torch.utils.data.DataLoader(
            self.DS, shuffle=False, num_workers=0, batch_size=3
        )
        self.epochs = 10
=======

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512 * 512, 3)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
>>>>>>> Stefan

    def loop(self):

<<<<<<< HEAD
        for e in tqdm_notebook(range(self.epochs), desc="Epochs"):
=======
<<<<<<< HEAD
        for e in tqdm_notebook(range(self.epochs), desc = f"Epochs"):
>>>>>>> 92ffdc16b57eb91eb776a67aae73c7cf51f81d33
            running_loss = 0
            for images, labels in tqdm_notebook(self.loader, desc=f"Batch number: {e + 1}"):
                # false if image is not readable
                if images is not False:
                    self.model.train()
                    self.optimizer.zero_grad()
                    output = self.model(images)
                    self.loss = self.criterion(output, labels)
                    self.loss.backward()
                    self.optimizer.step()
                    running_loss += self.loss.item()
<<<<<<< HEAD

            if images is not False:
                with torch.no_grad():
                    self.model.eval()
                    top_p, top_class = torch.exp(self.model(images)).topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    self.accuracy = torch.mean(equals.type(torch.FloatTensor))
                print(f"Accuracy: {self.accuracy.item()*100}%")
=======
            else:
                if images is not False:
                    with torch.no_grad():
                        self.model.eval()
                        top_p, top_class = torch.exp(self.model(images)).topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        self.accuracy = torch.mean(equals.type(torch.FloatTensor))
                    print(f"Accuracy: {self.accuracy.item()*100}%")
=======
        return x


model = Classifier()
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.NLLLoss()
DS = Dataset_fetcher("data/raw/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset")
loader = torch.utils.data.DataLoader(DS, shuffle=False, num_workers=0, batch_size=3)

epochs = 10

for e in tqdm_notebook(range(epochs), desc=f"Epochs"):
    running_loss = 0
    for images, labels in tqdm_notebook(loader, desc=f"Batch number: {e + 1}"):
        # false if image is not readable
        if images is not False:
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    else:
        if images is not False:
            with torch.no_grad():
                model.eval()
                top_p, top_class = torch.exp(model(images)).topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f"Accuracy: {accuracy.item()*100}%")
>>>>>>> Stefan
>>>>>>> 92ffdc16b57eb91eb776a67aae73c7cf51f81d33
