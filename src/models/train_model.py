from datetime import datetime
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm.notebook import tqdm_notebook
from dataset_fetcher import Dataset_fetcher
from model_architecture import XrayClassifier

class neural_net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(512*512, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 3)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim = 1)

        return x



class Training_loop():
    def __init__(self, path_img = 'D:\Technical University of Denmark\Machine Learning Operations\MLOps-Course-Project\data\preprocessed\covid_not_norm\\train_images.pt',\
        path_lab = 'D:\Technical University of Denmark\Machine Learning Operations\MLOps-Course-Project\data\preprocessed\covid_not_norm\\train_labels.pt'):
        
        #self.model = XrayClassifier(3)
        self.model = neural_net()        
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.003)
        self.criterion = nn.NLLLoss()
        self.DS = Dataset_fetcher(path_img, path_lab)
        self.loader = torch.utils.data.DataLoader(self.DS, shuffle=False, num_workers=0, batch_size=3)
        self.epochs = 10

    def loop(self):

        for e in tqdm_notebook(range(self.epochs), desc = f"Epochs"):
            running_loss = 0
            for images, labels in tqdm_notebook(self.loader, desc = f"Batch number: {e + 1}"):
                # false if image is not readable
                if images is not False:
                    self.model.train()
                    self.optimizer.zero_grad()
                    output = self.model(images)
                    self.loss = self.criterion(output, labels)
                    self.loss.backward()
                    self.optimizer.step()
                    running_loss += self.loss.item()
            else:
                if images is not False:
                    with torch.no_grad():
                        self.model.eval()
                        top_p, top_class = torch.exp(self.model(images)).topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        self.accuracy = torch.mean(equals.type(torch.FloatTensor))
                    print(f"Accuracy: {self.accuracy.item()*100}%")