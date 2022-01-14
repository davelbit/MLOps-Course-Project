import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm.notebook import tqdm_notebook
from dataset_fetcher import Dataset_fetcher

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512*512, 3)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.log_softmax(self.fc1(x))

        return x

    
model = Classifier()
optimizer = optim.Adam(model.parameters(), lr = 0.003)
criterion = nn.NLLLoss()
DS = Dataset_fetcher('data/raw/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset')
loader = torch.utils.data.DataLoader(DS, shuffle=False, num_workers=0, batch_size=3)

epochs = 10

for e in tqdm_notebook(range(epochs), desc = f"Epochs"):
    running_loss = 0
    for images, labels in tqdm_notebook(loader, desc = f"Batch number: {e + 1}"):
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
                top_p, top_class = torch.exp(model(images)).topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f"Accuracy: {accuracy.item()*100}%")