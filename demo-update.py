# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#% matplotlib inline
import matplotlib.pyplot as plt

plt.style.use("seaborn-white")

import random
import torch
from torch import nn
import torch.nn.functional as F
# from torch.nn import functional as F
from torch import optim
from tqdm import tqdm

from data import PermutedMNIST
from utils import EWC, ewc_train, normal_train, test

# %%
class MLP(nn.Module):
    def __init__(self, hidden_size=200):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 10)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# %%
# https://pytorch.org/docs/stable/notes/cuda.html

# %%
epochs = 50 # 3 # 50
lr = 0.001 # 1e-3
batch_size = 128 # 100 # 128
sample_size = 200
hidden_size = 200
num_task = 3

torch.manual_seed(0)

# TODO: pointer to constructor.
classifier = MLP # ConvNet


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

if cuda:
    print('cuda IS available')
else:
    print('cuda / GPU not available.')
    
# torch.cuda.device(1)


# %%
def get_permute_mnist():
    train_loader = {}
    test_loader = {}
    idx = list(range(28 * 28))
    for i in range(num_task):
        train_loader[i] = torch.utils.data.DataLoader(PermutedMNIST(train=True, permute_idx=idx),
                                                      batch_size=batch_size) #,
                                                      # num_workers=4)
        test_loader[i] = torch.utils.data.DataLoader(PermutedMNIST(train=False, permute_idx=idx),
                                                     batch_size=batch_size)
        random.shuffle(idx)
    return train_loader, test_loader


train_loader, test_loader = get_permute_mnist()


# %%
def standard_process(epochs, use_cuda=True, weight=True):
    model = classifier() # TODO: fix parms MLP(hidden_size)
    if torch.cuda.is_available() and use_cuda:
        model.cuda()

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
        
    # model.to(device)
    
    optimizer = optim.SGD(params=model.parameters(), lr=lr)

    loss, acc = {}, {}
    for task in range(num_task):
        loss[task] = []
        acc[task] = []
        for _ in tqdm(range(epochs)):
            loss[task].append(normal_train(model, optimizer, train_loader[task]))
            for sub_task in range(task + 1):
                acc[sub_task].append(test(model, test_loader[sub_task]))
        if task == 0 and weight:
            weight = model.state_dict()
    return loss, acc, weight


def ewc_process(epochs, importance, use_cuda=True, weight=None):
    model = classifier() # TODO: fix parms MLP(hidden_size)
    if torch.cuda.is_available() and use_cuda:
        model.cuda()
        
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
        
    # model.to(device)
        
    optimizer = optim.SGD(params=model.parameters(), lr=lr)

    loss, acc, ewc = {}, {}, {}
    for task in range(num_task):
        loss[task] = []
        acc[task] = []

        if task == 0:
            if weight:
                model.load_state_dict(weight)
            else:
                for _ in tqdm(range(epochs)):
                    loss[task].append(normal_train(model, optimizer, train_loader[task]))
                    acc[task].append(test(model, test_loader[task]))
        else:
            old_tasks = []
            for sub_task in range(task):
                old_tasks = old_tasks + train_loader[sub_task].dataset.get_sample(sample_size)
            old_tasks = random.sample(old_tasks, k=sample_size)
            for _ in tqdm(range(epochs)):
                loss[task].append(ewc_train(model, optimizer, train_loader[task], EWC(model, old_tasks), importance))
                for sub_task in range(task + 1):
                    acc[sub_task].append(test(model, test_loader[sub_task]))

    return loss, acc


# %%
def loss_plot(x):
    plt.figure()
    for t, v in x.items():
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)

def accuracy_plot(x):
    plt.figure()
    for t, v in x.items():
        plt.plot(list(range(t * epochs, num_task * epochs)), v)
    plt.ylim(0, 1)


# %%
loss, acc, weight = standard_process(epochs)


# %%

from pathlib import Path
Path('./output').mkdir(parents=True, exist_ok=True)

loss_plot(loss)
plt.savefig('./output/loss_plot_normal.png')


# %%
accuracy_plot(acc)
plt.savefig('./output/accuracy_normal.png')



# %%
# TODO: why weight commented out.
loss_ewc, acc_ewc = ewc_process(epochs, importance=1000) # , weight=weight)


# %%
loss_plot(loss_ewc)
plt.savefig('./output/loss_plot_ewc.png')

# %%
accuracy_plot(acc_ewc)
plt.savefig('./output/accuracy_ewc.png')

# %%
plt.plot(acc[0], label="sgd")
plt.plot(acc_ewc[0], label="ewc")
plt.legend()

plt.savefig('./output/sgd-vs-ewc.png')


