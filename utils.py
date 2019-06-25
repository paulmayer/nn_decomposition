import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt


def get_synthetic_data(num_obs, seed=1):
    np.random.seed(seed)
    def get_label(x):
        if np.linalg.norm(x, 2) <= 0.6:
            return 1
        else:
            return 1 if np.random.random() < 0.15 else 0
        
    x1, x2 = np.meshgrid(np.linspace(-1, 1, int(np.sqrt(num_obs))), np.linspace(-1, 1, int(np.sqrt(num_obs))))
    xs = np.vstack([x1.ravel(), x2.ravel()]).transpose()
    ys = np.apply_along_axis(get_label, 1, xs).reshape(num_obs, 1)
    return xs, ys


class SyntheticData(Dataset):
    
    def __init__(self, num_obs, seed=0):
        if int(np.sqrt(num_obs))**2 != num_obs:
            raise ValueError("Cannot create grid for %d observations." % num_obs)
        
        xs, ys = get_synthetic_data(num_obs, seed)
        self.data = torch.from_numpy(xs).float()
        self.target = torch.from_numpy(ys).float()

    def __getitem__(self, index):
        return self.data[index], self.target[index]
        
    def __len__(self):
        return len(self.data)
        
def plot_decision_boundary(n, model, train_data):
    xs = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(xs, xs)
    grid = np.stack((xx, yy))
    grid = grid.T.reshape(-1,2)
    
    x_grid = torch.from_numpy(grid).float()
    outputs = torch.nn.Sigmoid()(model(x_grid))
    predictions = (outputs > 0.5).numpy()
    y1 = predictions.T[0].reshape(xs.shape[0],xs.shape[0])
    
    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.contourf(yy, xx, y1, levels=[-1, 0, 1, 2], colors=["white", "black"], alpha=0.3)
    
    xs_train = train_data.data.numpy()
    ys_train = train_data.target.int().numpy().reshape(xs_train.shape[0])
    
    ax.scatter(xs_train[ys_train==1, 0], xs_train[ys_train==1, 1],  color="red", s=1.5)
    ax.scatter(xs_train[ys_train==0, 0], xs_train[ys_train==0, 1],  color="blue", s=1.5)
    ax.set_aspect(1)
    
def get_accuracy(probabilities, labels):
    predictions = (probabilities > 0.5).int()
    correct = (predictions  == labels.int()).sum()
    return float(correct)/len(labels)