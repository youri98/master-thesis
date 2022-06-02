import torch
print(torch.nn.BCELoss()(torch.zeros((3)), torch.ones((3))))