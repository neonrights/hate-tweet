import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils_data

from transformer import Transformer
from dataloader import ELMoDataset


class SemevalModel(nn.Module):
    def __init__(self, layers, heads, embed_dim):
        super().__init__()
        self.transformer = Transformer(layers, heads, embed_dim, dropout=0.5)
        self.fc = nn.Linear(embed_dim, 3)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    en_train = utils_data.DataLoader(ELMoDataset("filepath"), batch_size=10, shuffle=True)
    en_test = None

    model = SemevalModel(6, 1, None)
    gpu = torch.cuda.device(0)
    #model.cuda()

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(100):
        print("epoch {}".format(epoch+1))
        for i, batch in enumerate(en_train):
            optimizer.zero_grad()
            logits = model(batch[0])
            loss = loss_function(logits, batch[1])
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("train_loss: {}".format(loss.item())

    # calculate test loss


