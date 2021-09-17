import sys
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

debug = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(13, 50, True)
        self.hidden2 = nn.Linear(50, 25, True)
        self.hidden3 = nn.Linear(25, 15, True)
        self.output = nn.Linear(15, 1, True)
        self.to(get_device())

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = torch.sigmoid(self.output(x))
        return x


class OptionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        if self.transform:
            row = self.transform(row)
        return row


def get_device():
    if torch.cuda.is_available() and not debug:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU
    return device

def init_weights(m):
    print(m)

def print_hist(y_pred, y, th, norms):
    size = len(y)
    pbins = [0]*11
    tpbins = [0]*11
    #print(y_pred)
    #print(y)
    with torch.no_grad():
        y_th = torch.heaviside(y_pred - torch.tensor(th).expand_as(y_pred), torch.tensor(1.0).expand_as(y_pred))
        tp = torch.logical_and(y_th, y)
        #print(tp)
        for i in range(size):
            if y_th[i]:
                n = int(norms[i]*10 + 0.5) + 4
                if n < 0:
                    n = 0
                elif n > 10:
                    n = 10
                pbins[n] += 1
                if tp[i]:
                    tpbins[n] += 1

    for i in range(11):
        if pbins[i] == 0:
            continue
        n = float(i - 4)/10
        p = float(tpbins[i]*100)/pbins[i]
        print(f"{n},{p},{pbins[i]}")

def calc_groi(y_pred, th, roi):
    size = len(y)
    groi = 0.0
    with torch.no_grad():
        y_th = torch.heaviside(y_pred - torch.tensor(th).expand_as(y_pred), torch.tensor(1.0).expand_as(y_pred))
        #print(y_th)
        #print(roi)
        for i in range(size):
            if y_th[i]:
                groi += roi[i]
    return groi

def calc_acc(y_pred, y, th):
    #tp = 0
    #tn = 0
    size = len(y)
    #for i in range(size):
    #    if y_pred[i] >= th and y[i] == 1:
    #        tp += 1
    #    elif y_pred[i] < th and y[i] == 0:
    #        tn += 1
    #return float(tp + tn)*100/size
    with torch.no_grad():
        #print(y)
        #print(y_pred)
        y_th = torch.heaviside(y_pred - torch.tensor(th).expand_as(y_pred), torch.tensor(1.0).expand_as(y_pred))
        tp = torch.sum(torch.logical_and(y_th, y))
        p_pred = torch.sum(y_th)
        p_label = torch.sum(y)
        #print(y_th)
        #print(torch.logical_xor(y_th, y))
        acc = float(size - torch.sum(torch.logical_xor(y_th, y)))*100/size
        prec = float(tp)*100/p_pred
        rec = float(tp)*100/p_label
    return [acc, prec, rec]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: python3 {sys.argv[0]} file [model]')
        sys.exit(-1)
    #print(torch.__version__)
    input_file = sys.argv[1]
    #size = int(sys.argv[2])
    #input_df = pd.read_csv(input_file, nrows=size).sample(frac=1).values
    #input_df = pd.read_csv(input_file).sample(frac=1).values
    input_df = np.array((list(csv.reader(open(input_file, "r"))))[1:], dtype=object)
    #size_train = int(min(size, len(input_df)) / 10 * 9)
    size_train = len(input_df)
    #print(size_train)
    device = get_device()
    ml_th = 0.5
    tp = 0
    tn = 0
    ps = 0
    min_groi = 0.0
    input_groi = 0.0
    min_norm = 10000.0
    max_norm = -10000.0
    
    for i in range(len(input_df)):
        for j in range(4,23):
            input_df[i][j] = float(input_df[i][j])
        #label
        real0 = (input_df[i][6] + input_df[i][9])/2
        label = 1 if input_df[i][17] >= real0 - 0.000000001 else 0
        #if label != int(input_df[i][23]):
        #    print(input_df[i][0] + ": " + str(input_df[i][17]) + " ? " + str(real0) + "=" + str(input_df[i][6]) + "+" + str(input_df[i][9]) + "/2 " + str(label))
        forecast = 1 if input_df[i][15] >= real0 - 0.000000001 else 0
        input_df[i][3] = forecast
        input_df[i][1] = input_df[i][17]/real0 - 1.0 #RoI
        input_groi += input_df[i][1]
        input_df[i][17] = label
        ps += forecast
        if forecast == 1:
            min_groi += input_df[i][1]

        if forecast == 1 and label == 1:
            tp += 1
        elif forecast == 0 and label == 0:
            tn += 1
        #normalization
        #print(input_df[i][0])
        strike = float(input_df[i][0].split(' ')[-2][1:])
        s_ask = input_df[i][10] - strike
        s_bid = input_df[i][11] - strike
        s = (input_df[i][10] + input_df[i][11])/2 
        #norm = (s - strike)/(s + strike)
        norm = (s - strike)/s
        if norm > max_norm:
            max_norm = norm
        if norm < min_norm:
            min_norm = norm
        input_df[i][2] = norm

        total = 0.0
        for j in range(4, 10):
            total += input_df[i][j]
        av = total/6.0
        sigma = 0.0
        for j in range(4, 10):
            sigma += (input_df[i][j] - av) ** 2
        sigma = math.sqrt(sigma/5.0)
        for j in range(4, 10):
            input_df[i][j] = (input_df[i][j] - av)/sigma
        input_df[i][10] = (s_ask - av)/sigma
        input_df[i][11] = (s_bid - av)/sigma
        input_df[i][15] = (input_df[i][15] - av)/sigma
        input_df[i][16] = (input_df[i][16] - av)/sigma
        #print(input_df[i])

    #print(torch.__version__)
    #X = torch.from_numpy(input_df[0:size_train, 2:-2]).float().to(device)
    #X = torch.from_numpy(input_df[0:size_train, 4:17]).float().to(device)
    #print(input_df[0:size_train, 4:17])
    X = torch.Tensor(input_df[0:size_train, 4:17].astype(float)).float().to(device)
    #print(X)
    y = torch.from_numpy(input_df[0:size_train, 17:18].astype(float)).float().to(device)
    #print(y.shape)
    #print(y)
    #print(torch.sum(y))
    #y.resize_((size_train))
    #X_test = torch.from_numpy(input_df[size_train:, 2:-2]).float().to(device)
    #y_test = torch.from_numpy(input_df[size_train:, -2:-1]).float().to(device)
    if len(sys.argv) > 2:
        nnn = torch.load(sys.argv[2])
        nnn.eval()
        train = False
    else:
        nnn = Net()
        train = True
    #nnn.apply(torch.nn.init.xavier_uniform_)
    #nnn.apply(init_weights)
    #print(nnn)
    optimizer = torch.optim.Adam(nnn.parameters(), lr=0.0001, weight_decay=0.0001)
    loss_func = nn.BCELoss()

    for t in range(10000):
        y_pred = nnn(X)
        #print(y_pred)
        loss = loss_func(y_pred, y)
        if not train:
            break

        if t % 1000 == 0:
            print(f'Epoch {int(t)}: Loss {loss}')
            [ml_acc, ml_prec, ml_rec] = calc_acc(y_pred, y, ml_th)
            print(f'ML accuracy {ml_acc}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #print(y_pred)
    print(f'Final loss: {loss}')
    print(f'Threshold {ml_th}')
    [ml_acc, ml_prec, ml_rec] = calc_acc(y_pred, y, ml_th)
    ml_groi = calc_groi(y_pred, ml_th, input_df[:, 1])
    print(f'Final ML accuracy {ml_acc}')
    print(f'Final ML precision {ml_prec}')
    print(f'Final ML recall {ml_rec}')
    print(f'Final ML gross RoI {ml_groi}')
    min_acc = float(tp+tn)*100/size_train
    min_prec = float(tp)*100/ps
    min_rec = float(tp)*100/torch.sum(y)
    print(f'Minimizer accuracy {min_acc}')
    print(f'Minimizer precision {min_prec}')
    print(f'Minimizer recall {min_rec}')
    print(f'Minimizer gross RoI {min_groi}')
    inp_pos = float(torch.sum(y))*100/size_train
    print(f'Input positivity rate {inp_pos}')
    print(f'Input gross RoI {input_groi}')
    print(f'Min norm {min_norm}')
    print(f'Max norm {max_norm}')
    print_hist(y_pred, y, ml_th, input_df[:, 2])
    #print_hist(torch.from_numpy(input_df[0:size_train, 3:4].astype(float)).float().to(device), y, ml_th, input_df[:, 2])
    if train:
        torch.save(nnn, "./nnn/model1")
    #print(y_pred)
    exit(0)
    result = nnn(X_test)
    test_loss = loss_func(result, y_test)

    print(f'Test loss is {test_loss};')
    print('For the final five test options,')
    print('Estimates and real prices are:')
    print('Estimation\t\t\t\tReal\t\t\t\tMinimizer')
    for i in range(1, 6):
        print(f'{result[-i].cpu().detach().numpy()}\t\t\t{y_test[-i].cpu().detach().numpy()}'
              f'\t\t\t{X_test[-i].cpu().detach().numpy()[-2:]}')
