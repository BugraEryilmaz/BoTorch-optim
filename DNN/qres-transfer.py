# %%
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Module, ModuleList, Linear, ReLU, MSELoss, L1Loss
from torch.nn.init import xavier_uniform_, calculate_gain
import torch.nn.functional as F
from torch.optim import Adam, SGD
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='Parse the DNNBO arguments')

parser.add_argument('--dataset_loc1',dest='dataset_loc1',nargs='*',help='''The location of the datasets to be used. If multiple given it will try all possiblities. The dataset location is "datasets_transfer/dataset_{dataset_loc1}.csv". Default: "uniform"''',default=['uniform'])         
parser.add_argument('--model_loc',dest='model_loc',nargs='?',help='''The location of the model to be used. Default: ""model_qres_0_[100, 50, 20, 10]_uniform.pt""''',default='model_qres_0_[100, 50, 20, 10]_uniform.pt')         
parser.add_argument('--model_layers',dest='model_layers',nargs='*',help='''Layer configuration of the model. Ex: 100 4. Default: 100 4''',default=[100, 4], type=int)         
parser.add_argument('--init',dest='init',nargs='*',help='''True means it will reinitialize last layer. If multiple given it will try all possiblities. Ex. --init True False. Default False''',default=[False], type=lambda x: bool(strtobool(x)))         
parser.add_argument('--train_last',dest='train_last',nargs='*',help='''True means it will only train the last layer. If multiple given it will try all possiblities. Ex: --train_last True False. Default: False''',default=[False], type=lambda x: bool(strtobool(x)))         

args = parser.parse_args()

dataset_loc1 = args.dataset_loc1

model_loc = args.model_loc

layer_conf = args.model_layers
# %%
# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, df, data_fields, result_fields):
        # store the inputs and outputs
        df_data = df[data_fields]
        df_res = df[result_fields]

        data = [c.values for n,c in df_data.items()]
        res = [c.values for n,c in df_res.items()]

        X = np.stack(data, 1)
        y = np.stack(res, 1)

        if normalize:
            X = StandardScaler().fit_transform(X)
            y = StandardScaler().fit_transform(y)

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

# %%
# model definition
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


# QRes Layer: inspired by https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class QResLayer(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(QResLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_2 = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        #nn.init.kaiming_uniform_(self.weight_1, a=math.sqrt(5))
        #nn.init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.weight_1)
        nn.init.xavier_uniform_(self.weight_2)
        

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        h_1 = F.linear(input, self.weight_1, bias=None)
        h_2 = F.linear(input, self.weight_2, bias=None)
        return torch.add(
            torch.mul(h_1, h_2), 
            F.linear(input, self.weight_1, self.bias)
        )

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    
torch.nn.QResLayer = QResLayer    
        
# Multi-layer Perceptron
class QRes(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        depth,
        act=torch.nn.Tanh
    ):
        super(QRes, self).__init__()
        
        layers = [('input', torch.nn.QResLayer(input_size, hidden_size))]
        layers.append(('input_activation', act()))
        for i in range(depth): 
            layers.append(
                ('hidden_%d' % i, torch.nn.QResLayer(hidden_size, hidden_size))
            )
            layers.append(('activation_%d' % i, act()))
        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))

        layerDict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out
    

model = QRes(8, layer_conf[0], 3, layer_conf[1])

for loc1 in dataset_loc1:
    for train_last in args.train_last:
        for reinit in args.init:
            n_EPOCHS = 3000
            normalize = False
            lr = 0.0005
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            dataset_loc = "datasets_transfer/dataset_" + loc1 + ".csv"
            data_fields = ["XLE1", "XLE2", "CHORD1_1", "CHORD1_2", "CHORD2_1", "CHORD2_2", "SSPAN1_2", "SSPAN2_2"]
            result_fields = ["CL_CD","CD","XCP"]
            train_batchsize = 37
            test_batchsize = 13

            model.load_state_dict(torch.load(model_loc))
            model = model.to(device)
            # Reinitialize last layer
            if reinit:
                xavier_uniform_(model.out_layer.weight)
            # %%
            df = pd.read_csv(dataset_loc,)
            dataset = CSVDataset(df, data_fields, result_fields)
            train_num = 37
            test_num = 13
            train, test = random_split(dataset, [train_num, test_num])

            train_dl = DataLoader(train, batch_size=train_batchsize, shuffle=True)
            test_dl = DataLoader(test, batch_size=test_batchsize, shuffle=False)

            lossfn = MSELoss()
            #lossfn = L1Loss()
            if train_last:
                optimizer = Adam(model.out_layer.parameters(), lr=lr)
            else:
                optimizer = Adam(model.parameters(), lr=lr)

            train_losses = []
            train_losses_mae = []
            valid_losses = []
            valid_losses_mae = []
            best_loss = np.inf
            for epoch in range(n_EPOCHS):
                model.eval()
                cur_loss = 0
                cur_loss_mae = 0
                num_batch = 0
                with torch.no_grad():
                    for i, (inputs, targets) in enumerate(test_dl):
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        # compute the model output
                        yp = model(inputs)
                        # calculate loss
                        loss = lossfn(yp, targets)
                        mae = F.l1_loss(yp, targets)
                        cur_loss += loss
                        cur_loss_mae += mae
                        num_batch += 1
                valid_losses.append(cur_loss.cpu().detach().numpy()/num_batch)
                valid_losses_mae.append(cur_loss_mae.cpu().detach().numpy()/num_batch)

                if cur_loss < best_loss:
                    #save model
                    torch.save(model.state_dict(), f"model_qres_{'without_reset'}.pt")
                    best_loss = cur_loss
                
                model.train()
                train_loss = 0
                train_loss_mae = 0
                train_num_batch = 0
                for i, (inputs, targets) in enumerate(train_dl):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    # clear the gradients
                    optimizer.zero_grad()
                    # compute the model output
                    yp = model(inputs)
                    # calculate loss
                    loss = lossfn(yp, targets)
                    mae = F.l1_loss(yp, targets)
                    # credit assignment
                    loss.backward()
                    # update model weights
                    optimizer.step()
                    train_loss += loss
                    train_loss_mae += mae
                    train_num_batch += 1
                train_losses.append(train_loss.cpu().detach().numpy()/train_num_batch)
                train_losses_mae.append(train_loss_mae.cpu().detach().numpy()/train_num_batch)

                print(cur_loss)
            np.savetxt(f"train_loss_qres_{'without_reset' if not reinit else 'with_reset'}_{'only_last' if train_last else 'all_trained'}_{loc1}.csv", np.asarray(train_losses), delimiter=',')
            np.savetxt(f"train_loss_qres_{'without_reset' if not reinit else 'with_reset'}_{'only_last' if train_last else 'all_trained'}_{loc1}.mae.csv", np.asarray(train_losses_mae), delimiter=',')
            np.savetxt(f"valid_loss_qres_{'without_reset' if not reinit else 'with_reset'}_{'only_last' if train_last else 'all_trained'}_{loc1}.csv", np.asarray(valid_losses), delimiter=',')
            np.savetxt(f"valid_loss_qres_{'without_reset' if not reinit else 'with_reset'}_{'only_last' if train_last else 'all_trained'}_{loc1}.mae.csv", np.asarray(valid_losses_mae), delimiter=',')
