# coding = utf-8

import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
import copy
import itertools
from torch.cuda.amp import GradScaler, autocast

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Residual_Units(nn.Module):
    def __init__(self, dim_stack, hidden_units):
        """Residual Units.
        Args:
            :param hidden_unit: A list. Neural network hidden units.
            :param dim_stack: A scalar. The dimension of inputs unit.
        :return:
        """
        super(Residual_Units, self).__init__()
        self.layer1 = nn.Linear(dim_stack, hidden_units)
        self.layer2 = nn.Linear(hidden_units, dim_stack)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = inputs
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        outputs = self.relu(x + inputs)
        return outputs


class AccSurrogateNet_GRU_Residual(nn.Module):
    def __init__(self, da_dim, op_dim, embed_size, DaAttention_factor, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4,
                 out_dim, BiDirection, global_max_oplen):
        super(AccSurrogateNet_GRU_Residual, self).__init__()

        self.global_max_oplen = global_max_oplen
        self.embed_size = embed_size
        "1. Dataset Embedding"
        self.reduction_layer_da = nn.Sequential(nn.Linear(da_dim, embed_size), nn.BatchNorm1d(embed_size))
        self.DaAttention_factor = DaAttention_factor

        "2-1. Evaluation Pipeline Embedding"
        for i in range(self.global_max_oplen):
            setattr(self, 'reduction_layer_op' + str(i + 1),
                    nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size)))
        if BiDirection:
            self.gru = nn.GRU(input_size=embed_size, hidden_size=int(embed_size / 2), num_layers=2, bias=True,
                              batch_first=True, dropout=0, bidirectional=True)
        else:
            self.gru = nn.GRU(input_size=embed_size, hidden_size=embed_size, num_layers=1, bias=True,
                              batch_first=True, dropout=0, bidirectional=False)

        for name, para in self.gru.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(para)
            elif name.startswith("bias"):
                nn.init.constant_(para, 0)
        "2-2. Evaluation Pipeline & Dataset Attention"
        self.attention_W = nn.Parameter(torch.Tensor(embed_size, self.DaAttention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(self.DaAttention_factor))
        self.projection_h = nn.Parameter(torch.Tensor(self.DaAttention_factor, 1))
        for tensor in [self.attention_W, self.projection_h]:
            nn.init.xavier_normal_(tensor, )
        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor, )

        "3-1. Accompanying Pipeline Embedding"
        for i in range(self.global_max_oplen):
            setattr(self, 'reduction_layer_acc' + str(i + 1),
                    nn.Sequential(nn.Linear(op_dim, embed_size), nn.BatchNorm1d(embed_size)))
        "3-2. Accompanying Pipeline & Dataset Attention"
        self.attention_W_acc = nn.Parameter(torch.Tensor(embed_size, self.DaAttention_factor))
        self.attention_b_acc = nn.Parameter(torch.Tensor(self.DaAttention_factor))
        self.projection_h_acc = nn.Parameter(torch.Tensor(self.DaAttention_factor, 1))
        for tensor in [self.attention_W_acc, self.projection_h_acc]:
            nn.init.xavier_normal_(tensor, )
        for tensor in [self.attention_b_acc]:
            nn.init.zeros_(tensor, )

        "4. Multiple Residual & Output Layer"
        in_dim = embed_size * 4 + 1
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = Residual_Units(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3))
        self.layer4 = Residual_Units(n_hidden_3, n_hidden_4)
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim))

    def forward(self, input_da, input_ops, input_ops_mask, input_acc, input_acc_mask, input_acc_perform):

        "1. Dataset Embedding"
        da_embed = F.relu(self.reduction_layer_da(input_da))

        "2-1. Evaluation Pipeline Embedding"
        op_num = input_ops.shape[1]
        batch_size = input_ops.shape[0]
        op_embeds = []
        for i in range(op_num):
            op_embeds.append(F.relu(getattr(self, 'reduction_layer_op' + str(i + 1))(input_ops[:, i, :])))
        op_embeds = torch.stack(op_embeds, dim=1)
        masked_op_embeds = nn.utils.rnn.pack_padded_sequence(op_embeds, input_ops_mask, batch_first=True,
                                                             enforce_sorted=False)
        gru_out, _ = self.gru(masked_op_embeds)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        indices = input_ops_mask - 1
        gru_out_valid = gru_out[torch.arange(batch_size), indices, :]
        "2-2. Evaluation Pipeline & Dataset Attention"
        op_num = input_ops.shape[1]
        op_embeds = op_embeds.reshape(op_num, batch_size, self.embed_size)
        attention_input = op_embeds * da_embed
        attention_input = attention_input.reshape(batch_size, op_num, self.embed_size)
        attention_temp = F.relu(torch.tensordot(attention_input, self.attention_W, dims=([-1], [0])) + self.attention_b)
        self.normalized_att_score = F.softmax(torch.tensordot(attention_temp, self.projection_h, dims=([-1], [0])),
                                              dim=1)
        attention_output = torch.sum(self.normalized_att_score * attention_input, dim=1)

        "3-1. Accompanying Pipeline Embedding"
        op_num = input_acc.shape[1]
        batch_size = input_acc.shape[0]
        acc_embeds = []
        for i in range(op_num):
            acc_embeds.append(F.relu(getattr(self, 'reduction_layer_acc' + str(i + 1))(input_acc[:, i, :])))
        acc_embeds = torch.stack(acc_embeds, dim=1)
        masked_acc_embeds = nn.utils.rnn.pack_padded_sequence(acc_embeds, input_acc_mask, batch_first=True,
                                                              enforce_sorted=False)
        gru_acc_out, _ = self.gru(masked_acc_embeds)
        gru_acc_out, _ = nn.utils.rnn.pad_packed_sequence(gru_acc_out, batch_first=True)
        indices = input_acc_mask - 1
        gru_acc_out_valid = gru_acc_out[torch.arange(batch_size), indices, :]
        "3-2. Accompanying Pipeline & Dataset Attention"
        op_num = input_acc.shape[1]
        acc_embeds = acc_embeds.reshape(op_num, batch_size, self.embed_size)
        attention_input_acc = acc_embeds * da_embed
        attention_input_acc = attention_input_acc.reshape(batch_size, op_num, self.embed_size)
        attention_temp_acc = F.relu(torch.tensordot(attention_input_acc, self.attention_W_acc, dims=([-1], [0])) + self.attention_b_acc)
        self.normalized_att_score_acc = F.softmax(
            torch.tensordot(attention_temp_acc, self.projection_h_acc, dims=([-1], [0])), dim=1)
        attention_output_acc = torch.sum(self.normalized_att_score_acc * attention_input_acc, dim=1)

        "4. Multiple Residual & Output Layer"
        pred_input = torch.cat(
            (input_acc_perform, gru_acc_out_valid, attention_output_acc, attention_output, gru_out_valid),
            dim=1)
        pred_input = pred_input.float()
        hidden_1_out = F.relu(self.layer1(pred_input))
        hidden_2_out = F.relu(self.layer2(hidden_1_out))
        hidden_3_out = F.relu(self.layer3(hidden_2_out))
        hidden_4_out = F.relu(self.layer4(hidden_3_out))
        out = torch.sigmoid(self.layer5(hidden_4_out))
        return out


def evaluate(data_loader, net, criterion):
    net.eval()
    eval_loss = 0
    eval_r2 = 0
    eval_rmse = 0
    eval_mae = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            " Prepare data "
            eval_x_da, eval_x_ops, eval_x_ops_mask, eval_y, eval_x_acc, eval_x_acc_mask, eval_y_acc = data
            if torch.cuda.is_available():
                eval_x_da = eval_x_da.to(device)
                eval_x_ops = eval_x_ops.to(device)
                eval_y = eval_y.to(device)
                eval_x_acc = eval_x_acc.to(device)
                eval_y_acc = eval_y_acc.to(device)

            "Preprocess for possible nan&inf"
            nan_mask = torch.isnan(eval_x_da)
            inf_mask = torch.isinf(eval_x_da)
            eval_x_da[nan_mask] = 0
            max_val = torch.max(eval_x_da[~inf_mask])
            eval_x_da[inf_mask] = max_val

            "Forward"
            with autocast():
                out = net(eval_x_da, eval_x_ops, eval_x_ops_mask, eval_x_acc, eval_x_acc_mask, eval_y_acc)
                loss = criterion(out.float(), eval_y.float())

            "Record Metrics"
            eval_loss += loss.data.item()
            true = eval_y.cpu().numpy()
            pred = out.cpu().detach().numpy()
            eval_r2 += r2_score(true, pred)
            eval_rmse += np.sqrt(mean_squared_error(true, pred))
            eval_mae += mean_absolute_error(true, pred)

        "Compute Metrics"
        eval_loss = eval_loss / len(data_loader)
        eval_r2 = eval_r2 / len(data_loader)
        eval_rmse = eval_rmse / len(data_loader)
        eval_mae = eval_mae / len(data_loader)

        return eval_loss, eval_r2, eval_rmse, eval_mae


def train(train_loader, val_loader, test_loader, num_epochs, criterion, optimizer):
    model = AccSurrogateNet_GRU_Residual(da_dim=111, op_dim=768, embed_size=100, DaAttention_factor=6,
                                      n_hidden_1=512, n_hidden_2=512, n_hidden_3=256, n_hidden_4=256, out_dim=1,
                                      BiDirection=True, global_max_oplen=12)
    if torch.cuda.is_available():
        model = model.to(device)

    model_name = model.__class__.__name__
    timestamp = time.strftime("%Y%m%d%H%M%S")
    print("-----------------------------")
    print("Model Name:", model_name, "Time Stamp: ", timestamp)
    print("-----------------------------")


    val_r2_max = -1000
    train_loss_list = []
    val_loss_list = []
    val_r2_list = []

    add = 0
    scaler = GradScaler()
    print("Start Training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        for i, data in enumerate(train_loader, 0):
            " Prepare data "
            train_x_da, train_x_ops, train_x_ops_mask, train_y, train_x_acc, train_x_acc_mask ,train_y_acc = data
            if torch.cuda.is_available():
                train_x_da = train_x_da.to(device)
                train_x_ops = train_x_ops.to(device)
                train_y = train_y.to(device)
                train_x_acc = train_x_acc.to(device)
                train_y_acc = train_y_acc.to(device)
            "Preprocess for possible nan&inf"
            nan_mask = torch.isnan(train_x_da)
            inf_mask = torch.isinf(train_x_da)
            train_x_da[nan_mask] = 0
            max_val = torch.max(train_x_da[~inf_mask])
            train_x_da[inf_mask] = max_val
            " Forward "
            with autocast():
                out = model(train_x_da, train_x_ops, train_x_ops_mask ,train_x_acc, train_x_acc_mask, train_y_acc)
                loss = criterion(out.float(), train_y.float())
                train_loss = loss.data.item()
            " Backward "
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            " Update "
            scaler.step(optimizer)
            scaler.update()

        if epoch % 1 == 0:
            val_loss, val_r2, val_rmse, val_mae = evaluate(val_loader, model, criterion)  # evaluate the validation set
            train_loss1, train_r2, train_rmse, train_mae = evaluate(train_loader, model, criterion)  # evaluate the training set
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss)
            val_r2_list.append(val_r2)
            print(
                f'@@@epoch: {epoch}, Train Loss:{train_loss:.6f}, Train R2:{train_r2:.6f}, Train RMSE:{train_rmse:.6f}, Train MAE:{train_mae:.6f}, Val Loss:{val_loss:.6f}, Val R2:{val_r2:.6f}, Val RMSE:{val_rmse:.6f}, Val MAE:{val_mae:.6f}')

            if val_r2 > val_r2_max:
                val_r2_max = val_r2
                model_best = copy.deepcopy(model)
                epoch_best = epoch

        if len(train_loss_list) >= 2:
            add = add+1 if train_loss_list[-1] > train_loss_list[-2] else 0
            if add >= 5:
                print("Exit Training...")
                print(epoch_best, val_r2_max)
                test_loss, test_r2, test_rmse, test_mae = evaluate(test_loader, model, criterion)  # evaluate the testing set
                print(f'Results----Test Loss:{test_loss:.6f}, Test R2:{test_r2:.6f}')
                break

    return model_best
