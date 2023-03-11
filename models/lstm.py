import torch
import torch.nn as nn
import torch.nn.functional as F

class MTLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MTLSTM, self).__init__()

        self.input_size = input_size
        self.hiddnen_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, input):
        output, _ = self.lstm(input)
        y = self.fc(torch.mean(output, dim=1))
        return y