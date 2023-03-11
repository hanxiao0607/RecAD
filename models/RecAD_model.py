import torch
import torch.nn as nn

class RecAD(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RecAD, self).__init__()

        self.input_size = input_size
        self.hiddnen_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size*2, input_size)

    def forward(self, window, delta):

        output, _ = self.lstm(window)
        out_wind = self.fc1(torch.mean(output, dim=1))
        out_delta = self.fc2(delta)
        out_concat = torch.cat((out_wind, out_delta), 1)
        pred = self.fc3(out_concat)
        return pred