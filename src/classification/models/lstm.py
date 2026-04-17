import torch
import torch.nn as nn

# Architecture Constants
N_STEPS = 2
H_SIZE = 128

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=H_SIZE, num_layers=2, num_classes=2, dropout=0.5):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x)
        # pick only the last hidden state
        out = out[:, -1, :] 
        return self.fc(out)
