import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(300, 128, 3, dropout=0.2, batch_first=True)
        self.fcn1 = nn.Linear(128, 64)
        self.fcn2 = nn.Linear(64, 32)
        self.fcn3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(p=0.25)
        
    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(3,batch_size,128)), Variable(torch.zeros(3,batch_size,128)))
    
    def forward(self, x, batch_size):
        self.hidden = self.init_hidden(batch_size)
        output, hn = self.lstm(x, self.hidden)
        output, seq_index = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        new_tensor = Variable(torch.zeros(len(seq_index), 128))

        for idx, batch_element in enumerate(output):
            col_index = seq_index[idx] - 1
            col_element = batch_element[col_index]
            new_tensor[idx] = col_element
        
        output = self.fcn1(new_tensor)
        output = self.dropout(output)
        output = self.fcn2(output)
        output = self.dropout(output)
        output = self.fcn3(output)
        output = F.log_softmax(output, dim=1)
        return output