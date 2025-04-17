import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model = 2, row_dim = 0, col_dim = 1):
        '''
        the initialization for the masked self-attention class is exactly the same 
        as the initialization for the self-attention class 
        '''
        super().__init__()

        self.W_q = nn.Linear(in_features = d_model, out_features = d_model, bias = False)
        self.W_k = nn.Linear(in_features = d_model, out_features = d_model, bias = False)
        self.W_v = nn.Linear(in_features = d_model, out_features = d_model, bias = False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encodings, mask = None):

        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)
        '''
        this bit of code calculates masked self-attention(Q,K,V)
        '''
        similarities = torch.matmul(q,k.transpose(dim0 = self.row_dim, dim1 = self.col_dim))
        scaled_similarities = similarities / torch.tensor(k.size(self.col_dim)**0.5)
        '''
        we use the masked_fill method to replace the values the mask indicates (by True) by -1e9 (-inf)
        '''
        if mask is not None:
            scaled_similarities = scaled_similarities.masked_fill(mask = mask, value = -1e9)
            attention_percents = F.softmax(scaled_similarities, dim = self.col_dim)
            attention_scores = torch.matmul(attention_percents, v)

        return attention_scores

torch.manual_seed(42)

maskedSelfAttention = MaskedSelfAttention(d_model = 2, row_dim = 0,col_dim = 1)
'''
we create a 3x3 matrix of ones, then we pass it to torch.tril which turns the 1s in the upper triangle to 0s
then we add mask = mask == 0 to convert 0s into True and 1s into Falses
'''
mask = torch.tril(torch.ones(3,3))
mask = mask == 0

X = torch.tensor([[1.16, 0.23],
                  [0.57, 1.36],
                  [4.41, -2.16]], dtype=torch.float)

print("\nMasked Self-Attention one step:\n")
print(maskedSelfAttention.forward(X,mask))
