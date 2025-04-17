import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model = 2, row_dim = 0, col_dim = 1):
        super().__init__()

        self.W_q = nn.Linear(in_features = d_model, out_features = d_model, bias = False)
        self.W_k = nn.Linear(in_features = d_model, out_features = d_model, bias = False)
        self.W_v = nn.Linear(in_features = d_model, out_features = d_model, bias = False)

        self.row_dim = row_dim
        self.col_dim = col_dim
        

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask = None):
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        similarities = torch.matmul(q,k.transpose(dim0 = self.row_dim, dim1 = self.col_dim))
        scaled_similarities = similarities / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_similarities = scaled_similarities.masked_fill(mask = mask, value = -1e9)

        attention_percents = F.softmax(scaled_similarities, dim = self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores
'''
since most of the implementation is already done in the attention file, this class is rather short
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model = 2, row_dim = 0, col_dim = 1, num_heads = 1):
        super().__init__()
        '''
        we initiate each attention head as an instance of the Attention class, for a total of num_heads Attention heads
        we store them in a module list (a list of models we can index)
        '''
        self.heads = nn.ModuleList(Attention(d_model, row_dim, col_dim) for _ in range(num_heads))
        '''
        since our multiple attention heads return more attention values than we want, we need to project it back to the number of word embeddings
        we do this via output_projection, that will bring us back to the projection of the model (equivalent to the W0 in the original paper)
        '''
        self.output_projection = nn.Linear(d_model * num_heads, d_model, bias=False)

        self.col_dim = col_dim
    
    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask = None):
        '''
        we run a loop to pass the matrices on to each attention head
        the attention values returned by each head are then concatenated, projected back into d_model and returned
        '''
        return self.output_projection(torch.cat([head(encodings_for_q, encodings_for_k, encodings_for_v) for head in self.heads], dim = self.col_dim))
    
torch.manual_seed(42)

multiHeadAttention = MultiHeadAttention(d_model=2, row_dim=0, col_dim=1, num_heads=2)

encodings_for_q = torch.tensor([[1.16, 0.23],
                  [0.57, 1.36],
                  [4.41, -2.16]], dtype=torch.float)
encodings_for_k = torch.tensor([[1.16, 0.23],
                  [0.57, 1.36],
                  [4.41, -2.16]], dtype=torch.float)
encodings_for_v = torch.tensor([[1.16, 0.23],
                  [0.57, 1.36],
                  [4.41, -2.16]], dtype=torch.float)
'''
if we pass the same values with only 1 head, we're doing the same thing as we where before, and we ge the same results
'''
print(multiHeadAttention(encodings_for_q, encodings_for_k, encodings_for_v))

