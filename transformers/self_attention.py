import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SelfAttention(nn.Module):

    def __init__(self, d_model = 2, row_dim = 0, col_dim = 1):
        '''
        d_model = size of the weight matrices for Q, K, and V
            example: if d_model = 2, we are using 2 word embeddings per token,
                and after adding positional encoding to each word embedding,
                we have 2 encoded values per token, our weight matrices will be 2x2   
        row_dim and col_dim are just convenience parameters to change when necessary
        '''
        super().__init__()
        '''
        this line creates the query weight matrix
        in_features defines how many rows are in the weight matrix -> d_model
        out_features defines the number of columns in the weight matrix -> d_model
        the original Transformer doesn't add bias, so we wont either.  
        '''
        self.W_q = nn.Linear(in_features = d_model, out_features= d_model, bias = False)
        '''
        we do the same thing to create a linear object for the key and the value weight matrices
        '''
        self.W_k = nn.Linear(in_features = d_model, out_features= d_model, bias = False)
        self.W_v = nn.Linear(in_features = d_model, out_features= d_model, bias = False)

        self.row_dim = row_dim
        self.col_dim = col_dim
        '''
        this method is where we will calculate the self-attention values for each token
        token_encodings = word embeddings + positional encoding for each input token
        '''
    def forward(self, token_encodings):
        '''
        nn.Linear does all the calculations for us
        so q, k and v store the query, key and value numbers
        '''
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)
        '''
        this bit of code calculates Attention(Q,K,V)
        '''
        similarities = torch.matmul(q,k.transpose(dim0 = self.row_dim, dim1 = self.col_dim))
        scaled_similarities = similarities / torch.tensor(k.size(self.col_dim)**0.5)
        attention_percents = F.softmax(scaled_similarities, dim = self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores
    
torch.manual_seed(42)

selfAttention = SelfAttention(d_model = 2, row_dim = 0,col_dim = 1)

X = torch.tensor([[1.16, 0.23],
                  [0.57, 1.36],
                  [4.41, -2.16]], dtype=torch.float)
'''
in this example, we are showing what attention should look like after training
however, in the real world, there's no way of knowing the result beforehand
we train attention via supervising the final output of the model, and attention learns implicitly to distribute focus.
'''
Y_target = torch.tensor([[1.0, 0.5],
                         [0.5, 1.0],
                         [1.0, 1.0]], dtype=torch.float)
'''
the choice of optimizer and loss is based on current approaches
'''
optimizer = optim.Adam(selfAttention.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    Y_pred = selfAttention(X)
    loss = loss_fn(Y_pred, Y_target)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0 or epoch == 99:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

print("\nSelf-attention Training\nFinal Output:\n", selfAttention(X).detach())

