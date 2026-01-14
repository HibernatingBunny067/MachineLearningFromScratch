import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embed,4*config.embed),
            nn.ReLU(),
            nn.Linear(4*config.embed,config.embed),
            nn.Dropout(config.dropout)
        )
    def forward(self,x):
        return self.net(x)

class causalAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.embed % config.n_heads == 0
        self.c_attn = nn.Linear(config.embed,3*config.embed,bias=False)
        self.proj = nn.Linear(config.embed,config.embed,bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_heads = config.n_heads
        self.n_embed = config.embed
    def forward(self,x):
        B,T,C = x.shape

        kqv = self.c_attn(x) ## x(B,T,C) -> kqv (B,T,3*C)

        k,q,v = kqv.split(self.n_embed,dim=2) ## each matrix has dimension (B,T,C)

        k = k.view(B,T,self.n_heads,C // self.n_heads).transpose(1,2) ## k(B,T,C) -> (B,T,n_heads,head_size) -> (B,n_heads,T,head_size)
        q = q.view(B,T,self.n_heads,C // self.n_heads).transpose(1,2) ## q(B,T,C) -> (B,T,n_heads,head_size) -> (B,n_heads,T,head_size)
        v = v.view(B,T,self.n_heads,C // self.n_heads).transpose(1,2) ## v(B,T,C) -> (B,T,n_heads,head_size) -> (B,n_heads,T,head_size)

        ##attention calculation for each head we want (T,head_size) which we'll concatenate

        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5) ## wei: (B,n_heads,T,T)
        trill = torch.tril(torch.ones(T,T,device=x.device)).view(1,1,T,T)
        wei = wei.masked_fill(trill == 0,float('-inf'))

        wei = F.softmax(wei,dim=-1)
        wei = self.attn_dropout(wei)
        
        y = wei@v

        y = y.transpose(1,2).contiguous().view(B,T,C)

        out = self.proj(y)
        out = self.resid_dropout(out)
        return out

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.sa = causalAttention(config)
        self.ffwd = FeedForward(config)

        self.ln1 = nn.LayerNorm(config.embed)
        self.ln2 = nn.LayerNorm(config.embed)
    
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, config, vocab_size): # Added config
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, config.embed)
        self.position_embedding_table = nn.Embedding(config.block_size, config.embed)
        self.dropout = nn.Dropout(config.dropout)
        # 3. THE BLOCKS: A stack of Transformer Blocks (Sequential is fine here)
        self.blocks = nn.Sequential(*[
            Block(config) for _ in range(config.n_layer)
        ])
        
        # 4. FINAL LAYER NORM: Standard Pre-Norm practice
        self.ln_f = nn.LayerNorm(config.embed)
        
        # 5. LM HEAD: Projects from embedding dim back to vocab size
        self.lm_head = nn.Linear(config.embed, vocab_size)

        self.block_size = config.block_size

    def forward(self, idx, labels=None):
        B, T = idx.shape

        pos_idx = torch.arange(T, device=idx.device) 
        
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(pos_idx) # (T, C)
        
        x = tok_emb + pos_emb # (B, T, C)
        x = self.dropout(x)
        x = self.blocks(x) # (B, T, C)
        
        x = self.ln_f(x) # (B, T, C)
        
        logits = self.lm_head(x) # (B, T, vocab_size)

        if labels is None:
            loss = None
        else:
            # Reshape for Cross Entropy (Standard PyTorch requirement)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            labels = labels.view(-1)
            loss = F.cross_entropy(logits, labels)

        return logits, loss

    def generate(self, idx, max_length):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_length):
            # NEW: Crop context to the last block_size tokens
            # If idx is longer than block_size, position embeddings will crash!
            idx_cond = idx[:, -self.block_size:]
            
            # Get the predictions
            logits, loss = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


@dataclass
class config:
    n_layer = 4
    block_size = 8
    embed = 256
    n_heads = 16 # (256 / 16 = 16 head_size)

if __name__ == '__main__':
    device = 'cpu'
    
    x = torch.randint(0,65,(4,8),device=device)
    
    # Initialize the BLOCK, not just the attention
    # block = Block(config).to(device)
    model = NanoGPT(config,vocab_size=65).to(device)
    y,loss = model(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", y.shape) # Should match input
    
    # # Check if variance is preserved (Residuals help with this)
    # print("Input Var:", x.var().item())
    # print("Output Var:", y.var().item())