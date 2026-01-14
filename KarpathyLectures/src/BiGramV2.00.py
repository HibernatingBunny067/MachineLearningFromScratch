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
            nn.Linear(4*config.embed,config.embed)
        )
    def forward(self,x):
        return self.net(x)

class causalAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.embed % config.n_heads == 0
        self.c_attn = nn.Linear(config.embed,3*config.embed,bias=False)
        self.proj = nn.Linear(config.embed,config.embed,bias=False)

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
        
        y = wei@v

        y = y.transpose(1,2).contiguous().view(B,T,C)

        out = self.proj(y)

        return out



class Head(nn.Module):
    '''Single attention head
        x: (batch_size,time_step,channels(embedding_dimesion))
        key,query,value: (batch_size,head_size)
        wei: (T,T)
        v: (T,head_size)
        out: (T,head_size)

        Note: this is a single head, usually we use multiple heads (n = int(embedding_dim//head_size)) and use them in parallel
    '''

    def __init__(self,input_size,head_size):
        super().__init__()
        self.key = nn.Linear(input_size,head_size,bias=False)
        self.query = nn.Linear(input_size,head_size,bias=False)
        self.value = nn.Linear(input_size,head_size,bias=False)

    def forward(self,x:torch.Tensor):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        trill = torch.tril(torch.ones(T,T,device=x.device))
        wei = q @ k.transpose(-2,-1) * (k.shape[-1])**(-0.5)
        wei = wei.masked_fill(trill == 0,float('-inf'))

        wei = F.softmax(wei,dim=-1)
        
        out = wei@v
        return out 
        

class Bigram(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.lookup_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,labels=None):
        #idx (B,T) -> forward -> (B,T,C)
        logits = self.lookup_table(idx) #(B,T,C) -> (4,8,65)
        if labels is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            labels = labels.view(-1)
            loss = F.cross_entropy(logits,labels)
        return logits,loss

    def generate(self,idx,max_length):
        ##idx is (B,T) for batch and timesteps of the input context
        for _ in range(max_length):
            logits,loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat([idx,idx_next],dim=1)
        return idx

@dataclass
class config:
    embed = 256
    n_heads = 16

if __name__ == '__main__':
    x = torch.randn((4,8,config.embed),device='mps')
    m = causalAttention(config=config).to('mps')
    y=m(x)
    assert x.shape == y.shape
    print(print(1))
    print(y.shape)
    print(x.var())
    print(y.var())