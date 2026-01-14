import torch
import math
torch.manual_seed(42)

def get_batch(data,context_window=8,batch_size=8,DEVICE='cpu'):
    idx = torch.randint(len(data)-context_window,(batch_size,)) 
    X = torch.stack([data[i:i+context_window] for i in idx])
    y = torch.stack([data[i+1:i+context_window+1] for i in idx])
    X = X.to(DEVICE)
    y = y.to(DEVICE)
    return X,y

@torch.no_grad()
def estimate_loss(model,train_data,val_data,eval_iters,context_windo,batch_size,device):
    out = {}
    splits = ['train','val']
    model.eval()
    for idx,data in enumerate([train_data,val_data]):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,y = get_batch(data=data,context_window=context_windo,batch_size=batch_size,DEVICE=device)
            _,loss = model(X,y)
            losses[k] = loss
        out[splits[idx]] = losses.mean()
    model.train()
    return out

class custom_tokenizer():
    def __init__(self,data_path) -> None:
        self.data_path = data_path
        self.load()

    def load(self):
        try:
            with open(self.data_path,'r') as f:
                text = f.read()
        except Exception as e:
            print(str(e))

        self.data = text
        self.vocab = sorted(set(text))
        self.ch2idx = {ch:idx for idx,ch in enumerate(self.vocab)}
        self.idx2ch = {idx:ch for idx,ch in enumerate(self.vocab)}
    
    def encode(self,text:str):
        return [self.ch2idx[ch] for ch in text] 
    
    def decode(self,ids:list):
        return ''.join([self.idx2ch[idx] for idx in ids])

    def _size_(self):
        return len(self.vocab)
    
    def get_coded_data_split(self):
        data_tensor = torch.tensor(self.encode(self.data))
        n = int(0.9*len(data_tensor))
        train,val = data_tensor[:n],data_tensor[n:]
        return train,val

def get_lr(it,config):
    # 1. Linear Warmup Phase
    if it < config.warmup_iters:
        return config.max_lr * (it + 1) / config.warmup_iters
    
    # 2. If we go past max_iters, just return min_lr
    if it > config.max_iters:
        return config.min_lr
    
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # ranges 0..1
    return config.min_lr + coeff * (config.max_lr - config.min_lr)
    