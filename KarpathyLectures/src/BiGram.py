import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from KarpathyLectures.src.utils import get_batch,estimate_loss,custom_tokenizer
torch.manual_seed(42)

@dataclass
class config:
    batch_size = 8
    window_size = 8
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    max_iters_training = 5000
    lr = 1e-3
    eval_iters = 300
    data_path = r'data/tiny_shakespeare.txt'

tokenizer = custom_tokenizer(data_path=config.data_path)
train_data,val_data = tokenizer.get_coded_data_split()


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

model = Bigram(tokenizer._size_())
model.to(config.device)

optimizer = torch.optim.AdamW(model.parameters(),lr=config.lr)

print('Starting Training....')
for iter in range(config.max_iters_training):

    if iter% config.eval_iters == 0:
        losses = estimate_loss(model,train_data,val_data,config.eval_iters,config.window_size,config.batch_size,config.device)
        print(f'step {iter+1}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}')

    X,y = get_batch(train_data,config.window_size,config.batch_size,config.device)
    logits,loss = model(X,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Trained model output:\n")
context = torch.zeros((1,1),dtype=torch.long,device=config.device)
print(tokenizer.decode(model.generate(context,500).squeeze(0).tolist()))