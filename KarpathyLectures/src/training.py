import torch
from KarpathyLectures.src.NanoGPT import NanoGPT
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from KarpathyLectures.src.utils import get_batch,estimate_loss,custom_tokenizer,get_lr
torch.manual_seed(42)
from tqdm.notebook import tqdm, trange 

@dataclass
class config:
    n_layer = 6
    block_size = 256
    embed = 384
    n_heads = 6
    batch_size = 64
    dropout = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_iters_training = 10000
    max_lr = 3e-4        
    min_lr = 3e-5        
    warmup_iters = 100   
    max_iters = max_iters_training 
    eval_iters = 300
    data_path = r'/kaggle/input/tiny-shakespeare-karpathys-repo/tiny_shakespeare.txt'

# 1. Setup Data & Model
tokenizer = custom_tokenizer(data_path=config.data_path)
train_data, val_data = tokenizer.get_coded_data_split()

model = NanoGPT(config, tokenizer._size_()) # Ensure this method returns int
model.to(config.device)

# 2. Setup Optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr)
trLoss,valLoss = [],[]

best_metric = float('inf') # Start at infinity so first loss is always "better"
patience = 5               # How many evals to wait before stopping
trigger = 0
print('Starting Training....')

# 3. Training Loop
training_loop = trange(config.max_iters_training, desc='Training') 

for step in training_loop:
    # A. Update Learning Rate
    lr = get_lr(step,config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # B. Evaluation (Every eval_iters)
    if step % config.eval_iters == 0 or step == config.max_iters_training - 1:
        losses = estimate_loss(model, train_data, val_data, config.eval_iters, config.block_size, config.batch_size, config.device)
        training_loop.set_postfix(train_loss=losses['train'].item(), val_loss=losses['val'].item(), lr=lr)
        tqdm.write(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        trLoss.append(losses['train'].item())
        valLoss.append(losses['val'].item())
        if(losses['val'] < best_metric):
            best_metric = losses['val']
            trigger = 0
            torch.save(model.state_dict(),'nanoGPT.pth')
        else:
            trigger+=1

    if trigger > patience:
        print('EARLY STOPPING')
        break

    # C. Training Step
    xb, yb = get_batch(train_data, config.block_size, config.batch_size, config.device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) 
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()


# print("Trained model output:\n")
# context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
# print(tokenizer.decode(model.generate(context, 500).squeeze(0).tolist()))