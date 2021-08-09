from models import *
from dataset import SemanticDataset
from datasets_benchmark import *

import torch
import matplotlib.pyplot as plt


# Load dataset
sts12 = STS12().df_train
dataset = SemanticDataset(sts12)
loader = DataLoader(dataset=dataset, batch_size=16) #, shuffle=True)

# Hyperparameters
lr = 0.00005
epochs = 10

model_checkpoint = 'bert-base-uncased'
device = 'cuda'

# Create class objects - Models
tokenizer = Tokenizer(model_checkpoint)
model = Model(modules=[LanguageModel(model_checkpoint,'cuda'),Pooling()])

# Loss function & Optimizer
loss = CosineSimilarityLoss(model)
optimizer = torch.optim.Adam(loss.parameters(),lr=lr)

# Training
model.train()
loss_record = []

for epoch in range(epochs):
    for batch in loader:
        torch.cuda.empty_cache()
        terms1 = list(batch[0])
        terms2 = list(batch[1])
        scores = batch[2].to(device)
        batch = [terms1,terms2]
        out = tokenizer.tokenize(batch)
        optimizer.zero_grad()
        loss_value = loss(out,scores.float())
        loss_value.backward()
        optimizer.step()


        loss_record.append(loss_value.detach().to('cpu'))

    print('Epochs {}, loss: {}'.format(epoch,loss_value.detach().to('cpu')))
