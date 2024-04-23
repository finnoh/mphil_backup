import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.autograd as autograd
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#from torchviz import make_dot
from transformers import AutoTokenizer, GPT2LMHeadModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.autograd as autograd
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#from torchviz import make_dot
from transformers import AutoTokenizer, GPT2LMHeadModel

from undecorated import undecorated
from types import MethodType
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import pickle
import io
import os

print(os.getcwd())

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def load_embeddings(sPathEmbeddings, sNameEmbeddingA, sNameEmbeddingB):
    
    # Load pickle file
    with open(sPathEmbeddings + "/" + sNameEmbeddingA, 'rb') as f:
        lEmbeddingsA = CPU_Unpickler(f).load()
    
    with open(sPathEmbeddings + "/" + sNameEmbeddingB, 'rb') as f:
        lEmbeddingsB = CPU_Unpickler(f).load()
        
    tEmbeddingA = torch.concatenate(lEmbeddingsA, dim=1)
    tEmbeddingB = torch.concatenate(lEmbeddingsB, dim=1)
    
    tEmbeddingA = tEmbeddingA.squeeze()
    tEmbeddingB = tEmbeddingB.squeeze()
        
    return tEmbeddingA, tEmbeddingB

sPathEmbeddings = "data/processed/summary_embeddings"
sNameEmbeddingA = "tangible_hair_prec.pkl"
sNameEmbeddingB = "intangible_hair_prec.pkl"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# device = "cpu"
print(f"Using {device} device")

# Hair product
lClaimsHair = [
    "Experience 50% more visible shine after just one use.",
    "Formulated with light-reflecting technology for a glossy finish.",
    "Transform dull strands into radiant, luminous locks.",
    "Infused with nourishing oils that enhance natural shine.",
    "See instant brilliance with our advanced shine-boosting formula.",
    "Locks in moisture to amplify hair's natural luster.",
    "Achieve salon-quality shine without leaving home.",
    "Visible reduction in dullness, replaced with stunning shine.",
    "Say goodbye to lackluster hair, hello to mirror-like shine.",
    "Clinically proven to enhance shine by up to 70%.", # ^tangible
    "Elevate your confidence with hair that gleams under any light.",
    "Embrace the allure of luminous hair that turns heads.",
    "Unleash the power of radiant hair that speaks volumes.",
    "Transform your look with hair that exudes brilliance.",
    "Feel the difference of hair that shines with vitality and health.",
    "Rediscover the joy of hair that beams with inner vibrancy.",
    "Indulge in the luxury of hair that shimmers with elegance.",
    "Step into the spotlight with hair that radiates beauty.",
    "Experience the magic of hair that dazzles with every movement.",
    "Unlock the secret to hair that shines from within, reflecting your inner glow."
]

tEmbeddingA, tEmbeddingB = load_embeddings(sPathEmbeddings, sNameEmbeddingA, sNameEmbeddingB)
tEmbeddings = torch.cat([tEmbeddingA, tEmbeddingB], dim=0)

iClaims = tEmbeddings.shape[0]

# INIT
i_seed_value = 42
s_model = "gpt2"
s_target_string = "The"

torch.manual_seed(i_seed_value)
model = GPT2LMHeadModel.from_pretrained(s_model)
tokenizer = AutoTokenizer.from_pretrained(s_model)
tokenizer.pad_token_id = tokenizer.eos_token_id

# BUG: Issue that greedy search does not find max likelihood sequence? Does this matter?
i_num_beams = 1 # greedy search for i_num_beams = 1
i_no_repeat_ngram_size = 1 # TODO: This creates inplace gradient error
i_num_return = 1
i_max_new_tokens = tokenizer.encode(s_target_string, return_tensors="pt").shape[1]

# TODO: Link to this https://github.com/huggingface/transformers/issues/15552 (last comment)
generate_with_grad = model.generate.__closure__[1].cell_contents
model.generate_with_grad = MethodType(generate_with_grad, model)
input_embedding = autograd.Variable(torch.mean(model.get_input_embeddings()(tokenizer.encode(s_target_string, return_tensors="pt")), 1, keepdim=True), requires_grad=True)
gd = torch.optim.RMSprop([input_embedding],
                            lr=1, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False)

print(f"Dimensions of input embedding: {input_embedding.shape}")
target_sequence = tokenizer.encode(s_target_string, return_tensors="pt")
print("TARGET IN TOKENS")

fn_generate = lambda x: model.generate_with_grad(
        inputs_embeds=x,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=i_max_new_tokens,
        num_beams=i_num_beams,
        num_return_sequences=i_num_return,
        return_dict_in_generate=True,
        output_scores=True)
fn_generate = lambda x: model.generate(
        inputs_embeds=x,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=i_max_new_tokens,
        num_beams=i_num_beams,
        num_return_sequences=i_num_return,
        return_dict_in_generate=True,
        output_scores=True)

s_target_string = "can"
target_sequence = tokenizer.encode(s_target_string, return_tensors="pt")
summary_embedding = autograd.Variable(torch.mean(model.get_input_embeddings()(target_sequence), 1, keepdim=True), requires_grad=True)
argument = autograd.Variable(torch.rand(iClaims), requires_grad=True)
gd = torch.optim.RMSprop([summary_embedding],
                            lr=1e-1, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False)
gd = torch.optim.RMSprop([argument],
                            lr=1e-1, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False)
def create_input_data(iClaims):
    return torch.eye(iClaims)

# Defining Autoencoder model
class Autoencoder(nn.Module):
   def __init__(self, fn_generate, input_size, encoding_dim, output_size):
       super(Autoencoder, self).__init__()
       self.encoder = nn.Sequential(
           nn.Linear(input_size, encoding_dim)
       )
       self.decoder = nn.Sequential(
           nn.Linear(encoding_dim, output_size)
       )

   def forward(self, x):
       x = self.encoder(x)
       x = self.decoder(x)
       return x


tOHETarget = create_input_data(tEmbeddings.shape[0])
tOHETarget = create_input_data(2)

# Setting random seed for reproducibility
torch.manual_seed(4523522)

input_size = tOHETarget.shape[1]  # Number of input features
encoding_dim = 1  # Desired number of output dimensions
output_size = tEmbeddings.shape[1]  # Number of output features
ae = Autoencoder(fn_generate, input_size, encoding_dim, output_size)

# Loss function and optimizer
optimizer = optim.Adam(ae.parameters(), lr=0.003)

# Training the autoencoder
num_epochs = 1000
for epoch in range(num_epochs):
   # Forward pass
   ae_outputs = ae(tOHETarget[0, :].reshape(1, -1))
   print("!")
   outputs = fn_generate(ae_outputs.reshape(-1, 1))
   
   loss = llikelihood(outputs, target_sequence)

   # Backward pass and optimization
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

   # Loss for each epoch
   print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Encoding the data using the trained autoencoder
encoded_data = ae.encoder(tOHETarget).detach().numpy()