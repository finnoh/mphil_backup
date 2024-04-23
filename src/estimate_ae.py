# SETUP --------------------------
from analysis.toolbox import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer, GPT2LMHeadModel
from undecorated import undecorated
from types import MethodType

import time
import pickle
import io
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(device)
print(f"Using {device} device")
print(torch.tensor([1, 2, 3]).device)

# MAGICKS --------------------------
s_model = "gpt2"
iSeed = 4523522
sPadding = "left"
sNameAEModel = "autoencoder_test"

lTargetStrings = [
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
lTargetStrings = [
    "Locks in moisture to amplify hair's natural luster.",
    "Achieve salon-quality shine without leaving home.",
    "Visible reduction in dullness, replaced with stunning shine.",
    "Say goodbye to lackluster hair, hello to mirror-like shine.",
    "Clinically proven to enhance shine by up to 70%.", # ^tangible
    "Elevate your confidence with hair that gleams under any light.",
    "Embrace the allure of luminous hair that turns heads.",
    "Unleash the power of radiant hair that speaks volumes.",
    "Transform your look with hair that exudes brilliance.",
    "Feel the difference of hair that shines with vitality and health."]
lPromptStrings = [
    "This new haircare product will",
    "This new haircare product will",
    "This new haircare product will",
    "This new haircare product will",
    "This new haircare product will",
    "This new haircare product will",
    "This new haircare product will",
    "This new haircare product will",
    "This new haircare product will",
    "This new haircare product will"]

# Unleash the power of radiant hair that speaks volumes.
# Feel the difference of hair that shines with vitality and health.

# epoch time ~lin with n max tokens
# epoch time ~lin with n claims

i_num_beams = 1
i_no_repeat_ngram_size = 1
i_num_return = 1
iClaims = len(lTargetStrings)
print(f"Number of claims: {iClaims}")
iEncodingDim = 2  # Desired number of outtput dimensions
iOutputSize = 768  # Number of output features
iVocab = 50257
iBurnIn = 500
ae = Autoencoder(iClaims, iEncodingDim, iOutputSize)
ae_burnin = Autoencoder(iClaims, iEncodingDim, iOutputSize)
dLearningRate = 5e-2
dLearningRate = 1
optimizer = optim.Adam(ae.parameters(), lr=dLearningRate)
iEpoch = 0
iEpochBurnin = 0
dEps = 1
bStop = False
aLLHistory = np.ones((1, iClaims))
lGradWeightsDecoder = []
lGradBiasDecoder = []
lGradWeightsEncoder = []
lGradBiasEncoder = []
lStepsWeightsDecoder = []
lStepsBiasDecoder = []
lStepsWeightsEncoder = []
lStepsBiasEncoder = []
aLLHistory[0, :] = np.nan
aLLHistory = np.ones((1, iClaims))
aLLHistory[0, :] = np.nan


# INIT --------------------------
torch.manual_seed(iSeed)

model = GPT2LMHeadModel.from_pretrained(s_model, resume_download=True)
# TODO: Link to this https://github.com/huggingface/transformers/issues/15552 (last comment)
generate_with_grad = model.generate.__closure__[1].cell_contents
model.generate_with_grad = MethodType(generate_with_grad, model)

tokenizer = AutoTokenizer.from_pretrained(s_model, resume_download=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = sPadding
print("EOS TOKEN ID", tokenizer.eos_token_id)

tOHETarget = create_input_data(iClaims)
tTargetStrings, lLengths = TokenizeClaims(lTargetStrings, tokenizer)
print(tTargetStrings)
print(tTargetStrings.shape)

# Get input embeddings of model for tokenized sentences
tInputEmbeddings = model.get_input_embeddings()(torch.tensor([tokenizer.encode(sentence, add_special_tokens=True) for sentence in lPromptStrings]))
tInputEmbeddings = torch.mean(tInputEmbeddings, dim=1).squeeze()
print(f"IE shape {tInputEmbeddings.shape}")

iMaxTokens = max(lLengths) + 1 # add one for eos, see TokenizeClaims() fct.
print(f"Lengths of target strings {lLengths}")
print(f"Max number of tokens {iMaxTokens}")

# TODO: Remove min_new_tokens again?
fn_generate_i = lambda x, i: model.generate_with_grad(
        inputs_embeds=x,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=i,
        #min_new_tokens=int(i),
        num_beams=i_num_beams,
        num_return_sequences=i_num_return,
        return_dict_in_generate=True,
        output_scores=True)
# fn_generate_max = lambda x: model.generate_with_grad(
#         inputs_embeds=x,
#         do_sample=False,
#         eos_token_id=tokenizer.eos_token_id,
#         forced_eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,
#         num_beams=i_num_beams,
#         num_return_sequences=i_num_return,
#         return_dict_in_generate=True,
#         output_scores=True)

fn_generate_max = lambda x: fn_generate_i(x, iMaxTokens)

# Burn-in period
# Using an Adam Optimizer with lr = 0.1
optimizer_burnin = torch.optim.Adam(ae_burnin.parameters(),
                             lr = dLearningRate,
                             weight_decay = 1e-8)
criterion_burnin = torch.nn.MSELoss(reduction='mean')

# TODO: Init ae with lin comb of summary
# TODO: Are the EOS tokens working?

tic_total = time.time()
tic = time.time()
lLoss = []

# while iEpochBurnin < iBurnIn:
#     tAEOutputs = ae_burnin(tOHETarget).reshape(tOHETarget.shape[0], 1, -1)
#     loss_burnin = criterion_burnin(tAEOutputs.squeeze(), tInputEmbeddings)
#     print(f"Burn-in {iEpoch}: {loss_burnin.item()}")
#     optimizer_burnin.zero_grad()
#     loss_burnin.backward(retain_graph=True)
#     optimizer_burnin.step()
#     iEpochBurnin += 1

# tGenNew = fn_generate_max(ae_burnin(tOHETarget).reshape(tOHETarget.shape[0], 1, -1)).sequences
# for i in range(tOHETarget.shape[0]):
#     print(f"Claim {i}: {tokenizer.decode(tGenNew[i])}")

# # init ae with burnin weights
# ae.load_state_dict(ae_burnin.state_dict())

# tGenNew = fn_generate_max(ae(tOHETarget).reshape(tOHETarget.shape[0], 1, -1)).sequences
# for i in range(tOHETarget.shape[0]):
#     print(f"Claim {i}: {tokenizer.decode(tGenNew[i])}")

while not bStop:

    # Forward pass, Generate and compute loss
    tAEOutputs = ae(tOHETarget).reshape(tOHETarget.shape[0], 1, -1)
    # BUG: Generated less that iMaxTokens elements for all claims, then crashed
    outputs = fn_generate_max(tAEOutputs)
    
    loss, loss_monitor = LLikelihood(outputs.scores, tTargetStrings, iMaxTokens, iClaims, iVocab)
    lLoss.append(loss.item())

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()

    # Check stopping criterion; logging
    aLLHistory = np.vstack([aLLHistory, loss_monitor.cpu().detach().numpy()])
    bStop = loss < dEps
    bStop = np.all(aLLHistory[iEpoch, :] < dEps)
    
    tDecoderWOld = ae.decoder[0].weight.cpu().detach().numpy()
    tDecoderBOld = ae.decoder[0].bias.cpu().detach().numpy()
    tEncoderWOld = ae.encoder[0].weight.cpu().detach().numpy()
    # tEncoderBOld = ae.encoder[0].bias.cpu().detach().numpy()
    
    lGradWeightsDecoder.append(torch.norm(ae.decoder[0].weight.grad).cpu().detach().numpy())
    lGradBiasDecoder.append(torch.norm(ae.decoder[0].bias.grad).cpu().detach().numpy())
    lGradWeightsEncoder.append(torch.norm(ae.encoder[0].weight.grad).cpu().detach().numpy())
    # lGradBiasEncoder.append(torch.norm(ae.encoder[0].bias.grad).cpu().detach().numpy())
    
    optimizer.step()
    
    lStepsWeightsDecoder.append(np.abs(ae.decoder[0].weight.cpu().detach().numpy() - tDecoderWOld))
    lStepsBiasDecoder.append(np.abs(ae.decoder[0].bias.cpu().detach().numpy() - tDecoderBOld))
    lStepsWeightsEncoder.append(np.abs(ae.encoder[0].weight.cpu().detach().numpy() - tEncoderWOld))
    # lStepsBiasEncoder.append(np.abs(ae.encoder[0].bias.cpu().detach().numpy() - tEncoderBOld))

    # Monitoring
    if (iEpoch % 10 == 0) or bStop:
        toc = time.time()
        plot_results(sNameAEModel, iEncodingDim, ae, lLoss, aLLHistory, lStepsWeightsEncoder, lStepsBiasEncoder, lStepsWeightsDecoder, lStepsBiasDecoder, lGradWeightsEncoder, lGradBiasEncoder, lGradWeightsDecoder, lGradBiasDecoder, iEpoch, dEps, tOHETarget, fn_generate_max, tokenizer)
        print(f"time: {np.round(toc - tic, 2)} s")
        tic = time.time()

    iEpoch = 1 + iEpoch

toc = time.time()
print(f"Total time: {np.round(toc - tic_total, 2)} s")

# Encoding the data using the trained autoencoder
torch.save(ae.state_dict(), sNameAEModel + "_dict.pt")
model_scripted = torch.jit.script(ae) # Export to TorchScript
model_scripted.save(sNameAEModel + ".pt") # Save


# FUNCTIONS --------------------------

# MAIN --------------------------
if __name__ == "__main__":
    
    print("pause")