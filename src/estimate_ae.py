# SETUP --------------------------
from analysis.toolbox import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer, GPT2LMHeadModel, PhrasalConstraint
from undecorated import undecorated
from types import MethodType

import time
import pickle
import io
import os
from tqdm import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device = "cpu"
torch.set_default_device(device)
print(f"Using {device} device")
print(torch.tensor([1, 2, 3]).device)

# MAGICKS --------------------------
s_model = "gpt2"
iSeed = 4523522
sPadding = "left"
sNameAEModel = "autoencoder_hair20_eos"

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
# lTargetStrings = [
#     "Experience 50% more visible shine after just one use.",
#     "Formulated with light-reflecting technology for a glossy finish.",
#     "Transform dull strands into radiant, luminous locks."
# ]
# lTargetStrings = [
#     "Locks in moisture to amplify hair's natural luster.",
#     "Achieve salon-quality shine without leaving home.",
#     "Visible reduction in dullness, replaced with stunning shine.",
#     "Say goodbye to lackluster hair, hello to mirror-like shine.",
#     "Clinically proven to enhance shine by up to 70%.", # ^tangible
#     "Elevate your confidence with hair that gleams under any light.",
#     "Embrace the allure of luminous hair that turns heads.",
#     "Unleash the power of radiant hair that speaks volumes.",
#     "Transform your look with hair that exudes brilliance.",
#     "Feel the difference of hair that shines with vitality and health."]

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
dLearningRatePrefit = 1e1
dLearningRate = 1
dLearningRate = 5
optimizer = optim.Adam(ae.parameters(), lr=dLearningRate)
iEpoch = 0
iEpochBurnin = 0
dEps = 1


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

lConstraints = [PhrasalConstraint(tokenizer(sentence, add_special_tokens=False).input_ids) for sentence in lTargetStrings]
lForceWords = [tokenizer(sentence, add_special_tokens=False).input_ids for sentence in lTargetStrings]
print("Force Words")
print(lForceWords)

tTargetStrings, lLengths = TokenizeClaims(lTargetStrings, tokenizer)
#lForceWords = [tokenizer(sentence, add_special_tokens=False).input_ids for sentence in lTargetStrings]
print(lConstraints)

# Get input embeddings of model for tokenized sentences
# tInputEmbeddings = model.get_input_embeddings()(torch.tensor([tokenizer.encode(sentence, add_special_tokens=True) for sentence in lPromptStrings]))
# tInputEmbeddings = torch.mean(tInputEmbeddings, dim=1).squeeze()
# print(f"IE shape {tInputEmbeddings.shape}")

iMaxTokens = max(lLengths) + 1 # add one for eos, see TokenizeClaims() fct.
iMinTokens = min(lLengths)
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
        num_beams=1,
        num_return_sequences=i_num_return,
        return_dict_in_generate=True,
        output_scores=True)

fn_generate_max_constr = lambda x: model.generate_with_grad(
    inputs_embeds=x,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=iMaxTokens,
    #constraints = lConstraints,
    force_words_ids = lForceWords,
    num_beams=10,
    num_return_sequences=i_num_return,
    return_dict_in_generate=True,
    output_scores=True)

fn_generate_max = lambda x: fn_generate_i(x, iMaxTokens)


# TODO: Init ae with lin comb of summary
# TODO: Are the EOS tokens working?

tic_total = time.time()
tic = time.time()
lLoss = []

lAE = []

for i in tqdm(range(iClaims)):
    
    tic_it = time.time()

    # temporary objects
    ae_it = Autoencoder(1, iEncodingDim, iOutputSize)
    optimizer_it = optim.Adam(ae_it.parameters(), lr=dLearningRatePrefit)
    OHETarget_it = create_input_data(1)
    TargetStrings_it, _ = TokenizeClaims([lTargetStrings[i]], tokenizer)
    print([tokenizer(lTargetStrings[i], add_special_tokens=False).input_ids])
    
    fn_generate_max_constr = lambda x: model.generate_with_grad(
        inputs_embeds=x,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=iMaxTokens,
        force_words_ids = [tokenizer(sentence, add_special_tokens=False).input_ids for sentence in [lTargetStrings[i]]],
        num_beams=10,
        num_return_sequences=i_num_return,
        return_dict_in_generate=True,
        output_scores=True)

    lAE.append(fit_ae(fn_generate_max_constr, fn_generate_max, tokenizer, ae_it, optimizer_it, OHETarget_it, TargetStrings_it, iMaxTokens, iClaims, iVocab, dEps, i))
    toc_it = time.time()
    print(f"Iteration {i} time: {np.round(toc_it - tic_it, 2)} s")

toc = time.time()
print(f"Total time: {np.round(toc - tic_total, 2)} s")
torch.save(torch.stack([ae(OHETarget_it) for ae in lAE], dim=1), "test_indv_embeddings.pt")

tInputEmbeddings = torch.load("test_indv_embeddings.pt").squeeze(0)
print(tInputEmbeddings)
print(tInputEmbeddings.shape)

# Burn-in period
# Using an Adam Optimizer with lr = 0.1
iBurnIn = 1000
iEpochBurnin = 0
ae_burnin = Autoencoder(iClaims, iEncodingDim, iOutputSize)
optimizer_burnin = torch.optim.Adam(ae_burnin.parameters(),
                             lr = dLearningRate,
                             weight_decay = 1e-8)
criterion_burnin = torch.nn.MSELoss(reduction='mean')

while iEpochBurnin < iBurnIn:
    tAEOutputs = ae_burnin(tOHETarget).reshape(tOHETarget.shape[0], 1, -1)
    loss_burnin = criterion_burnin(tAEOutputs.squeeze(), tInputEmbeddings)
    print(f"Burn-in {iEpochBurnin}: {loss_burnin.item()}")
    optimizer_burnin.zero_grad()
    loss_burnin.backward(retain_graph=True)
    optimizer_burnin.step()
    iEpochBurnin += 1

# # 3 Training together
iEpoch = 0

ae.load_state_dict(ae_burnin.state_dict())
optimizer = torch.optim.Adam(ae.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)
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
lLoss = []
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

print(f"Verify the final generation...\n\n")
tGenNew = fn_generate_max(ae(tOHETarget).reshape(tOHETarget.shape[0], 1, -1)).sequences
for i in range(tOHETarget.shape[0]):
    print(f"Claim {i}: {tokenizer.decode(tGenNew[i])}")


# Encoding the data using the trained autoencoder
torch.save(ae.state_dict(), sNameAEModel + "_dict.pt")
model_scripted = torch.jit.script(ae) # Export to TorchScript
model_scripted.save(sNameAEModel + ".pt") # Save


# FUNCTIONS --------------------------

# MAIN --------------------------
if __name__ == "__main__":
    
    print("pause")