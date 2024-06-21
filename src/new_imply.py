# toolbox.py^ --------------------------

# Read in the sentences from lClaims
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer, GPT2LMHeadModel, PhrasalConstraint

import time
import pickle
import io
import os

import mlflow

def TokenizeClaims(lClaims, tokenizer):
    # Tokenize the sentences
    tokenized_sentences = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in lClaims]

    # Find the maximum length of the tokenized sentences
    lLengths = [len(sentence) for sentence in tokenized_sentences]
    max_length = max(len(sentence) for sentence in tokenized_sentences) + 1 # add on to account for the eos token

    # Pad the tokenized sentences on the left to have the same length
    padded_sentences = [sentence + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * (max_length - len(sentence) - 1) for sentence in tokenized_sentences]
    
    # Convert the padded sentences to tensors
    padded_sentences = torch.tensor(padded_sentences)

    return padded_sentences, torch.tensor(lLengths)

def create_input_data(iClaims):
    return torch.eye(iClaims)

def load_data(sDataPath):
    with open(sDataPath, 'r') as file:
        lClaims = file.readlines()
    lClaims = [claim.replace('\n', '') for claim in lClaims]
    return lClaims

# Defining Autoencoder model
class Autoencoder(nn.Module):
   def __init__(self, input_size, encoding_dim, output_size):
       super(Autoencoder, self).__init__()
       self.encoder = nn.Sequential(
           nn.Linear(input_size, encoding_dim, bias=True)
       )
       self.decoder = nn.Sequential(
           nn.Linear(encoding_dim, output_size, bias=True)#,
           #nn.Dropout(p = 10/768),
           #nn.LayerNorm(output_size)
       )

   def forward(self, x):
       x = self.encoder(x)
       x = self.decoder(x)
       return x
   
   
def ForwardPass(tOHETarget, tInputEmbeddings, tTarget, llm, ae, iBatchSize):
        tSummaryEmbedding = ae(tOHETarget).unsqueeze(1)
        tInputs = torch.cat([tSummaryEmbedding, tInputEmbeddings], dim=1)
        
        # NOTE: Necessary to keep making use of batching of the LLM, otherwise runs sequential ...
        loss = torch.tensor([0], dtype=torch.float32, requires_grad=True)
        for i, b in enumerate(range(0, iClaims, iBatchSize)):
            # NOTE: What if we exceed dim?, unit test
            out = llm(inputs_embeds = tInputs[b:((i + 1)*iBatchSize), :, :], labels=tTarget[b:((i + 1)*iBatchSize), :])
            loss = loss + out[0]
        del out
        return loss.item() # use item() to save memory
   
   
def plot_results(dTime, sNameAEModel, iEncodingDim, ae, lLoss, aLLHistory, lStepsWeightsEncoder, lStepsBiasEncoder, lStepsWeightsDecoder, lStepsBiasDecoder, lGradWeightsEncoder, lGradBiasEncoder, lGradWeightsDecoder, lGradBiasDecoder, iEpoch, dEps, tOHETarget, fn_generate_max, tokenizer):
            
            #print(f"Progress: {aLLHistory}\n")
            
            fig, axs = plt.subplots(2, 4, figsize=(14, 7))
            
            if aLLHistory is not None:
                # Plot 1
                for i in range(tOHETarget.shape[0]):
                    axs[0, 0].plot(aLLHistory[0:iEpoch, i], label=f"Claim {i + 1}")
                axs[0, 0].axhline(y=dEps, linestyle='--', color='black', label="Threshold")
                axs[0, 0].legend()
                axs[0, 0].set_xlabel("Epoch")
                axs[0, 0].set_ylabel("log1p (-LL)")
                axs[0, 0].set_title(f"N-LL {iEpoch}, time per 10 epochs: {dTime} s")

            # Plot 1
            axs[0, 1].plot(np.asarray(lLoss))
            axs[0, 1].axhline(y=dEps, linestyle='--', color='black', label="Threshold")
            axs[0, 1].legend()
            axs[0, 1].set_xlabel("Epoch")
            axs[0, 1].set_ylabel("log1p (-LL)")
            axs[0, 1].set_title(f"N-LL {iEpoch}")

            # # Plot 3
            # print(np.asarray(lStepsWeightsEncoder).shape)
            for i in range(iEncodingDim):
                axs[0, 2].plot(np.max(np.asarray(lStepsWeightsEncoder)[0:iEpoch, i, :], axis=1), linestyle='--', label=f"Weights {i + 1}")
            axs[0, 2].set_xlabel("Epoch")
            axs[0, 2].set_ylabel("Stepsize")
            axs[0, 2].set_title(f"SM Encoder {iEpoch}")
            #axs0, [2].legend()
            
            for i in range(iEncodingDim):
                axs[1, 0].plot(np.max(np.asarray(lStepsWeightsDecoder)[0:iEpoch, :, i], axis=1), linestyle='--', label=f"Weights {i + 1}")
            axs[1, 0].set_xlabel("Epoch")
            axs[1, 0].set_ylabel("Stepsize")
            axs[1, 0].set_title(f"SM Decoder {iEpoch}")
            
            # # # Plot 3
            axs[1, 1].plot(np.asarray(lStepsBiasEncoder)[:, 0], linestyle='--', label=f"Bias Encoder")
            axs[1, 1].set_xlabel("Epoch")
            axs[1, 1].set_ylabel("Stepsize")
            axs[1, 1].set_title(f"SM Bias Encoder {iEpoch}")
            axs[1, 1].legend()

            axs[1, 2].plot(np.asarray(lStepsBiasDecoder)[:, 0], linestyle='--', label=f"Bias Decoder")
            axs[1, 2].set_xlabel("Epoch")
            axs[1, 2].set_ylabel("Stepsize")
            axs[1, 2].set_title(f"SM Bias Decoder {iEpoch}")
            axs[1, 2].legend()
            
            # # # Plot 4
                        
            axs[1, 3].plot(np.asarray(lGradBiasEncoder), linestyle='--', label=f"Grad Bias Encoder")
            axs[1, 3].set_xlabel("Epoch")
            axs[1, 3].set_ylabel("Stepsize")
            axs[1, 3].set_title(f"Grad Bias")
            axs[1, 3].legend()

            axs[1, 3].plot(np.asarray(lGradBiasDecoder), linestyle='--', label=f"Grad Bias Decoder")
            axs[1, 3].set_xlabel("Epoch")
            axs[1, 3].set_ylabel("Stepsize")
            axs[1, 3].set_title(f"Grad Bias")
            axs[1, 3].legend()
            
            # # # Plot 4
            axs[0, 3].plot(np.asarray(lGradWeightsEncoder), linestyle='--', label=f"Grad Weights Encoder")
            axs[0, 3].set_xlabel("Epoch")
            axs[0, 3].set_ylabel("Stepsize")
            axs[0, 3].set_title(f"Grad Weights")
            axs[0, 3].legend()

            axs[0, 3].plot(np.asarray(lGradWeightsDecoder), linestyle='--', label=f"Grad Weights Decoder")
            axs[0, 3].set_xlabel("Epoch")
            axs[0, 3].set_ylabel("Stepsize")
            axs[0, 3].set_title(f"Grad Weights")
            axs[0, 3].legend()
            
            mlflow.log_figure(fig, './plots/dashboard.png')

            # try:
            #     plt.savefig(f"./models/training_monitor_{sNameAEModel}.png")
            # except:
            #     plt.savefig(f"training_monitor_{sNameAEModel}.png")

            # tGenNew = fn_generate_max(ae(tOHETarget).reshape(tOHETarget.shape[0], 1, -1)).sequences
            # for i in range(tOHETarget.shape[0]):
            #     print(f"Claim {i}: {tokenizer.decode(tGenNew[i])}")
                
def plot_results_last10(dTime, sNameAEModel, iEncodingDim, ae, lLoss, aLLHistory, lStepsWeightsEncoder, lStepsBiasEncoder, lStepsWeightsDecoder, lStepsBiasDecoder, lGradWeightsEncoder, lGradBiasEncoder, lGradWeightsDecoder, lGradBiasDecoder, iEpoch, dEps, tOHETarget, fn_generate_max, tokenizer):
            
            #print(f"Progress: {aLLHistory}\n")
            
            fig, axs = plt.subplots(2, 4, figsize=(14, 7))
            
            if aLLHistory is not None:
                # Plot 1
                for i in range(tOHETarget.shape[0]):
                    axs[0, 0].plot(aLLHistory[0:iEpoch, i], label=f"Claim {i + 1}")
                axs[0, 0].axhline(y=dEps, linestyle='--', color='black', label="Threshold")
                axs[0, 0].legend()
                axs[0, 0].set_xlabel("Epoch")
                axs[0, 0].set_ylabel("log1p (-LL)")
                axs[0, 0].set_title(f"N-LL {iEpoch}, time per 10 epochs: {dTime} s")

            # Plot 1
            axs[0, 1].plot(np.asarray(lLoss[-10:]))
            axs[0, 1].axhline(y=dEps, linestyle='--', color='black', label="Threshold")
            axs[0, 1].legend()
            axs[0, 1].set_xlabel("Epoch")
            axs[0, 1].set_ylabel("log1p (-LL)")
            axs[0, 1].set_title(f"N-LL {iEpoch}")

            # # Plot 3
            # print(np.asarray(lStepsWeightsEncoder).shape)
            for i in range(iEncodingDim):
                axs[0, 2].plot(np.max(np.asarray(lStepsWeightsEncoder)[-10:, i, :], axis=1), linestyle='--', label=f"Weights {i + 1}")
            axs[0, 2].set_xlabel("Epoch")
            axs[0, 2].set_ylabel("Stepsize")
            axs[0, 2].set_title(f"SM Encoder {iEpoch}")
            #axs0, [2].legend()
            
            for i in range(iEncodingDim):
                axs[1, 0].plot(np.max(np.asarray(lStepsWeightsDecoder)[-10:, :, i], axis=1), linestyle='--', label=f"Weights {i + 1}")
            axs[1, 0].set_xlabel("Epoch")
            axs[1, 0].set_ylabel("Stepsize")
            axs[1, 0].set_title(f"SM Decoder {iEpoch}")
            #axs[3].legend()
            
            # # # Plot 3
            axs[1, 1].plot(np.asarray(lStepsBiasEncoder)[-10:, 0], linestyle='--', label=f"Bias Encoder")
            axs[1, 1].set_xlabel("Epoch")
            axs[1, 1].set_ylabel("Stepsize")
            axs[1, 1].set_title(f"SM Bias Encoder {iEpoch}")
            axs[1, 1].legend()

            axs[1, 2].plot(np.asarray(lStepsBiasDecoder)[-10:, 0], linestyle='--', label=f"Bias Decoder")
            axs[1, 2].set_xlabel("Epoch")
            axs[1, 2].set_ylabel("Stepsize")
            axs[1, 2].set_title(f"SM Bias Decoder {iEpoch}")
            axs[1, 2].legend()
            
            # # # Plot 4
                        
            axs[1, 3].plot(np.asarray(lGradBiasEncoder[-10:]), linestyle='--', label=f"Grad Bias Encoder")
            axs[1, 3].set_xlabel("Epoch")
            axs[1, 3].set_ylabel("Stepsize")
            axs[1, 3].set_title(f"Grad Bias")
            axs[1, 3].legend()

            axs[1, 3].plot(np.asarray(lGradBiasDecoder[-10:]), linestyle='--', label=f"Grad Bias Decoder")
            axs[1, 3].set_xlabel("Epoch")
            axs[1, 3].set_ylabel("Stepsize")
            axs[1, 3].set_title(f"Grad Bias")
            axs[1, 3].legend()
            
            # # # Plot 4
            axs[0, 3].plot(np.asarray(lGradWeightsEncoder[-10:]), linestyle='--', label=f"Grad Weights Encoder")
            axs[0, 3].set_xlabel("Epoch")
            axs[0, 3].set_ylabel("Stepsize")
            axs[0, 3].set_title(f"Grad Weights")
            axs[0, 3].legend()

            axs[0, 3].plot(np.asarray(lGradWeightsDecoder[-10:]), linestyle='--', label=f"Grad Weights Decoder")
            axs[0, 3].set_xlabel("Epoch")
            axs[0, 3].set_ylabel("Stepsize")
            axs[0, 3].set_title(f"Grad Weights")
            axs[0, 3].legend()
            
            mlflow.log_figure(fig, f'./plots/dashboard_last10_{iEpoch}.png')

            # try:
            #     plt.savefig(f"./models/training_monitor_{sNameAEModel}.png")
            # except:
            #     plt.savefig(f"training_monitor_{sNameAEModel}.png")

            # tGenNew = fn_generate_max(ae(tOHETarget).reshape(tOHETarget.shape[0], 1, -1)).sequences
            # for i in range(tOHETarget.shape[0]):
            #     print(f"Claim {i}: {tokenizer.decode(tGenNew[i])}")


# SETUP --------------------------
import sys

from src.toolbox import *
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from transformers import AutoTokenizer, GPT2LMHeadModel

import time
import yaml

import mlflow
import tempfile
from pathlib import Path

# GPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(torch.cuda.get_device_name(torch.cuda.current_device()))
torch.set_default_device(device)
print(torch.cuda.get_device_name(torch.cuda.current_device()))

print(f"Using {device} device")
print(torch.tensor([1, 2, 3]).device)


# load config
# Read YAML file
print(sys.argv)
with open(f"config/{sys.argv[1]}", 'r') as stream:
    ddConfig = yaml.safe_load(stream)

# with open("config/experiment_hair20.yaml", 'r') as stream:
#     ddConfig = yaml.safe_load(stream)

print(ddConfig)

# MAGICKS --------------------------
s_model = ddConfig["s_model"]
iSeed = 4523522
sPadding = "left"
sNameAEModel = ddConfig["sNameAEModel"]

lTargetStrings = load_data(ddConfig["sDataPath"])
iClaims = len(lTargetStrings)
iEncodingDim = ddConfig["iEncodingDim"]
iOutputSize = ddConfig["iOutputSize"]

torch.manual_seed(iSeed)
model = GPT2LMHeadModel.from_pretrained(s_model, device_map=device, resume_download=True)
#model = model.to(memory_format=torch.channels_last)
tokenizer = AutoTokenizer.from_pretrained(s_model, resume_download=True)

if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

tOHETarget = create_input_data(iClaims)
tTargetStrings, lLengths = TokenizeClaims(lTargetStrings, tokenizer)
iMaxTokens = max(lLengths)

tInputEmbeddings = model.get_input_embeddings()(tTargetStrings)
# TODO: Look into `past_key_values`, can speed things up by caching the hidden states
tTarget = torch.cat([torch.tensor([-100] * iClaims).unsqueeze(-1), tTargetStrings], dim = 1)


# activation, hidden units, bias, other
sAENameArchitecture = ddConfig["sAENameArchitecture"]
ae = Autoencoder(iClaims, iEncodingDim, iOutputSize)
#ae.load_state_dict(torch.load(f"../models/{sNameAEModel}.pt"))

lGradWeightsDecoder = []
lGradBiasDecoder = []
lGradWeightsEncoder = []
lGradBiasEncoder = []
lStepsWeightsDecoder = []
lStepsBiasDecoder = []
lStepsWeightsEncoder = []
lStepsBiasEncoder = []
lStepsWeightsDecoder = []
lStepsBiasDecoder = []
lStepsWeightsEncoder = []
lStepsBiasEncoder = []

try:
    mlflow.end_run()
except:
    pass
# localhost:5050
#mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment(sAENameArchitecture)

# RUN --------------------------
mlflow.pytorch.autolog()
sRunName = ddConfig["sRunName"]

dLearningRate = float(ddConfig["dLearningRate"])
dEps = float(ddConfig["dEps"])
dWeightDecay = float(ddConfig["dWeightDecay"])
if ddConfig["dMaxGrad"] == "None":
    dMaxGrad = None
else:
    dMaxGrad = float(ddConfig["dMaxGrad"])
iBreakAtEpochs = int(ddConfig["iBreakAtEpochs"])
iBatchSize = min(32, iClaims)
bSaveModels = ddConfig["bSaveModels"]

optimizer = optim.Adam(ae.parameters(), lr=dLearningRate, weight_decay=dWeightDecay)

# Start an MLflow experiment
with mlflow.start_run(run_name=sRunName):
    
    mlflow.log_param("Optimizer", optimizer)
    mlflow.log_param("iClaims", iClaims)
    mlflow.log_param("iBatchSize", iBatchSize)
    mlflow.log_param("iBreakAtEpochs", iBreakAtEpochs)
    mlflow.log_param("dEps", dEps)
    mlflow.log_param("dLearningRate", dLearningRate)
    mlflow.log_param("dWeightDecay", dWeightDecay)
    mlflow.log_param("dMaxGrad", dMaxGrad)
    mlflow.log_param("bSaveModels", bSaveModels)
    
    iEpoch = 0
    lLoss = []
    lTime = []
    while True:

        iEpoch += 1
        tic = time.time()

        tSummaryEmbedding = ae(tOHETarget).unsqueeze(1)
        tInputs = torch.cat([tSummaryEmbedding, tInputEmbeddings], dim=1)
        
        optimizer.zero_grad()
        
        # NOTE: Necessary to keep making use of batching of the LLM, otherwise runs sequential ...
        loss = torch.tensor([0], dtype=torch.float32, requires_grad=True)
        for i, b in enumerate(range(0, iClaims, iBatchSize)):
            # NOTE: What if we exceed dim?, unit test
            out = model(inputs_embeds = tInputs[b:((i + 1)*iBatchSize), :, :], labels=tTarget[b:((i + 1)*iBatchSize), :])
            loss = loss + out[0]         

        loss.backward(retain_graph=True)
        lLoss.append(loss.item())

        
        # NOTE: Clip the gradients to avoid exploding gradients
        if dMaxGrad is not None:
            assert dMaxGrad > 0, "Max grad should be positive"
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=dMaxGrad) # adjust for level
        
        tDecoderWOld = ae.decoder[0].weight.cpu().detach().numpy()
        tDecoderBOld = ae.decoder[0].bias.cpu().detach().numpy()
        tEncoderWOld = ae.encoder[0].weight.cpu().detach().numpy()
        tEncoderBOld = ae.encoder[0].bias.cpu().detach().numpy()
        
        lGradWeightsDecoder.append(torch.norm(ae.decoder[0].weight.grad).cpu().detach().numpy())
        lGradBiasDecoder.append(torch.norm(ae.decoder[0].bias.grad).cpu().detach().numpy())
        lGradWeightsEncoder.append(torch.norm(ae.encoder[0].weight.grad).cpu().detach().numpy())
        lGradBiasEncoder.append(torch.norm(ae.encoder[0].bias.grad).cpu().detach().numpy())
        
        optimizer.step()
        
        lStepsWeightsDecoder.append(np.abs(ae.decoder[0].weight.cpu().detach().numpy() - tDecoderWOld))
        lStepsBiasDecoder.append(np.abs(ae.decoder[0].bias.cpu().detach().numpy() - tDecoderBOld))
        lStepsWeightsEncoder.append(np.abs(ae.encoder[0].weight.cpu().detach().numpy() - tEncoderWOld))
        lStepsBiasEncoder.append(np.abs(ae.encoder[0].bias.cpu().detach().numpy() - tEncoderBOld))

        # Log the loss to MLflow
        mlflow.log_metric("nLL", loss.item())
        mlflow.log_metric("iNEpochs", iEpoch)

        toc = time.time()
        dTime = toc - tic
        lTime.append(dTime)
        mlflow.log_metric("TimePerEpoch", toc - tic)
        
        if bSaveModels:
            mlflow.pytorch.save_model(ae, f"log_models/models_{iEpoch}")
        
        if iEpoch % 10 == 0:
            plot_results(dTime, sAENameArchitecture, iEncodingDim, ae, lLoss, None, lStepsWeightsEncoder, lStepsBiasEncoder, lStepsWeightsDecoder, lStepsBiasDecoder, lGradWeightsEncoder, lGradBiasEncoder, lGradWeightsDecoder, lGradBiasDecoder, iEpoch, dEps, tOHETarget, lambda x: model.generate(inputs_embeds = x, max_new_tokens = iMaxTokens + 1, pad_token_id = tokenizer.pad_token_id), tokenizer)
            plot_results_last10(dTime, sAENameArchitecture, iEncodingDim, ae, lLoss, None, lStepsWeightsEncoder, lStepsBiasEncoder, lStepsWeightsDecoder, lStepsBiasDecoder, lGradWeightsEncoder, lGradBiasEncoder, lGradWeightsDecoder, lGradBiasDecoder, iEpoch, dEps, tOHETarget, lambda x: model.generate(inputs_embeds = x, max_new_tokens = iMaxTokens + 1, pad_token_id = tokenizer.pad_token_id), tokenizer)
            fig, axs = plt.subplots(1)
            axs.plot(lLoss)
            mlflow.log_figure(fig, './plots/my_plot.png')
            mlflow.pytorch.log_model(ae, "models")
            plt.close()
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir, "ae.pt")
                torch.save(ae.state_dict(), path)
            # With artifact_path=None write features.txt under
            # root artifact_uri/artifacts directory
            with mlflow.start_run():
                mlflow.log_artifact(path)
        
        if (iEpoch % 100 == 0) or (iEpoch == 1):
            print(f"Epoch {iEpoch}, Loss: {loss.item()}, Avg. time per Epoch {np.mean(lTime):.2f} sec")
        
        if iEpoch >= iBreakAtEpochs:
            break
        
        if loss.item() < dEps:
            break