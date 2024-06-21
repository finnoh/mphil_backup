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
