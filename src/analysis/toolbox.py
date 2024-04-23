# Read in the sentences from lClaims
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

# TODO: Consider removing this function
def PadScores(tOut, tLengths, iClaims, iVocab: int=50257):

    tOutAdj = tOut.clone()
    for i in range(iClaims):
        tOutAdj[i, tLengths[i]:, :-1] = float('-inf')
        tOutAdj[i, tLengths[i]:, -1] = 0
        
    # tOutAdj[:, tLengths[:, None]:, :-1] = float('-inf')
    # tOutAdj[:, tLengths[:, None]:, -1] = 0

    # TODO: Fix and double-check this, tOut, get rid of things that are not needed
    # TODO: Set to zero to get rid of it
    # TODO: Do we get the gradient? Ignore the signal from excess tokens

    
    return tOutAdj

def LLikelihood(lScores, tTargetStrings, iMaxTokens, iClaims, iVocab: int=50257):
    
    if len(lScores) < iMaxTokens:
        print("Appending")
        # NOTE: After taking the log_softmax, this comes out as -inf and 0
        tAppend = torch.zeros(iClaims, iVocab)
        tAppend[:, -1] = 1
        lScores = lScores + (tAppend, ) * (iMaxTokens - len(lScores))
    
    # get the prob. scores for all generated tokens, of all claims, mProb is iClaims x iMaxTokens x iVocabSize
    mProb = torch.nn.functional.log_softmax(torch.stack(lScores).transpose(0, 1), dim=2)
    # Use advanced indexing to get the values of the target tokens, mProbSub is iClaims x iMaxTokens; Elements are the log-probabilities of the target tokens at
    # that position
    
    # Add a new dimension to B to make it compatible with A
    tTargetStrings = tTargetStrings.unsqueeze(-1)
    mProbSub = mProb.gather(2, tTargetStrings).squeeze(-1)
    # # BUG: Do we really need the sum over the whole matrix?
    # # mProbSub is iClaims x iMaxTokens
    dLL = torch.sum(mProbSub) * (-1)

    return dLL, torch.sum(mProbSub, dim = 1) * (-1)

# Defining Autoencoder model
class Autoencoder(nn.Module):
   def __init__(self, input_size, encoding_dim, output_size):
       super(Autoencoder, self).__init__()
       self.encoder = nn.Sequential(
           # TODO: Consider killing this bias term
           nn.Linear(input_size, encoding_dim, bias=False)
       )
       self.decoder = nn.Sequential(
           nn.Linear(encoding_dim, output_size, bias=True),
           #nn.Dropout(p = 10/768),
           nn.LayerNorm(output_size)
       )

   def forward(self, x):
       x = self.encoder(x)
       x = self.decoder(x)
       return x
   
   
def plot_results(sNameAEModel, iEncodingDim, ae, lLoss, aLLHistory, lStepsWeightsEncoder, lStepsBiasEncoder, lStepsWeightsDecoder, lStepsBiasDecoder, lGradWeightsEncoder, lGradBiasEncoder, lGradWeightsDecoder, lGradBiasDecoder, iEpoch, dEps, tOHETarget, fn_generate_max, tokenizer):
            
            #print(f"Progress: {aLLHistory}\n")
            
            fig, axs = plt.subplots(2, 3, figsize=(16, 16))
            
            # Plot 1
            for i in range(tOHETarget.shape[0]):
                axs[0, 0].plot(np.log1p(aLLHistory[0:iEpoch, i]), label=f"Claim {i + 1}")
            axs[0, 0].axhline(y=np.log1p(dEps), linestyle='--', color='black', label="Threshold")
            axs[0, 0].legend()
            axs[0, 0].set_xlabel("Epoch")
            axs[0, 0].set_ylabel("log1p (-LL)")
            axs[0, 0].set_title(f"N-LL {iEpoch}")

            # Plot 1
            axs[0, 1].plot(np.log1p(np.asarray(lLoss)))
            axs[0, 1].axhline(y=np.log1p(dEps), linestyle='--', color='black', label="Threshold")
            axs[0, 1].legend()
            axs[0, 1].set_xlabel("Epoch")
            axs[0, 1].set_ylabel("log1p (-LL)")
            axs[0, 1].set_title(f"N-LL {iEpoch}")

            # # Plot 3
            # print(np.asarray(lStepsWeightsEncoder).shape)
            for i in range(iEncodingDim):
                axs[0, 2].plot(np.mean(np.asarray(lStepsWeightsEncoder)[0:iEpoch, i, :], axis=1), linestyle='--', label=f"Weights {i + 1}")
            axs[0, 2].set_xlabel("Epoch")
            axs[0, 2].set_ylabel("Stepsize")
            axs[0, 2].set_title(f"Step made Encoder {iEpoch}")
            #axs0, [2].legend()
            
            for i in range(iEncodingDim):
                axs[1, 0].plot(np.mean(np.asarray(lStepsWeightsDecoder)[0:iEpoch, :, i], axis=1), linestyle='--', label=f"Weights {i + 1}")
            axs[1, 0].set_xlabel("Epoch")
            axs[1, 0].set_ylabel("Stepsize")
            axs[1, 0].set_title(f"Step made Decoder {iEpoch}")
            #axs[3].legend()
            
            # # # Plot 3
            # axs[1, 1].plot(np.asarray(lStepsBiasEncoder)[:, 0], linestyle='--', label=f"Bias Encoder")
            # axs[1, 1].set_xlabel("Epoch")
            # axs[1, 1].set_ylabel("Stepsize")
            # axs[1, 1].set_title(f"Step made Bias Encoder {iEpoch}")
            # axs[1, 1].legend()

            axs[1, 2].plot(np.asarray(lStepsBiasDecoder)[:, 0], linestyle='--', label=f"Bias Decoder")
            axs[1, 2].set_xlabel("Epoch")
            axs[1, 2].set_ylabel("Stepsize")
            axs[1, 2].set_title(f"Step made Bias Decoder {iEpoch}")
            axs[1, 2].legend()

            plt.savefig(f"./models/training_monitor_{sNameAEModel}.png")

            tGenNew = fn_generate_max(ae(tOHETarget).reshape(tOHETarget.shape[0], 1, -1)).sequences
            for i in range(tOHETarget.shape[0]):
                print(f"Claim {i}: {tokenizer.decode(tGenNew[i])}")