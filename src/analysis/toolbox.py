# Read in the sentences from lClaims
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
   
   
def plot_results(dTime, sNameAEModel, iEncodingDim, ae, lLoss, aLLHistory, lStepsWeightsEncoder, lStepsBiasEncoder, lStepsWeightsDecoder, lStepsBiasDecoder, lGradWeightsEncoder, lGradBiasEncoder, lGradWeightsDecoder, lGradBiasDecoder, iEpoch, dEps, tOHETarget, fn_generate_max, tokenizer):
            
            #print(f"Progress: {aLLHistory}\n")
            
            fig, axs = plt.subplots(2, 4, figsize=(14, 7))
            
            if aLLHistory is not None:
                # Plot 1
                for i in range(tOHETarget.shape[0]):
                    axs[0, 0].plot(np.log1p(aLLHistory[0:iEpoch, i]), label=f"Claim {i + 1}")
                axs[0, 0].axhline(y=np.log1p(dEps), linestyle='--', color='black', label="Threshold")
                axs[0, 0].legend()
                axs[0, 0].set_xlabel("Epoch")
                axs[0, 0].set_ylabel("log1p (-LL)")
                axs[0, 0].set_title(f"N-LL {iEpoch}, time per 10 epochs: {dTime} s")

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
            axs[0, 2].set_title(f"SM Encoder {iEpoch}")
            #axs0, [2].legend()
            
            for i in range(iEncodingDim):
                axs[1, 0].plot(np.mean(np.asarray(lStepsWeightsDecoder)[0:iEpoch, :, i], axis=1), linestyle='--', label=f"Weights {i + 1}")
            axs[1, 0].set_xlabel("Epoch")
            axs[1, 0].set_ylabel("Stepsize")
            axs[1, 0].set_title(f"SM Decoder {iEpoch}")
            #axs[3].legend()
            
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
                    axs[0, 0].plot(np.log1p(aLLHistory[0:iEpoch, i]), label=f"Claim {i + 1}")
                axs[0, 0].axhline(y=np.log1p(dEps), linestyle='--', color='black', label="Threshold")
                axs[0, 0].legend()
                axs[0, 0].set_xlabel("Epoch")
                axs[0, 0].set_ylabel("log1p (-LL)")
                axs[0, 0].set_title(f"N-LL {iEpoch}, time per 10 epochs: {dTime} s")

            # Plot 1
            axs[0, 1].plot(np.log1p(np.asarray(lLoss[-10:])))
            axs[0, 1].axhline(y=np.log1p(dEps), linestyle='--', color='black', label="Threshold")
            axs[0, 1].legend()
            axs[0, 1].set_xlabel("Epoch")
            axs[0, 1].set_ylabel("log1p (-LL)")
            axs[0, 1].set_title(f"N-LL {iEpoch}")

            # # Plot 3
            # print(np.asarray(lStepsWeightsEncoder).shape)
            for i in range(iEncodingDim):
                axs[0, 2].plot(np.mean(np.asarray(lStepsWeightsEncoder)[-10:, i, :], axis=1), linestyle='--', label=f"Weights {i + 1}")
            axs[0, 2].set_xlabel("Epoch")
            axs[0, 2].set_ylabel("Stepsize")
            axs[0, 2].set_title(f"SM Encoder {iEpoch}")
            #axs0, [2].legend()
            
            for i in range(iEncodingDim):
                axs[1, 0].plot(np.mean(np.asarray(lStepsWeightsDecoder)[-10:, :, i], axis=1), linestyle='--', label=f"Weights {i + 1}")
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
                
                
def fit_ae(fn_generate_max_constr, fn_generate_max, tokenizer, ae, optimizer, OHETarget, TargetStrings, iMaxTokens, iClaims, iVocab, dEps, i):
    iEpoch = 0
    lLoss = list()
    bStop = False
    tic = time.time()
    while not bStop:

        # Forward pass, Generate and compute loss
        tAEOutputs = ae(OHETarget).reshape(OHETarget.shape[0], 1, -1)

        # BUG: Generated less that iMaxTokens elements for all claims, then crashed
        outputs = fn_generate_max_constr(tAEOutputs)
        
        # TODO revert to outputs.scores
        loss, loss_monitor = LLikelihood(outputs.scores, TargetStrings, iMaxTokens, iClaims, iVocab)
        lLoss.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        # Check stopping criterion; logging
        bStop = loss < dEps        
        optimizer.step()

        # Monitoring
        if (iEpoch % 10 == 0) or bStop:
            toc = time.time()
            plt.plot(np.log1p(lLoss))
            plt.axhline(y=np.log1p(dEps), linestyle='--', color='black')
            plt.legend()
            print(f"time: {np.round(toc - tic, 2)} s")
            plt.savefig(f"./models/training_monitor_monitor.png")
            tic = time.time()
            tGenNew = fn_generate_max(ae(OHETarget).reshape(OHETarget.shape[0], 1, -1)).sequences
            tGenNewConstrNew = fn_generate_max_constr(ae(OHETarget).reshape(OHETarget.shape[0], 1, -1)).sequences
            print(f"Free: {tokenizer.decode(tGenNew[0])}")
            print(f"Constrained: {tokenizer.decode(tGenNewConstrNew[0])}")
            print("\n")

        iEpoch = 1 + iEpoch
    print(f"Verify the final generation...\n\n")
    tGenNew = fn_generate_max(ae(OHETarget).reshape(OHETarget.shape[0], 1, -1)).sequences
    print(f"Claim {i}: {tokenizer.decode(tGenNew[0])}")
    return ae