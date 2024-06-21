# SETUP --------------------------
from src.toolbox import *
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from transformers import AutoTokenizer, GPT2LMHeadModel

import time
import yaml

import mlflow

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
with open("config/experiment_test.yaml", 'r') as stream:
    ddConfig = yaml.safe_load(stream)

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
        
        if (iEpoch % 100 == 0) or (iEpoch == 1):
            print(f"Epoch {iEpoch}, Loss: {loss.item()}, Avg. time per Epoch {np.mean(lTime):.2f} sec")
        
        if iEpoch >= iBreakAtEpochs:
            break
        
        if loss.item() < dEps:
            break