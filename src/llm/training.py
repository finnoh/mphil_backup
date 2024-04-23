# SETUP --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.autograd as autograd
#from torchviz import make_dot
from transformers import AutoTokenizer, GPT2LMHeadModel

from undecorated import undecorated
from types import MethodType
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# FUNCTIONS --------------------------

class ReverseEngineer:
    
    def __init__(self, target_sequence: torch.Tensor):
        self.target_sequence = target_sequence

    def llikelihood(self, output: tuple) -> float:
        """Calculate the log-likelihood of the target tokens in the output sequence.
        NOTE: This only works for greedy search, not beam search (i.e. set num_beams=1).
        NOTE: We use a minimizer

        Args:
            output (tuple): Output from a HF LLM text generation model. Only pass one sequence.
            target_sequence (torch.Tensor): The target tokens (the sequence) to calculate the log-likelihood of.

        Returns:
            float: log-llikelihood for the sequence
        """

        # get scores and the sequences
        scores = torch.stack(output.scores).reshape(len(output.scores), -1).transpose(0, 1)

        # get the scores into a matrix, use log_softmax
        m_prob = torch.nn.functional.log_softmax(scores, dim=0)
        m_prob_sub = m_prob[self.target_sequence.squeeze()]

        # get the diagonal of the matrix
        v_trans_prob = torch.diag(m_prob_sub)
        d_llikelihood = torch.sum(v_trans_prob) * (-1)

        return d_llikelihood

    def training_joint(self, summary_embedding, tokenizer, gd, fn_generate):
            # training based on p(t_1, t_2, ..., t_n | D)
            print(summary_embedding.shape)

            summary_embedding_start = torch.clone(summary_embedding) # save it
            output_start = fn_generate(summary_embedding_start)

            print("START\n")
            print(summary_embedding_start)
            print(tokenizer.decode(output_start.sequences[0]))
            print(output_start.sequences[0])

            ll = self.llikelihood(output_start)

            print(ll)
            history_gd = []

            for i in tqdm(range(10000)):
                if i+1 == 1:
                    print("\nFirst iteration (before any training)\n")
                    print(tokenizer.decode(fn_generate(summary_embedding).sequences[0]))

                gd.zero_grad()

                # generate output for the current input
                objective = self.llikelihood(fn_generate(summary_embedding))
                objective.backward()
                gd.step()
                history_gd.append(objective.item())

                if (i > 1) and (i % 10 == 0):
                    print("\n")
                    print(tokenizer.decode(fn_generate(summary_embedding).sequences[0]))
                    print(f"neg. LL: {history_gd[-1]}")
                    print(f"Delta: {np.abs(history_gd[-1] - history_gd[-2])}")
                    print("Mean abs gradient: ", torch.abs(summary_embedding.grad).mean().item())

                if (i>1) and ((torch.abs(summary_embedding.grad).mean().item()) < .001):
                    print("\nConvergence achieved in ", i+1, " iterations")
                    print("-LogL Value: ", objective.item())
                    print("Mean abs gradient: ", torch.abs(summary_embedding.grad).mean().item())
                    break

            print("\n")
            print("\n")
            print("RESULT\n")

            output_final = fn_generate(summary_embedding)
            
            return summary_embedding

# PARALLEL SETUP
def tp_ll_from_summary(tensor_summary, l_target_sequence, l_target_sequence_embeddings, fn_generate):
        l_llike = list()
        tensor_llike = torch.tensor([0.0] * len(l_target_sequence))
        # generate output for the current input
        l_generations = list()
        for i, target in enumerate(l_target_sequence):
            if i == 0:
                l_generations.append(fn_generate(tensor_summary))
            else:
                l_generations.append(fn_generate(torch.cat([tensor_summary, l_target_sequence_embeddings[i - 1]], dim=1)))
        
        # TODO: Vectorize this    
        for i, gen in enumerate(l_generations):
            tensor_llike[i] = llikelihood(gen, l_target_sequence[i][0][-1])

        return torch.sum(tensor_llike)

def tp_ll_from_summary_threading(tensor_summary, l_target_sequence, l_target_sequence_embeddings, fn_generate):
        l_llike = list()
        #with ThreadPoolExecutor() as executor:
        with ProcessPoolExecutor() as executor:
            futures = []
            
            for i, target in enumerate(l_target_sequence):
                if i == 0:
                    futures.append(executor.submit(llikelihood, fn_generate(tensor_summary), l_target_sequence[0][0][-1]))
                else:
                    futures.append(executor.submit(llikelihood, fn_generate(torch.cat([tensor_summary, l_target_sequence_embeddings[i - 1]], dim=1)), l_target_sequence[i][0][-1]))
            
            for future in futures:
                l_llike.append(future.result())
        
            return sum(l_llike)


def training_parallel(target_sequence, summary_embedding, tokenizer, gd, fn_generate, fn_generate_full, model, threading: bool = False):
        # triangle shape
        l_target_sequence = list()
        l_target_sequence_embeddings = list()
        for i, target in enumerate(target_sequence[0]):
            l_target_sequence.append(target_sequence[0][0:(i+1)].reshape(1, -1))
            l_target_sequence_embeddings.append(model.get_input_embeddings()(l_target_sequence[i]))

        history_gd = []
        
        for i in tqdm(range(10000)):
            gd.zero_grad()
            
            if threading:
                objective = tp_ll_from_summary_threading(summary_embedding, l_target_sequence, l_target_sequence_embeddings, fn_generate)
            else:
                objective = tp_ll_from_summary(summary_embedding, l_target_sequence, l_target_sequence_embeddings, fn_generate)

            # NOTE: Need this here. What are the implications?
            objective.backward(retain_graph = True)
            gd.step()
            history_gd.append(objective.item())

            if (i > 1) and (i % 5 == 0):
                #print(input_embedding)
                print("\n")
                print(tokenizer.decode(fn_generate_full(summary_embedding).sequences[0]))
                print(f"neg. LL: {history_gd[-1]}")
                print(f"Delta: {np.abs(history_gd[-1] - history_gd[-2])}")
                print("Mean abs gradient: ", torch.abs(summary_embedding.grad).mean().item())

            if (i>1) and ((# The code is calculating the mean of the absolute values of the gradients
            # of the `summary_embedding` tensor using PyTorch. The `.grad` attribute is
            # used to access the gradients of the tensor, `torch.abs()` is used to get
            # the absolute values of the gradients, `.mean()` is used to calculate the
            # mean of these absolute values, and `.item()` is used to get the mean as a
            # Python number.
            torch.abs(summary_embedding.grad).mean().item()) < .001):
                #if (i>1) and (np.abs(history_gd[-1] - history_gd[-2]) < .01):
                print("\nConvergence achieved in ", i+1, " iterations")
                print("-LogL Value: ", objective.item())
                print("Mean abs gradient: ", torch.abs(summary_embedding.grad).mean().item())
                print(tokenizer.decode(fn_generate_full(summary_embedding).sequences[0]))
                break
            
        return summary_embedding
    

def training_fwinduction(target_sequence, summary_embedding, tokenizer, gd, fn_generate_i, fn_generate_full, model):
    print(summary_embedding.shape)
    # triangle shape
    l_target_sequence = list()
    l_target_sequence_embeddings = list()
    for i, target in enumerate(target_sequence[0]):
        l_target_sequence.append(target_sequence[0][0:(i+1)].reshape(1, -1))
        l_target_sequence_embeddings.append(model.get_input_embeddings()(l_target_sequence[i]))

    summary_embedding_start = torch.clone(summary_embedding) # save it
    output_start = fn_generate_full(summary_embedding_start)

    print("START\n")
    print(summary_embedding_start)
    print(tokenizer.decode(output_start.sequences[0]))
    print(output_start.sequences[0])

    ll = llikelihood(output_start, target_sequence)
    
    for j in range(len(l_target_sequence)):
        print(j)
        print(ll)
        history_gd = []

        for i in tqdm(range(10000)):
            if i+1 == 1:
                print("\nFirst iteration (before any training)\n")
                print(tokenizer.decode(fn_generate_full(summary_embedding).sequences[0]))

            gd.zero_grad()

            # generate output for the current input
            objective = llikelihood(fn_generate_i(summary_embedding, j + 1), target_sequence)
            objective.backward()
            gd.step()
            history_gd.append(objective.item())

            if (i > 1) and (i % 10 == 0):
                #print(summary_embedding)
                print("\n")
                print(tokenizer.decode(fn_generate_full(summary_embedding).sequences[0]))
                print(f"neg. LL: {history_gd[-1]}")
                print(f"Delta: {np.abs(history_gd[-1] - history_gd[-2])}")
                print("Mean abs gradient: ", torch.abs(summary_embedding.grad).mean().item())

            if (i>1) and ((torch.abs(summary_embedding.grad).mean().item()) < .001):
                #if (i>1) and (np.abs(history_gd[-1] - history_gd[-2]) < .01):
                print("\nConvergence achieved in ", i+1, " iterations", " for ", j+1, " tokens")
                print("-LogL Value: ", objective.item())
                print("Mean abs gradient: ", torch.abs(summary_embedding.grad).mean().item())
                break

        print("\n")
        print("\n")
        print("RESULT\n")
    return summary_embedding, history_gd

# TODO: Implement reverse training, see forward expanding generation, but start from last token instead
# TODO: Compare the times of threading inside the function vs multiprocessing on different observations

# MAIN --------------------------
if __name__ == "__main__":
    # MAGICKS --------------------------
    # add other "special" backends
    b_use_backend = True
    i_seed_value = 42
    s_model = 'gpt2'
    #s_model = 'GroNLP/gpt2-small-dutch'

    # INIT
    if b_use_backend:
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            x = torch.ones(1, device=mps_device)
            torch.set_default_device('mps')
            print (f"Using MPS backend {x}")
        elif torch.cuda.is_available():
            print ("Using CUDA backend")
            torch.set_default_device('cuda')
        else:
            print ("Using CPU.")

    #s_prompt_string = "A hot"
    #s_target_string = "Potato"
    # s_target_string = "easy example." # text generated by this prompt for this seed!
    # s_target_string = "does it make sense to compare apples and oranges?" # text generated by this prompt for this seed!
    s_target_string = "Transform Your Hair's Health!!" # text generated by this prompt for this seed!
    # s_target_string = "Transform Your Hair's Health with our Revolutionary Haircare System: Experience Unparalleled Shine, Strength, and Vitality!" # text generated by this prompt for this seed!
    # #s_target_string = "Heeft het zin om appels en sinaasappelen te vergelijken?" # text generated by this prompt for this seed!

    torch.manual_seed(i_seed_value)
    model = GPT2LMHeadModel.from_pretrained(s_model)
    tokenizer = AutoTokenizer.from_pretrained(s_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # BUG: Issue that greedy search does not find max likelihood sequence? Does this matter?
    i_num_beams = 1 # greedy search for i_num_beams = 1
    i_no_repeat_ngram_size = 1 # TODO: This creates inplace gradient error
    i_num_return = 1
    i_max_new_tokens = tokenizer.encode(s_target_string, return_tensors="pt").shape[1]
    
    # INIT
    # TODO: Link to this https://github.com/huggingface/transformers/issues/15552 (last comment)
    generate_with_grad = model.generate.__closure__[1].cell_contents
    model.generate_with_grad = MethodType(generate_with_grad, model)
    input_embedding = autograd.Variable(torch.mean(model.get_input_embeddings()(tokenizer.encode(s_target_string, return_tensors="pt")), 1, keepdim=True), requires_grad=True)
    gd = torch.optim.RMSprop([input_embedding],
                             lr=1, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False)

    print(f"Dimensions of input embedding: {input_embedding.shape}")
    target_sequence = tokenizer.encode(s_target_string, return_tensors="pt")
    print("TARGET IN TOKENS")

    fn_generate_full = lambda x: model.generate_with_grad(
            inputs_embeds=x,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=i_max_new_tokens,
            num_beams=i_num_beams,
            num_return_sequences=i_num_return,
            return_dict_in_generate=True,
            output_scores=True)
    fn_generate_one = lambda x: model.generate_with_grad(
            inputs_embeds=x,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=1,
            num_beams=i_num_beams,
            num_return_sequences=i_num_return,
            return_dict_in_generate=True,
            output_scores=True)
    fn_generate_i = lambda x, i: model.generate_with_grad(
            inputs_embeds=x,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=i,
            num_beams=i_num_beams,
            num_return_sequences=i_num_return,
            return_dict_in_generate=True,
            output_scores=True)
    
    input_embedding_start = torch.clone(input_embedding) # save it
    output_start = fn_generate_full(input_embedding_start)

    print("START\n")
    print(input_embedding_start.shape)
    print(tokenizer.decode(output_start.sequences[0]))
    print(output_start.sequences[0])

    ll = llikelihood(output_start, target_sequence)
    print(ll)
    
    # input_embedding_3 = autograd.Variable(torch.mean(model.get_input_embeddings()(tokenizer.encode(s_target_string, return_tensors="pt")), 1, keepdim=True), requires_grad=True)
    # gd = torch.optim.RMSprop([input_embedding],
    #                         lr=1e-1, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False)

    # start_time = time.time()
    # summary_embedding_3 = training_parallel(target_sequence, input_embedding_3, tokenizer, gd, fn_generate_one, fn_generate_full, model, threading=True)
    # end_time = time.time()
    # execution_time_3 = end_time - start_time

    input_embedding1 = autograd.Variable(torch.mean(model.get_input_embeddings()(tokenizer.encode(s_target_string, return_tensors="pt")), 1, keepdim=True), requires_grad=True)
    gd = torch.optim.RMSprop([input_embedding1],
                            lr=1e-2, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False)

    start_time = time.time()
    summary_embedding_1 = training_joint(target_sequence, input_embedding1, tokenizer, gd, fn_generate_full)
    end_time = time.time()
    execution_time_1 = end_time - start_time

    input_embedding4 = autograd.Variable(torch.mean(model.get_input_embeddings()(tokenizer.encode(s_target_string, return_tensors="pt")), 1, keepdim=True), requires_grad=True)
    gd = torch.optim.RMSprop([input_embedding4],
                            lr=1e-2, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False)
    
    start_time = time.time()
    summary_embedding_4, history_gd = training_fwinduction(target_sequence, input_embedding4, tokenizer, gd, fn_generate_i, fn_generate_full, model)
    end_time = time.time()
    execution_time_4 = end_time - start_time
    #plt.plot(history_gd)
    #plt.show()
    
    input_embedding2 = autograd.Variable(torch.mean(model.get_input_embeddings()(tokenizer.encode(s_target_string, return_tensors="pt")), 1, keepdim=True), requires_grad=True)
    gd = torch.optim.RMSprop([input_embedding2],
                            lr=1e-2, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False)

    start_time = time.time()
    summary_embedding_2 = training_parallel(target_sequence, input_embedding2, tokenizer, gd, fn_generate_one, fn_generate_full, model)
    end_time = time.time()
    execution_time_2 = end_time - start_time

    # 72 iterations, 0.84, 120s -> gpt2, training_joint
    # 30 iterations, 0.56, 140s -> gpt2, training_parallel (no threading)
    # 30 iterations, 0.56, 100s -> gpt2, training_parallel (threading)
    
    print(f"Execution time training_joint: {execution_time_1} seconds")
    print(f"Execution time training_parallel (no threading): {execution_time_2} seconds")
    # print(f"Execution time training_parallel (threading): {execution_time_3} seconds")
    print(f"Execution time training_fw: {execution_time_4} seconds")
    
    print(tokenizer.decode(fn_generate_full(summary_embedding_1).sequences[0]))
    print(tokenizer.decode(fn_generate_full(summary_embedding_2).sequences[0]))
    # print(tokenizer.decode(fn_generate_full(summary_embedding_3).sequences[0]))
    print(tokenizer.decode(fn_generate_full(summary_embedding_4).sequences[0]))
    
    # print(torch.isclose(summary_embedding_1, summary_embedding_2, rtol=1e-1))
    # print(torch.isclose(summary_embedding_1, summary_embedding_3, rtol=1e-1))
    # print(torch.isclose(summary_embedding_2, summary_embedding_3, rtol=1e-1))
    print(torch.isclose(summary_embedding_1, summary_embedding_4, rtol=1e-1))
    # print(torch.isclose(summary_embedding_2, summary_embedding_4, rtol=1e-1))
    # print(torch.isclose(summary_embedding_3, summary_embedding_4, rtol=1e-1))
    
    # print(torch.nn.functional.cosine_similarity(summary_embedding_1, summary_embedding_2, dim=2))
    # print(torch.nn.functional.cosine_similarity(summary_embedding_1, summary_embedding_3, dim=2))
    # print(torch.nn.functional.cosine_similarity(summary_embedding_2, summary_embedding_3, dim=2))