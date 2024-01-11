# SETUP --------------------------
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import pandas as pd
import numpy as np

# MAGICKS --------------------------
s_model = 'gpt2'
s_string = "Finn writes code"

# FUNCTIONS --------------------------
def calc_joint_prob(transition_scores: torch.Tensor) -> float:
    # calculate the joint probability of a generated text
    return np.exp(np.sum(transition_scores.numpy()))

def calc_trans_prob(transition_scores: torch.Tensor) -> float:
    return np.exp(transition_scores.numpy())

def table_trans_prob(generated_tokens: torch.Tensor , transition_scores: torch.Tensor) -> pd.DataFrame : 
    # for beam search, loops over all generated texts
    l_df = list()
    for i in range(len(generated_tokens)):
        # create the dataframe and append to list
        l_df.append(pd.DataFrame({'token': generated_tokens[i].numpy(),
                        'trans_scores': transition_scores[i].numpy()}))
        # create other columns
        l_df[i]['token_str'] = l_df[i]['token'].apply(lambda x: tokenizer.decode(x))
        l_df[i]['trans_prob'] = l_df[i]['trans_scores'].apply(lambda x: np.exp(x))
        l_df[i]['generation'] = i+1
        l_df[i]['joint'] = calc_joint_prob(transition_scores[i])
    
    return pd.concat(l_df)

def passthrough(input_embedding, fn_model, model):
    outputs = fn_model(input_embedding)
    
    try:
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
        )
    except AttributeError:
        #print("\nNo beam search!\n")
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
    
    return calc_trans_prob(transition_scores[0]), outputs
        
# MAIN --------------------------
if __name__ == "__main__":
    

    # MAGICKS
    i_seed_value = 42
    
    s_string = "Chicken and the"
    s_model = 'gpt2'
    
    i_num_beams = 1
    i_new_tokens = 5
    
    # BUG: Does this seed work?
    torch.manual_seed(i_seed_value)
    model = GPT2LMHeadModel.from_pretrained(s_model)
    tokenizer = AutoTokenizer.from_pretrained(s_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # TEST: create test case
    s_string = "Finn writes code"
    b_string = True
    #input_embedding = torch.randn(1, 5, 768)
    input_embedding = model.get_input_embeddings()(tokenizer.encode(s_string, return_tensors="pt"))
    
    # NOTE: torch.Size([1, 5, 768])
    print(input_embedding.shape)
    print(input_embedding)
    
    # NOTE: num_beams=1 is greedy-search
    fn_model = lambda x: model.generate(
        inputs_embeds=x,
        max_new_tokens=i_new_tokens,
        num_beams=i_num_beams,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
    )

    v_trans_prob_all, outputs = passthrough(input_embedding, fn_model, model)
    s_string_test = tokenizer.decode(outputs.sequences[0][1:])
    
    l_trans_prob = list()
    s_string_copy = s_string
    
    for i in range(i_new_tokens):
        
        if b_string:
            input_embedding = model.get_input_embeddings()(tokenizer.encode(s_string, return_tensors="pt"))
        
        fn_model = lambda x: model.generate(
            inputs_embeds=x,
            max_new_tokens=1,
            num_beams=i_num_beams,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

        v_trans_prob, outputs = passthrough(input_embedding, fn_model, model)
        l_trans_prob.append(v_trans_prob)
        
        s_string += tokenizer.decode(outputs.sequences[0][1])
        print(s_string)
        b_string = True
        
    print("\n\n")
    print(s_string_copy)
    print("\n")
    print(f"Iter: {s_string[len(s_string_copy):]}")
    print(f"Full: {s_string_test}")

    print(f"Iter: {np.asarray(l_trans_prob).flatten()}")
    print(f"Full: {v_trans_prob_all}")
    
    if i_num_beams == 1:
        assert s_string[len(s_string_copy):] == s_string_test, "TEXT: Iterative not the same as gen. all at once for greedy search"
        assert np.allclose(np.asarray(l_trans_prob).flatten(), v_trans_prob_all, rtol=0.001), "PROB: Transition probs not the same"
    