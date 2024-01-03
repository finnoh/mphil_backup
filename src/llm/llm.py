# SETUP --------------------------
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import pandas as pd
import numpy as np

# MAGICKS --------------------------
s_model = 'gpt2'
s_string = "Finn writes code"

# FUNCTIONS --------------------------
class LLM:
    def __init__(self, model, tokenizer):
        # store tokenizer and model
        self.tokenizer = tokenizer
        self.model = model
        # standard settings for generating text
        self.max_new_tokens=5
        self.do_sample=True
        self.top_k=1
        self.num_return_sequences=1
        
        # tests
        self.test_roundtrip_tokenizer("This is a test string.")
        
    def decode(self, t_tokens):
        return self.tokenizer.decode(t_tokens)
    
    def encode(self, s_string):
        return self.tokenizer.encode(s_string, return_tensors="pt")
    
    def get_embedding(self, s_string):
        # get the input embedding of a string
        return self.model.get_input_embeddings()(self.encode(s_string))
        
    def generate_embedding2text(self, embedding):
        # generate text
        outputs = self.model.generate(inputs_embeds=embedding,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=self.do_sample,
                        top_k=self.top_k,
                        return_dict_in_generate=True,
                        num_return_sequences=self.num_return_sequences,
                        output_scores=True)
        # calculate the transition scores
        transition_scores = self.model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        normalize_logits=True
        )
        
        return outputs, transition_scores
    
    def analyze_generation(self, inputs, outputs, transition_scores):
        # TODO: Implement inputs and outputs
        # read out tokens
        generated_tokens = outputs.sequences[:, inputs.input_ids.shape[1]:]
        
        # Iterate over the generated tokens and transition scores
        df = pd.DataFrame({'token': generated_tokens[0].numpy(),
                           'trans_scores': transition_scores[0].numpy()})
        # create other columns
        df['token_str'] = df['token'].apply(lambda x: self.tokenizer.decode(x))
        df['trans_prob'] = df['trans_scores'].apply(lambda x: np.exp(x))
        
        return df[['token', 'token_str', 'trans_scores', 'trans_prob']]
    
    # TODO: Implement as actual test
    def test_roundtrip_tokenizer(self, s_string):
        t_embedding = self.encode(s_string)
        s_redecoded_string = self.decode(t_embedding[0])
        assert s_string == s_redecoded_string, "Check the tokenizer; Roundtrip test failed"
        
# MAIN --------------------------
if __name__ == "__main__":
    llm = LLM(model=GPT2LMHeadModel.from_pretrained(s_model),
              tokenizer=AutoTokenizer.from_pretrained(s_model))
    
    # TODO: Set up a function that gives the generated text and losses for an input embedding
    # TODO: Next step is to set up a tiny neural network, where we implement the training framework
    embedding = llm.get_embedding(s_string=s_string)
    outputs, transition_scores = llm.generate_embedding2text(embedding=embedding)

    print(outputs)
    print(llm.decode(outputs.sequences[0]))
    print("Transition\n")
    print(transition_scores)