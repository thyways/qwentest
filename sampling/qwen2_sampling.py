import torch
from transformers import TextStreamer

def qwen2_sampling(processor,model,input_tokens,max_new_tokens):
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    generated_ids = model.generate(**input_tokens, max_new_tokens=max_new_tokens,streamer=streamer,)
    return generated_ids