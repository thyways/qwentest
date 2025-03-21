import torch
from model.modeling_qwen2_vl_new import Qwen2VLForConditionalGeneration
from transformers import AutoProcessor
from sampling.utils import norm_logits,sample
from qwen_vl_utils import process_vision_info
from model.cache import FlashSimpleCache
processor = AutoProcessor.from_pretrained('/data1/bks/wumengke/model_weight/Qwen2-VL-2B-Instruct')
draft_model = Qwen2VLForConditionalGeneration.from_pretrained('/data1/bks/wumengke/model_weight/Qwen2-VL-2B-Instruct',torch_dtype = torch.float16,device_map="cuda:3")
#target_model = Qwen2VLForConditionalGeneration.from_pretrained('/data1/bks/wumengke/model_weight/Qwen2-VL-7B-Instruct',torch_dtype = torch.bfloat16,device_map="cuda:0")
messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type":"image","image":"/data1/bks/liurunze/qwentest/image/dog.jpg"},
                                    {"type": "text", "text": "Describe the image"},
                                ],
                            }
                        ]


text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
inputs = inputs.to(draft_model.device)
cache = FlashSimpleCache(draft_model,1024+40)
def autoregressive_sampling(inputs,model,cache,max_len):
    input_ids = inputs.input_ids.to(model.device)
    pixel_values = inputs.pixel_values.to(model.device)
    image_grid_thw = inputs.image_grid_thw.to(model.device)
    outputs = model(input_ids=input_ids,pixel_values = pixel_values,image_grid_thw=image_grid_thw,kv_cache=cache)
    logits = outputs.logits
    #torch.save({"cache":outputs.past_key_values},"/data1/bks/liurunze/qwentest/utils/qkv_save.pt")
    next_token_id = torch.argmax(logits[:,-1,:],dim=-1)
    if next_token_id.shape==torch.Size([1]):
        next_token_id = next_token_id.unsqueeze(0)
    generated_ids = []
    generated_ids.append(next_token_id.item())
    for _ in range(max_len-1):
        outputs = model(input_ids = next_token_id,kv_cache=cache)
        logits = outputs.logits
        next_token_id = sample(norm_logits(logits[:,-1,:],temperature=0.01,top_k=-1,top_p=0.9))
        if next_token_id.shape==torch.Size([1]):
            next_token_id = next_token_id.unsqueeze(0)
        generated_ids.append(next_token_id.item())
        print(next_token_id.item())
    print(generated_ids)
    generated_text = processor.decode(generated_ids,skip_special_tokens=True)
    return generated_text
output = autoregressive_sampling(inputs=inputs,model=draft_model,cache=cache,max_len=40)
print(output)
    