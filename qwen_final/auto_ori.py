import torch
from model.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from transformers import AutoProcessor
import torch.nn.functional as F
from sampling.utils import norm_logits,sample
from qwen_vl_utils import process_vision_info
import time
from model.cache import FlashSimpleCache
processor = AutoProcessor.from_pretrained('/data1/bks/wumengke/model_weight/Qwen2-VL-2B-Instruct')
model = Qwen2VLForConditionalGeneration.from_pretrained('/data1/bks/wumengke/model_weight/Qwen2-VL-2B-Instruct',torch_dtype = torch.float16,device_map="cuda:3")

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
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
inputs = inputs.to(model.device)
outputs = model(input_ids=inputs.input_ids[:,:-1],pixel_values=inputs.pixel_values,image_grid_thw=inputs.image_grid_thw)
torch.save({"logits":outputs.logits},"/data1/bks/liurunze/qwen_final/utils/qkv_save.pt")
generated_ids = model.generate(**inputs,max_new_tokens=128)
print(generated_ids)
generated_ids_trimmed = [out_ids[len(in_ids):]for in_ids,out_ids in zip(inputs.input_ids,generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)
print(output_text)
# def generate_text_with_image(model, processor, inputs, max_new_tokens=30):
#     # 首次调用模型
#     outputs = model(input_ids=inputs.input_ids, use_cache=True)
#     logits = outputs.logits
#     past_key_values = outputs.past_key_values

#     generated_ids = []
#     eos_token_id = processor.tokenizer.eos_token_id

#     # 采样第一个 token
#     next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
#     next_token_id = next_token_id.to(model.device)
#     generated_ids.append(next_token_id.item())

#     # 自回归生成
#     for _ in range(max_new_tokens - 1):
#         outputs = model(input_ids=next_token_id, past_key_values=past_key_values)
#         logits = outputs.logits
#         past_key_values = outputs.past_key_values

#         # 采样下一个 token
#         next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
#         next_token_id = next_token_id.to(model.device)
#         generated_ids.append(next_token_id.item())

#         # 检测终止符
#         if next_token_id.item() == eos_token_id:
#             break

#     generated_text = processor.decode(generated_ids, skip_special_tokens=True)
#     return generated_text
# output = generate_text_with_image(draft_model,processor=processor,inputs=inputs,max_new_tokens=30)
# print(output)
    
    

    