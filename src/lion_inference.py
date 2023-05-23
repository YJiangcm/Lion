import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import argparse
from tqdm import tqdm
import json
import os
import ray

from utils import get_json_list

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    
    
def run_eval(model_dir, data_dir, output_dir, num_gpus, load_in_8bit):
    # split question file into num_gpus files
    ques_jsons = get_json_list(data_dir)

    chunk_size = len(ques_jsons) // num_gpus
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        ans_handles.append(
            get_model_answers.remote(
                model_dir, ques_jsons[i : i + chunk_size], load_in_8bit
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    json_str = json.dumps(ans_jsons, indent=4)
    with open(output_dir, mode='w', encoding='utf-8') as json_file:
        json_file.write(json_str)
            

@ray.remote(num_gpus=1)
@torch.inference_mode()
def get_model_answers(model_dir, question_jsons, load_in_8bit):
    disable_torch_init()
    model_path = os.path.expanduser(model_dir)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_dir,
        load_in_8bit=load_in_8bit, # True may save memory (16GB to 10GB), but slower
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if not load_in_8bit:
        model.half()  # seems to fix bugs for some users.

    ans_jsons = []
    for _, line in enumerate(tqdm(question_jsons)):
        instruction = line['instruction']
        ipt = line['input']
        prompt = generate_prompt(instruction, ipt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda() # these are integers encoded from words
        generation_config = GenerationConfig(
            temperature=0.7,
            num_beams=1,
            do_sample=True,
        )
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=1024,
        )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True) # this will return a fully-wholely description like "Below is an instruction....Response:..."
        output = output.split("### Response:")[1].strip()
        ans_jsons.append(
            {
                "instruction": instruction,
                "input": ipt,
                "output": output,
            }
        )
    return ans_jsons
           

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""
    
    
def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_dir', type=str, help='path_to_hf_converted_lion_ckpt_and_tokenizer')
    parser.add_argument('--data_dir', type=str, help='')
    parser.add_argument('--output_dir', type=str, help='')
    parser.add_argument('--num_gpus', type=int, default=8, help='number of gpus to use')
    parser.add_argument('--load_in_8bit', action='store_true', help='')

    args = parser.parse_args()

    ray.init()
    run_eval(
        args.model_dir,
        args.data_dir,
        args.output_dir,
        args.num_gpus,
        args.load_in_8bit,
    )
                    
if __name__ == "__main__":
    main()



# An example of the data format of 'data_dir' is as follows:
'''
[
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
    },
    {
        "instruction": "What are the three primary colors?",
        "input": "",
    },
    {
        "instruction": "Explain why the following fraction is equivalent to 1/4",
        "input": "4/16",
    }
]
'''