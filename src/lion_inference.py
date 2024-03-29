import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import argparse
from tqdm import tqdm
import json
import os
import ray


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    
    
def run_eval(model_dir, data_dir, output_dir, num_gpus, load_in_8bit):
    # split question file into num_gpus files
    with open(data_dir, 'r') as fcc_file:
        ques_jsons = json.load(fcc_file)

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
        prompt = generate_prompt(instruction)
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
                "output": output,
            }
        )
    return ans_jsons
           

def generate_prompt(instruction):
    return f"""### Instruction:
{instruction}

### Response:
"""
    
    
def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_dir', type=str, help='')
    parser.add_argument('--data_dir', type=str, help='')
    parser.add_argument('--output_dir', type=str, help='')
    parser.add_argument('--num_gpus', type=int, default=8, help='')
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
