import time
import json
import os
import random
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils
from utils import get_json_list

import fire
import openai


def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("prompt_hard.txt").read() + "\n"
    inst_i_o = []
    for _, task_dict in enumerate(prompt_instructions):
        inst_i_o.append({"instruction": task_dict["instruction"], "output": task_dict["output"]})
        instruction = task_dict["instruction"]
        prompt += f"{instruction}\n"
    prompt += "\n#Created Instruction#:"
    return prompt, inst_i_o


def post_process_gpt3_response(response, inst_i_o):
    if response is None:
        return []
    
    instruction = response["message"]['content']

    if 'instruction' in instruction.lower():
        return []
    # if the decoding stops due to length, the last example is likely truncated so we discard it
    if response["finish_reason"] == "length":
        return []
    # filter out too short instructions
    if len(instruction.split()) <= 3:
        return []
    # filter those starting with punctuation
    if instruction[0] in string.punctuation:
        return []
    # filter those starting with non-english character
    if not instruction[0].isascii():
        return []
        
    instructions = [{
                    "seed_instruction": inst_i_o["instruction"],
                    "seed_output": inst_i_o["output"],
                    "instruction": instruction,
                    }]
    return instructions


def generate_instruction_following_data(
    output_dir=None,
    seed_tasks_path=None,
    all_tasks_path=None,
    num_instructions_to_generate=100,
    model_name="gpt-3.5-turbo",
    num_prompt_instructions=1,
    request_batch_size=1,
    temperature=1.0,
    top_p=1.0,
    num_cpus=3,
    api_key=None,
):
    openai.api_key = api_key

    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "output": t["assist1"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} seed instructions")


    all_instruction_data = get_json_list(all_tasks_path)
    print(f"Loaded {len(all_instruction_data)} all instructions")


    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "hard_regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "hard_regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in all_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        inst_i_os = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt, inst_i_o = encode_prompt(prompt_instructions)
            batch_inputs.append(prompt)
            inst_i_os.extend(inst_i_o)
        
        max_tokens = len(inst_i_os[0]['instruction'].split()) * 4 ####### automatically adjusted the max length
        
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=max_tokens,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
        )
        request_start = time.time()
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        
        assert len(results) == len(inst_i_os)
        for idx, result in enumerate(results):
            new_instructions = post_process_gpt3_response(result, inst_i_os[idx])
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
        
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "hard_regen.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
