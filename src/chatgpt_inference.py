import argparse
import json
import time

import openai
from tqdm import tqdm
import ray

import logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import get_json_list

MAX_API_RETRY = 5
REQ_TIME_GAP = 20

@ray.remote(num_cpus=1)
def get_eval(user_prompt: str, max_tokens: int, api_key: str):
    logging.basicConfig(level=logging.INFO)
    for i in range(MAX_API_RETRY):
        try:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                max_tokens=max_tokens,
                temperature=0.7,
                messages=[{
                    'role': 'system',
                    'content': "You are a helpful assistant that generates a response to a given task instruction.",
                }, {
                    'role': 'user',
                    'content': user_prompt,
                }],
            )
            content = response['choices'][0]['message']['content']
            logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
            time.sleep(5)
    logger.error(f'Failed after {MAX_API_RETRY} retries.')
    return 'error'


def generate_prompt(instruction):
    return f"""### Instruction:
{instruction}

### Response:
"""



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question-file-list')
    parser.add_argument('-o', '--output-review-file')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    parser.add_argument('--recheck', action='store_true', default=False)
    args = parser.parse_args()

    question_jsons = get_json_list(args.question_file_list)
    question_idx_list = list(range(len(question_jsons)))
    
    ques = [i['instruction'] for i in question_jsons]
    
    # if not args.recheck:
    ### first run
    ray.init()
    handles = []
    for idx in tqdm(question_idx_list):
        user_prompt = generate_prompt(ques[idx])
        
        # To avoid the rate limit set by OpenAI
        handles.append(get_eval.remote(user_prompt, args.max_tokens, args.api_key))
        logger.info(f'Waiting for {REQ_TIME_GAP} seconds before sending the next request.')
        time.sleep(REQ_TIME_GAP)
    reviews = ray.get(handles)
    
    assert len(question_jsons) == len(reviews)
    output = [{'instruction': ques[i],
                'output': reviews[i]}
                for i in question_idx_list]
    
    json_str = json.dumps(output, indent=4)
    with open(f'{args.output_review_file}', mode='w', encoding='utf-8') as json_file:
        json_file.write(json_str)
    
    ray.shutdown()
            
    # else:
    ### recheck
    old_question_idx_list = question_idx_list
    
    reviews = get_json_list(args.output_review_file)
    new_question_idx_list = []
    for idx in range(len(reviews)):
        if reviews[idx]['output']=='error' or reviews[idx]['output']=="":
            new_question_idx_list.append(idx)
            
    while len(new_question_idx_list) < len(old_question_idx_list):
        ray.init()
        handles = []
        for idx in tqdm(new_question_idx_list):
            user_prompt = generate_prompt(ques[idx])
            # To avoid the rate limit set by OpenAI
            handles.append(get_eval.remote(user_prompt, args.max_tokens, args.api_key))
            logger.info(f'Waiting for {REQ_TIME_GAP} seconds before sending the next request.')
            time.sleep(REQ_TIME_GAP)
        new_reviews = ray.get(handles)
        
        assert len(new_question_idx_list) == len(new_reviews)
        for idx, review in enumerate(new_reviews):
            reviews[new_question_idx_list[idx]]['output'] = review
        
        json_str = json.dumps(reviews, indent=4)
        with open(f'{args.output_review_file}', mode='w', encoding='utf-8') as json_file:
            json_file.write(json_str)
        
        old_question_idx_list = new_question_idx_list
        
        new_question_idx_list = []
        for idx in range(len(reviews)):
            if reviews[idx]['output']=='error' or reviews[idx]['output']=="":
                new_question_idx_list.append(idx)

        ray.shutdown()
    
