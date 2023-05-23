import argparse
import json
import os
import time

import openai
from tqdm import tqdm
import ray

import logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import get_json_list, get_json_list2

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
                temperature=0.2,
                messages=[{
                    'role': 'system',
                    'content': "You are a helpful and precise assistant for checking the quality of the answer.",
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


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception('Invalid score pair.')
    except Exception as e:
        logger.error(f'{e}\nContent: {review}\n'
                     'You must manually fix the score pair.')
        return [-1, -1]


def gen_prompt(ques, input, ans1, ans2):
    
    prompt_template = "[Question]\n{question}\n\n[Input]\n{input}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n" \
        + "[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n"
    
    evaluate_prompt = "We would like to request your feedback on the performance of two AI assistants in response to the user question and input displayed above." \
        + "\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. " \
        + "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance." \
        + "\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. " \
        + "In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
    
    prompt = prompt_template.format(question=ques, input=input, answer_1=ans1, answer_2=ans2, prompt=evaluate_prompt)

    return prompt



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-referee to compare the quality of two responses.')
    parser.add_argument('-a', '--answer-file-list', nargs='+', default=[])
    parser.add_argument('-o', '--output-review-file')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--max-tokens', type=int, default=512, help='maximum number of tokens produced in the output')
    parser.add_argument('--recheck', action='store_true', default=False)
    args = parser.parse_args()

    question_jsons = get_json_list(args.answer_file_list[0])
    answer1_jsons = get_json_list(args.answer_file_list[0])
    answer2_jsons = get_json_list(args.answer_file_list[1])

    # check if # of questions, answers are the same
    assert len(answer1_jsons) == len(answer2_jsons)
    question_idx_list = list(range(len(question_jsons)))
    
    ques = [i['instruction'] for i in question_jsons]
    input = [i['input'] for i in question_jsons]
    ans1 = [i['output'] for i in answer1_jsons]
    ans2 = [i['output'] for i in answer2_jsons]
    
    # if not args.recheck:
        ### first run
    ray.init()
    handles = []
    for idx in tqdm(question_idx_list):
        user_prompt = gen_prompt(ques[idx], input[idx], ans1[idx], ans2[idx])
        
        # To avoid the rate limit set by OpenAI
        handles.append(get_eval.remote(user_prompt, args.max_tokens, args.api_key))
        logger.info(f'Waiting for {REQ_TIME_GAP} seconds before sending the next request.')
        time.sleep(REQ_TIME_GAP)
        
    reviews = ray.get(handles)
    
    assert len(question_jsons) == len(reviews)
    with open(f'{args.output_review_file}', 'w') as output_review_file:
        for idx, review in enumerate(reviews):
            scores = parse_score(review)
            output_review_file.write(json.dumps({'score': scores, 'text': review}) + '\n')
    
    ray.shutdown()
            
    # else:
        ### recheck
    old_question_idx_list = question_idx_list
    
    reviews = get_json_list2(args.output_review_file)
    new_question_idx_list = []
    for idx in range(len(reviews)):
        if reviews[idx]['score']==[-1, -1]:
            new_question_idx_list.append(idx)
            
    while len(new_question_idx_list) < len(old_question_idx_list):
        ray.init()
        handles = []
        for idx in tqdm(new_question_idx_list):
            user_prompt = gen_prompt(ques[idx], input[idx], ans1[idx], ans2[idx])
            # To avoid the rate limit set by OpenAI
            handles.append(get_eval.remote(user_prompt, args.max_tokens, args.api_key))
            logger.info(f'Waiting for {REQ_TIME_GAP} seconds before sending the next request.')
            time.sleep(REQ_TIME_GAP)
        new_reviews = ray.get(handles)
        
        assert len(new_question_idx_list) == len(new_reviews)
        for idx, review in enumerate(new_reviews):
            scores = parse_score(review)
            reviews[new_question_idx_list[idx]]['score'] = scores
            reviews[new_question_idx_list[idx]]['text'] = review
        
        with open(f'{args.output_review_file}', 'w') as output_review_file:
            for idx, review in enumerate(reviews):
                output_review_file.write(json.dumps({'score': review['score'], 'text': review['text']}) + '\n')
        
        old_question_idx_list = new_question_idx_list
        
        new_question_idx_list = []
        for idx in range(len(reviews)):
            if reviews[idx]['score']==[-1, -1]:
                new_question_idx_list.append(idx)

        ray.shutdown()
    