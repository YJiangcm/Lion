import json
import pandas as pd
import argparse

from utils import get_json_list, get_json_list2
    

def discrinimation(review12_path, review21_path, chatgpt_inference_path, lion_inference_path, hard_save_path, easy_save_path):
    review12 = get_json_list2(review12_path)
    review21 = get_json_list2(review21_path)
    assistant1 = get_json_list(chatgpt_inference_path)
    assistant2 = get_json_list(lion_inference_path)

    ques = [i['instruction'] for i in assistant1]
    input = [i['input'] for i in assistant1]
    ans1 = [i['output'] for i in assistant1]
    ans2 = [i['output'] for i in assistant2]

    ans1_score12 = [i['score'][0] for i in review12]
    ans2_score12 = [i['score'][1] for i in review12]
    review_text12 = [i['text'] for i in review12]

    ans1_score21 = [i['score'][1] for i in review21]
    ans2_score21 = [i['score'][0] for i in review21]
    review_text21 = [i['text'] for i in review21]

    ans1_score = [(i['score'][0] + j['score'][1])/2 for i,j in zip(review12, review21)]
    ans2_score = [(i['score'][1] + j['score'][0])/2 for i,j in zip(review12, review21)]

    review_score_diff = [(i-j) for i,j in zip(ans1_score, ans2_score)]

    referee = pd.DataFrame({'instruction': ques,
                        'input': input,
                        'assist1': ans1,
                        'assist2': ans2,
                        'assist1_score12': ans1_score12,
                        'assist2_score12': ans2_score12,
                        'review_text12': review_text12,
                        'assist1_score21': ans1_score21,
                        'assist2_score21': ans2_score21,
                        'review_text21': review_text21,
                        'assist1_score': ans1_score,
                        'assist2_score': ans2_score,
                        'review_score_diff': review_score_diff,
                        })

    referee = referee.sort_values(by=['review_score_diff', 'assist1_score'], ascending=False)
    referee = referee.reset_index(drop=False)

    hard_instructions = referee[(referee['review_score_diff'] >= 1)]
    easy_instructions = referee[(referee['review_score_diff'] < 1)]

    print(f'Number of hard instructions: {len(hard_instructions)}, Number of easy instructions: {len(easy_instructions)}')

    # save the identified hard instructions
    hard_instructions = hard_instructions.reset_index(drop=False)
    with open(hard_save_path, 'w') as output_hard_file:
        for i in range(len(hard_instructions)):
            output_hard_file.write(json.dumps({'instruction': hard_instructions.iloc[i]['instruction'], 
                                                'input': hard_instructions.iloc[i]['input'],
                                                'assist1': hard_instructions.iloc[i]['assist1'],
                                                'assist2': hard_instructions.iloc[i]['assist2'],
                                                'assist1_score': hard_instructions.iloc[i]['assist1_score'],
                                                'assist2_score': hard_instructions.iloc[i]['assist2_score'],
                                                'review_score_diff': hard_instructions.iloc[i]['review_score_diff']}) + '\n')

    # save the identified easy instructions
    easy_instructions = easy_instructions.reset_index(drop=False)
    with open(easy_save_path, 'w') as output_easy_file:
        for i in range(len(easy_instructions)):
            output_easy_file.write(json.dumps({'instruction': easy_instructions.iloc[i]['instruction'], 
                                                'input': easy_instructions.iloc[i]['input'],
                                                'assist1': easy_instructions.iloc[i]['assist1'],
                                                'assist2': easy_instructions.iloc[i]['assist2'],
                                                'assist1_score': easy_instructions.iloc[i]['assist1_score'],
                                                'assist2_score': easy_instructions.iloc[i]['assist2_score'],
                                                'review_score_diff': easy_instructions.iloc[i]['review_score_diff']}) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='discriminate hard and easy instructions.')
    parser.add_argument('--review12_path', type=str)
    parser.add_argument('--review21_path', type=str)
    parser.add_argument('--chatgpt_inference_path', type=str)
    parser.add_argument('--lion_inference_path', type=str)
    parser.add_argument('--hard_save_path', type=str)
    parser.add_argument('--easy_save_path', type=str)
    args = parser.parse_args()

    discrinimation(
                    args.review12_path,
                    args.review21_path,
                    args.chatgpt_inference_path,
                    args.lion_inference_path,
                    args.hard_save_path,
                    args.easy_save_path
                   )
