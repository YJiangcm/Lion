import json
import pandas as pd
import argparse

from utils import get_json_list, get_json_list2
    

def discrinimation(review_path, chatgpt_inference_path, lion_inference_path, hard_save_path, easy_save_path):
    review = get_json_list2(review_path)
    assistant1 = get_json_list(chatgpt_inference_path)
    assistant2 = get_json_list(lion_inference_path)

    ques = [i['instruction'] for i in assistant1]
    input = [i['input'] for i in assistant1]
    ans1 = [i['output'] for i in assistant1]
    ans2 = [i['output'] for i in assistant2]
    ans1_score = [i['score'][0] for i in review]
    ans2_score = [i['score'][1] for i in review]
    review_text = [i['text'] for i in review]
    review_score_diff = [i['score'][0] - i['score'][1] for i in review]

    referee = pd.DataFrame({'instruction': ques,
                            'input': input,
                            'assist1': ans1,
                            'assist2': ans2,
                            'assist1_score': ans1_score,
                            'assist2_score': ans2_score,
                            'review_text': review_text,
                            'review_score_diff': review_score_diff,
                            })

    referee = referee.sort_values(by=['review_score_diff', 'assist1_score'], ascending=False)
    referee = referee.reset_index(drop=False)

    hard_instructions = referee[(referee['review_score_diff'] >= 2) & (referee['assist1_score'] >= 7)]
    easy_instructions = referee[(referee['review_score_diff'] < 2) | (referee['assist1_score'] < 7)]

    print(len(hard_instructions), len(easy_instructions))

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
                                                'review_text': hard_instructions.iloc[i]['review_text'],
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
                                                'review_text': easy_instructions.iloc[i]['review_text'],
                                                'review_score_diff': easy_instructions.iloc[i]['review_score_diff']}) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='discriminate hard and easy instructions.')
    parser.add_argument('--review_path', type=str)
    parser.add_argument('--chatgpt_inference_path', type=str)
    parser.add_argument('--lion_inference_path', type=str)
    parser.add_argument('--hard_save_path', type=str)
    parser.add_argument('--easy_save_path', type=str)
    args = parser.parse_args()

    discrinimation(
                    args.review_path,
                    args.chatgpt_inference_path,
                    args.lion_inference_path,
                    args.hard_save_path,
                    args.easy_save_path
                   )