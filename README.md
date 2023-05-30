# Lion: Adversarial Distillation of Closed-Source Large Language Model

<p align="center" width="100%">
  <a ><img src="pics/Lion.jpg" alt="Lion" style="width: 20%; min-width: 200px; display: block; margin: auto;"></a>
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2305.12870">[📄 Paper]</a> | 
  <a href="https://huggingface.co/YuxinJiang/Lion">[🤗 Lion Weights]</a> |
  <a href="https://60318613c90a7dbbd5.gradio.live/">[:desktop_computer: Demo]</a>
</p>
<hr>


[![Code License](https://img.shields.io/badge/code%20license-MIT-green)](https://github.com/YJiangcm/Lion/blob/master/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
[![Weight Diff License](https://img.shields.io/badge/Weight%20Diff%20License-CC%20By%20NC%204.0-yellow)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/WEIGHT_DIFF_LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

<!-- The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.  -->
<!-- The weight diff is also CC BY NC 4.0 (allowing only non-commercial use). -->

### Tuned on 70k instruction-following data, Lion (7B) can achieve 95% capability of ChatGPT!


## News
- **[May 26, 2023]** We released the model weights. Check out the [7B](https://huggingface.co/YuxinJiang/Lion) model!
- **[May 25, 2023]** We released an [online demo](https://60318613c90a7dbbd5.gradio.live/), try our model here!
- **[May 23, 2023]** We released the code for training and inference.

<!-- :pray: Since our team members are perparing for the PhD Qualifying Exam, we apologize for any possible delay in responding to your questions. We warmly welcome all inquiries and appreciate your constructive feedback :) -->

## Contents 

1. [Overview](#overview) 

2. [Online Demo](#online-demo) 

3. [Recovering Lion weights](#recovering-lion-weights) 

4. [Inference](#inference)

5. [Training Process](#training-process) 

6. [Evaluation](#evaluation)

7. [Citation](#citation)

8. [Disclaimer](#disclaimer)


## Overview
<p align="center">
  <img width="700" height="320" src="https://github.com/YJiangcm/Lion/blob/master/pics/overview.jpg">
</p>

The high-level overview of our adversarial distillation framework, where we craft a compact Student LLM based on a superior closed-source LLM that serves three roles: the **Teacher**, the **Referee**, and the **Generator**. From left to right, there are three stages in an iteration:  
1) an _imitation_ stage to align the student’s response with the teacher’s response;  
2) a _discrimination_ stage to identify hard samples;  
3) a _generation_ stage to produce new hard samples for escalating the challenges presented to the student model.


## Online Demo
We will provide our latest models for you to try for as long as possible. You may ask some questions to Lion and we are happy to hear your feedback!

[**Demo Link**](https://60318613c90a7dbbd5.gradio.live/) (the UI interface is shown below)

<p align="center">
  <img width="800" height="350" src="https://github.com/YJiangcm/Lion/blob/master/pics/english_case2.png">
</p>

Since the training data are English instruction-following examples, You'd better ask questions in English. However, we found Lion can also understand instructions in other languages to some extent. See the following case:

<p align="center">
  <img width="800" height="350" src="https://github.com/YJiangcm/Lion/blob/master/pics/chinese_case.png">
</p>


## Recovering Lion weights
We release Lion weights as delta weights to comply with the LLaMA model license.

- [Lion-7B (delta weights)](https://huggingface.co/YuxinJiang/Lion)

You can add our delta to the original LLaMA weights to obtain the Lion weights. Instructions:
1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama)
2. Please download our delta model from [Hugging Face](https://huggingface.co/YuxinJiang/Lion)  
3. Use the following scripts to get Lion weights by applying our delta:
```bash
python src/weight_diff.py recover --path_raw huggyllama/llama-7b --path_diff YuxinJiang/Lion --path_tuned <path_to_store_recovered_weights>
```

## Inference
For inference and training of Lion, please first install the requirements:
```bash
pip install -r requirements.txt
```

We provide the decoding script for Lion, which reads a input file and generates corresponding responses for each sample, and finally consolidates them into an output file. **It can be run on a single machine with 16GB GPU.**
```bash
python src/lion_inference.py \
    --model_dir <path_to_hf_converted_lion_ckpt_and_tokenizer> \
    --data_dir <path_to_input_json_file> \
    --output_dir <path_to_output_json_file> \
    --num_gpus 1
```


## Training Process
Below shows one iteration of our adversarial distillation framework.
### 1. Imitation Stage
#### 1.1 Acquire the teacher's response on the Train Pool

```bash
python src/chatgpt_inference.py \
    -q <path_to_json_file_for_the_Train_Pool> \
    -o <path_to_chatgpt_inference_for_the_Train_Pool> \
    --api_key <your_openai_api_key>
```

#### 1.2 Instruction-tuning the student based on the teacher’s response on the Train Pool

Fine-tuning was conducted on on a machine with 8 A100 80G GPUs.

```bash
torchrun --nproc_per_node=8 --master_port=<your_random_port> src/train.py \
    --model_name_or_path <path_to_hf_converted_ckpt_and_tokenizer> \
    --data_path <path_to_chatgpt_inference_for_the_Train_Pool> \
    --bf16 True \
    --output_dir result \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
```

### 2. Discrimination Stage
#### 2.1 Acquire the teacher's response on the Cache Pool

```bash
python src/chatgpt_inference.py \
    -q <path_to_json_file_for_the_Cache_Pool> \
    -o <path_to_chatgpt_inference_for_the_Cache_Pool> \
    --api_key <your_openai_api_key>
```

#### 2.2 Acquire the student's response on the Cache Pool

```bash
python src/lion_inference.py \
    --model_dir <path_to_hf_converted_lion_ckpt_and_tokenizer> \
    --data_dir <path_to_json_file_for_the_Cache_Pool> \
    --output_dir <path_to_lion_inference_for_the_Cache_Pool> \
    --num_gpus 8
```

#### 2.3 Ask the referee to output two scores according to the respose quality of the teacher and the student

```bash
python src/chatgpt_referee.py \
    -a <path_to_chatgpt_inference_for_the_Cache_Pool> <path_to_lion_inference_for_the_Cache_Pool> \
    -o <path_to_output_review_file> \
    --api_key <your_openai_api_key>
```

#### 2.4 Discriminate hard instructions and easy instructions

```bash
python src/discrimination.py \
    --review_path <path_to_output_review_file> \
    --chatgpt_inference_path <path_to_chatgpt_inference_for_the_Cache_Pool> \
    --lion_inference_path path_to_lion_inference_for_the_Cache_Pool \
    --hard_save_path <path_to_identified_hard_instructions> \
    --easy_save_path <path_to_identified_easy_instructions>
```

### 3. Generation Stage
Fill the `openai.api_key = "<you_openai_api_key>"` in [src/utils.py](https://github.com/YJiangcm/Lion/blob/master/src/utils.py).
#### 3.1 Generate new hard instructions

```bash
python -m src/generate_hard_instruction generate_instruction_following_data \
    --seed_tasks_path <path_to_identified_hard_instructions> \
    --output_dir <path_to_generated_hard_instructions> \
    --num_instructions_to_generate 3000
```
#### 3.2 Generate new easy instructions
```bash
python -m src/generate_easy_instruction generate_instruction_following_data \
    --seed_tasks_path <path_to_identified_easy_instructions> \
    --output_dir <path_to_generated_easy_instructions> \
    --num_instructions_to_generate 3000
```

## Evaluation

### Automatic Evaluation with GPT-4
we leverage GPT-4 to automatically rate the response quality (with scores from 1 to 10) between two models on 80 unseen [Vicuna-Instructions](https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/table/question.jsonl).
ChatGPT has been chosen as the reference model to estimate the relative capability of diverse LLMs against it. The relative score is reported in percentage, computed as the ratio of the sum of scores.

**Relative Overall Response Quality**:

<p align="center">
  <img width="500" height="250" src="https://github.com/YJiangcm/Lion/blob/master/pics/relative_quality_overall.jpg">
</p>

**Relative Response Quality of Diverse Task Categories**:

<p align="center">
  <img width="700" height="330" src="https://github.com/YJiangcm/Lion/blob/master/pics/relative_quality_category.jpg">
</p>

### Human Evaluation with Alignment Criteria
We employ the alignment criteria proposed by Askell et al. (2021), which define that an assistant is considered aligned if it is characterized by being helpful, honest, and
harmless (HHH). We performed a human evaluation on 252 [UserOriented-Instructions](https://github.com/yizhongw/self-instruct/blob/main/human_eval/user_oriented_instructions.jsonl). To estimate the won rate, we compare the frequency of won, tie, and lost between each pair
of models below.

<p align="center">
  <img width="500" height="300" src="https://github.com/YJiangcm/Lion/blob/master/pics/252task_win.jpg">
</p>


## Citation
Please cite our paper if you use the code in this repo.

```
@article{DBLP:journals/corr/abs-2305-12870,
  author       = {Yuxin Jiang and
                  Chunkit Chan and
                  Mingyang Chen and
                  Wei Wang},
  title        = {Lion: Adversarial Distillation of Closed-Source Large Language Model},
  journal      = {CoRR},
  volume       = {abs/2305.12870},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.12870},
  doi          = {10.48550/arXiv.2305.12870},
  eprinttype    = {arXiv},
  eprint       = {2305.12870},
  timestamp    = {Fri, 26 May 2023 11:29:33 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2305-12870.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```




## Disclaimer
⚠️ Lion is intended and licensed for **research use ONLY**. Commercial use is **strictly prohibited**.
The content produced by any version of Lion is influenced by uncontrollable variables such as randomness, and therefore, the accuracy of the output cannot be guaranteed by this project. 
This project does not accept any legal liability for the content of the model output, nor does it assume responsibility for any losses incurred due to the use of associated resources and output results.
