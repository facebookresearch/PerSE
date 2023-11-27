import click
import json
import torch
import argparse
import os
from tqdm.auto import tqdm
from typing import Optional, Dict, Sequence
import transformers

from model import PerSE
from utils.tools import clean_plot, clean_review, convert_messages_to_oasst_format, convert_messages_to_llama2_format
from utils.prompt_template import (
    SCORE_SYSTEM_PROMPT,
    SCORE_PROBLEM_PROMPT_TEMPLATE,
    SCORE_CASE_TEMPLATE,
    RANK_SYSTEM_PROMPT,
    RANK_PROBLEM_PROMPT_TEMPLATE,
    RANK_CASE_TEMPLATE,
    RANK_COMPLETION
)


MAX_SOURCE_LENGTH = 4096
MAX_TARGET_LENGTH = 512
print("Max source length: ", MAX_SOURCE_LENGTH)
print("MAX target length: ", MAX_TARGET_LENGTH)

model_server_message_end = '</s>'


def get_input():
    mode = input("Choose your mode (score / rank / quit): ")
    while mode not in ["score", "rank", "quit"]:
        mode = input("Rechoose your mode (score / rank / quit): ")
    if mode == "score":
        n = input("Please input the number of reviews you have to infer your preference (1-5): ")
        while not n.isnumeric() or eval(n) < 1 or eval(n) > 5:
            n = input("Please reinput the number of reviews you have to infer your preference (1-5): ")
        preference = []
        n = eval(n)
        for i in range(n):
            plot = input(f"[Preference {i}] Please input the plot: ")
            review = input(f"[Preference {i}] Please input your review: ")
            score = input(f"[Preference {i}] Please input your score: ")
            ex = {
                'plot': plot,
                'review': review,
                'score': score
            }
            preference.append(ex)
        plot = input(f"[Predict] Please input the new plot to be predicted: ")
        return mode, preference, plot
    elif mode == 'rank':
        preference = []
        plot1 = input(f"[Preference] Please input the plot A: ")
        plot2 = input(f"[Preference] Please input the plot B: ")
        choice = input(f"[Preference] select the preferred plot (plot A / plot B): ")
        ex = {
            'plot1': plot1,
            'plot2': plot2,
            'choice': choice
        }
        preference.append(ex)
        plot1 = input(f"[Predict] Please input the new plot A to be predicted: ")
        plot2 = input(f"[Predict] Please input the new plot B to be predicted: ")
        return mode, preference, [plot1, plot2]
    else:
        return [], [], []



def check_input_format(mode, preference, plot):
    assert isinstance(preference, list)
    if mode == 'score':
        for x in preference:
            assert 'plot' in x.keys() and isinstance(x['plot'], str)
            assert 'review' in x.keys() and isinstance(x['review'], str)
            assert 'score' in x.keys() and isinstance(x['score'], str) and x['score'].isnumeric()
        assert isinstance(plot, str)
    elif mode == 'rank':
        x = preference[0]
        assert 'plot1' in x.keys() and isinstance(x['plot1'], str)
        assert 'plot2' in x.keys() and isinstance(x['plot2'], str)
        assert 'choice' in x.keys() and isinstance(x['choice'], str)
        assert isinstance(plot, list) and isinstance(plot[0], str) and isinstance(plot[1], str)
    else:
        raise NotImplementedError

def make_prompt(mode, preference, plot, format_type="oasst"):
    check_input_format(mode, preference=preference, plot=plot)
    if mode == 'score':
        icl_content = ""
        for i,x in enumerate(preference):
            case = SCORE_CASE_TEMPLATE.format(n=i, plot=clean_plot(x['plot']), review=clean_review(x['review']), score=x['score'])
            icl_content += case + '\n'
        prompt = SCORE_PROBLEM_PROMPT_TEMPLATE.format(icl_example=icl_content, plot=clean_plot(plot))
        messages = [{'role': 'system', 'content': SCORE_SYSTEM_PROMPT + model_server_message_end},
                {'role': 'user', 'content': prompt + model_server_message_end},
                {'role': 'assistant', 'content': '[Review] Here is the Json format of the review: '}]
    elif mode == 'rank':
        x = preference[0]
        icl_content = RANK_CASE_TEMPLATE.format(plan1=clean_plot(x['plot1']), plan2=clean_plot(x['plot2']), completion=RANK_COMPLETION.format(choice=x['choice']))
        prompt = RANK_PROBLEM_PROMPT_TEMPLATE.format(icl_example=icl_content, plan1 = plot[0], plan2=plot[1])
        messages = [{'role': 'system', 'content': RANK_SYSTEM_PROMPT + model_server_message_end},
                {'role': 'user', 'content': prompt + model_server_message_end},
                {'role': 'assistant', 'content': 'Here is a JSON response: '}]
    else:
        raise NotImplementedError


    if format_type == "oasst":
        flat_messages = convert_messages_to_oasst_format(messages, model_server_message_end)
    elif format_type == 'llama2':
        flat_messages = convert_messages_to_llama2_format(messages)
    else:
        flat_messages = prompt
    
    return flat_messages


def get_options():
    args = argparse.ArgumentParser()
    # data options
    args.add_argument("--mode", type=str, default='evaluate', choices=['test', 'interactive'])
    args.add_argument("--input_file", type=str, default="../data/PerMPST.k3.sample.jsonl")
    args.add_argument("--save_file", type=str, default="../results/PerMPST.k3.result.jsonl")
    args.add_argument("--seed", type=int, default=42)

    # model options
    args.add_argument("--device", type=str, default="cuda")
    args.add_argument("--ckpt", type=str, default="output/checkpoint-217")
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--temperature", type=float, default=0.8)


    args = args.parse_args()
    return args


# python inference.py --input_file ../data/PerMPST.k3.sample.jsonl --ckpt output/checkpoint-240


if __name__ == "__main__":
    args = get_options()

    if args.mode == 'evaluate':
        data = [json.loads(line) for line in open(args.input_file)]
        prompts = [x['prompt'] if isinstance(x['prompt'], str) else x['prompt'][0] for x in data]
    elif args.mode == 'interactive':
        mode, preference, plot = get_input()
        if preference == [] and plot == []:
            exit()
        prompts = make_prompt(mode, preference, plot)
        print(prompts)
        prompts = [prompts]
    else:
        raise NotImplementedError
    
    # load model
    model = PerSE(model_name_or_path=args.ckpt, device="cuda")

    # inference
    print("Start inference for {} examples".format(len(prompts)))
    configs = {
        "max_new_tokens": MAX_TARGET_LENGTH,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature
        }
    responses = model.inference(prompts, batch_size=args.batch_size, **configs)

    # save results
    with open(args.save_file, "w") as f:
        for p, r in zip(prompts, responses):
            res = {
                'prompt': p,
                'response': r
            }
            f.write(json.dumps(res) + "\n")
