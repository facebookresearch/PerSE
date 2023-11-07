import os
import json
import random

import argparse

random.seed(42)

# python scripts/convert_data.py --input_file data/PerMPST.k3.sample.jsonl --output_file data/PerMPST.k3.src.json

def get_options():
    args = argparse.ArgumentParser()
    # data options
    args.add_argument("--input_file", type=str, default="data/input.json")
    args.add_argument("--output_file", type=str, default="output/")
    args.add_argument("--do_shuffle", type=bool, default=False)

    args = args.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_options()
    print(args)

    input_file =  args.input_file
    output_file = args.output_file

    dataset = {
        "type": "text2text", 
        "instances": []
    }
    with open(args.input_file) as fin:
        for line in fin:
            data = json.loads(line)
            ex = {
                'idx': data['idx'],
                'input': data['prompt'][0],
                'output': data['completion']
            }
            
            dataset['instances'].append(ex)

    if args.do_shuffle:
      random.shuffle(dataset['instances'])
    
    print("Convert {} examples to {}".format(len(dataset['instances']), output_file))
    json.dump(dataset, open(output_file, 'w'))

            
      

