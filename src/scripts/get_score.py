import os
import sys
import json
import fire

import re
import ast

import pandas as pd


from utils.correlation import calculate_correlation
from utils.tools import check_json

DEBUG=False

def check_review(response):
  if not isinstance(response, str):
    return False, None
  if response.startswith('"Review"'):
    response = "```json{" + response
  
  status, content = check_json(response)

  if status == False:
    return False, content
  
  if not isinstance(content, dict) or 'Review' not in content.keys() or 'Score' not in content.keys():
    return False, None
  if not isinstance(content['Score'], int):
    return False, None
  
  content = content.replace("\n", '') if isinstance(content, str) else content

  return True, content


def run(result_file, info_file, savename=None, max_num = -1):

  df_ = pd.read_json(result_file, orient='records', lines=True)
  info_df_ = pd.read_json(info_file, orient='records', lines=True)
  print('Ori Info:', info_df_.shape)

  if max_num > 0:
    df_ = df_[:max_num]
    print('Select Info:', df_.shape)

  info_df_['prompt_str'] = info_df_['prompt'].apply(lambda x:x[0].strip() if isinstance(x, list) else x.strip())
  df_['prompt_str'] = df_['prompt'].apply(lambda x:x[0].strip() if isinstance(x, list) else x.strip())
  info_df_ = info_df_.drop_duplicates('prompt_str')
  print('Info remove duplicate :', info_df_.shape)
  df_ = df_.drop_duplicates('prompt_str')
  print('Result remove duplicate :', df_.shape)
  
  info_df_['idx'] = range(0, len(info_df_))
  df_all = pd.merge(info_df_, df_, on=['prompt_str'])
  print('Merge Info:', df_all.shape)
  df_all = df_all.drop(['prompt_x', 'prompt_y'], axis=1)

  for idx, value in df_all.iterrows():
      status, refer = check_json(value['completion'])
      if status:
          df_all.at[idx, 'refer_score'] = refer['Score']
          df_all.at[idx, 'refer_text'] = refer['Review']
      else:
          df_all.at[idx, 'refer_score'] = -1
          if DEBUG:
            print("Refer", idx)
            print(value['completion'])

      status, hypo = check_json(value['response'])
      if status:
          df_all.at[idx, 'hypo_score'] = hypo['Score']
          df_all.at[idx, 'hypo_text'] = hypo['Review']
      else:
          df_all.at[idx, 'hypo_score'] = -1
          df_all.at[idx, 'hypo_text'] = value['response']
          if DEBUG:
            print("Hypo", idx)
            print(value['response'])

  if savename != None:
    print(f'Save to {savename}')
    df_all.to_csv(savename)

  human = df_all[df_all['hypo_score'] != -1]['refer_score'].values
  pred = df_all[df_all['hypo_score'] != -1]['hypo_score'].values

  res = calculate_correlation(pred_score=pred, human_score=human)
  print(res)

  return

# python get_score.py --result_file results/version.jsonl --info_file review.valid.c1  --savename logs/version.result.csv

if __name__ == "__main__":
    fire.Fire(run)