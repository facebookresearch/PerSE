import os
import sys
import json
import fire

import re
import ast

import pandas as pd
 
from utils.tools import check_json
from utils.correlation import calculate_correlation

DEBUG=False

def check_json(response):
  status, content = check_json(response)
  
  if status == False:
    response = '""' + content + '""'
    try:
      content = json.loads(response)
    except:
      return False, content

  return True, content

def format_answer(x):
  if x == 'Plot A':
    return 0
  elif x == 'Plot B':
    return 1
  elif x== 'N/A':
    return 2
  else:
    return -1

def get_results(data):
  query = sorted([k for k in data.keys() if k[:2] in ['Q1','Q3','Q4','Q5','Q6']])
  results = []
  for k in query:
    if k in data:
      results.append(format_answer(data[k]))
    else:
      results.append(-1)
  return results


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
    if status and 'Choice' in refer.keys():
      results = format_answer(refer['Choice'])
      df_all.loc[idx, 'refer_score'] = results
    else:
      df_all.loc[idx, 'refer_score'] = -1
      if DEBUG:
        print("==============Refer==============")
        print(value['completion'])

    status, hypo = check_json(value['response'])
    if status and isinstance(hypo, dict) and 'Choice' in hypo.keys():
      results = format_answer(hypo['Choice'])
      df_all.loc[idx, 'hypo_score'] = results
    else:
      df_all.loc[idx, 'hypo_score'] = -1
      if DEBUG:
        print("==============Hypo==============")
        print(value['response'])
        print(check_json(value['response']))


    if savename != None:
      print(f'Save to {savename}')
      df_all.to_csv(savename)

  if 'aspect' in df_all.columns:
    for name, df in df_all.groupby('aspect'):
      human = df[df['hypo_score'] != -1]['refer_score'].values
      pred = df[df['hypo_score'] != -1]['hypo_score'].values
      res = calculate_correlation(pred_score=pred, human_score=human)
      print(f'{name}\t',res)

  human = df_all[df_all['hypo_score'] != -1]['refer_score'].values
  pred = df_all[df_all['hypo_score'] != -1]['hypo_score'].values

  res = calculate_correlation(pred_score=pred, human_score=human)
  print(res)

  return

# python get_comparison.py --result_file results/version.jsonl --info_file annotation.valid.Q1  --savename logs/version.result.csv

if __name__ == "__main__":
    fire.Fire(run)
