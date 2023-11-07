import os
import sys

import re
import ast
import json

def load_json(path, index):
  start, end = None, None
  if index != "-1":
    parts = index.split('-')
    if len(parts) == 1:
      point = int(parts[0])
      start, end = point, point+1
    else:
      start, end = int(parts[0]), int(parts[1])
    print(f"Start: {start}, End: {end}")

  if os.path.isdir(path):
    data = []
    
    if start is None or end is None:
      files = [x for x in os.listdir(path) if x.endswith('.json')]
      start, end = 0, len(files)
      print(f"Start: {start}, End: {end}")

    for i in range(start, end):
      file = os.path.join(path, f'{i}.json')
      data.append(json.load(open(file)))

    return data, start, end
  
  else:

    if path.endswith('jsonl'):
      data = [json.loads(x) for x in open(path)]
    else:
      data = json.load(open(path))
      
    if start is None or end is None:
      start, end = 0, len(data)
      print(f"Start: {start}, End: {end}")
    return data[start:end], start, end
  



def get_valid_entry(content):
  try:
    valid_entry = ast.literal_eval('[' + content +']')
  except:
    key_value_pattern = r'([^:]+):(("[^"]+")|([^,]+)),?'
    
    valid_entry = {}
    for x in re.findall(key_value_pattern, content):
      key = x[0].strip().lstrip('{')
      value = x[1].strip().rstrip(';').rstrip('}')
      xx = '{' + key + ':' + value + '}'
      try:
        xx = ast.literal_eval(xx)
        if isinstance(xx, dict):
          valid_entry.update(xx)
      except:
        continue
  return valid_entry


def check_json(response):
  response = response.replace('\n', '').replace('\r', '')
  response = response.replace("  ", "")
  code_env = r'```([\s\S]*?)```'
  match_content = re.findall(code_env, response)
  if match_content != []:
    response = match_content[-1].lstrip('json').lstrip('css').lstrip('javascript').lstrip('vbnet')
    response = response.replace(': "', ':@').replace('","Sco', '@,"Sco').replace('"', "'").replace('@', '"')
  else:
    json_env = r'({([\s\S]*?)}+)'
    match_content = re.findall(json_env, response)
    if match_content  != []:
      response = match_content[-1][0]


  try:
    content = ast.literal_eval(response)
    response = json.dumps(content)
  except:
    repaired_response = response
    json_env = r'{([\s\S]*?)}'
    inner_content = response.strip()[1:-1]
    for x in re.findall(json_env, inner_content):
      xx = x.strip()
      valid_entries = get_valid_entry(xx)
      valid_str = json.dumps(valid_entries)
      repaired_response = response.replace('{'+x+'}', valid_str)
    response = repaired_response
  
  try:
    content = json.loads(response)
  except:
    return False, response

  return True, content



def clean_plot(plot):
  sentences = plot.split("\n")
  sentences = [x.strip() for x in sentences]
  if sentences[0].startswith('*'):
    sentences = [x for x in sentences if x.startswith('*')]
  if sentences[0].startswith('-'):
    sentences = [x for x in sentences if x.startswith('-')]
  content = "\n".join(sentences).strip()
  end_idx = content.rfind('.')
  content = content[:end_idx+1]
  return content

def clean_review(review, maxlen=150):
  if review == '':
    return review
  review = review.replace('"','').replace("'","")
  review = review + '.' if not review.endswith('.') else review
  review = " ".join(review.split(' ')[:maxlen])
  end_idx = review.rfind('.')
  review = review[:end_idx+1]
  return review


def convert_messages_to_oasst_format(messages, message_end):
    prompt = ''
    for message in messages:
        if message['role'] == 'system':
            continue
            # prompt += '<|system|>' + message['content']
        elif message['role'] == 'user':
            prompt += '<|prompter|>' + message['content']
        elif message['role'] == 'assistant':
            prompt += '<|assistant|>' + message['content']
    if messages[-1]['content'].endswith(message_end):
        if messages[-1]['role'] == 'user':
            prompt += '<|assistant|>'
        else:
            prompt += '<|prompter|>'
    return prompt


def convert_messages_to_llama2_format(dialog, message_start='<s>', message_end='</s>'):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    
    for d in dialog:
        d['content'] = d['content'].replace(message_start, "").replace(message_end, "")

    if dialog[0]["role"] != "system":
        dialog = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + dialog
    dialog = [
        {
            "role": dialog[1]["role"],
            "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
        }
    ] + dialog[2:]

    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )


    if dialog[-1]["role"] != "user":
        print(f"Last message must be from user, got {dialog[-1]['role']}. Remove last message: {dialog[-1]}")
        dialog = dialog[:-1]

    prompts = ""
    for prompt, answer in zip(dialog[::2],dialog[1::2]):
        prompts += message_start + f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} " + message_end

    if len(dialog) % 2 == 1:
        prompts += message_start + f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"

    return prompts
