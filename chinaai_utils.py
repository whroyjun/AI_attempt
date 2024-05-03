import json
import requests
import os
import numpy as np
# thru authentifiation to get access token

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

#openai.api_key = os.getenv('OPENAI_API_KEY')
API_KEY = ""
SECRET_KEY = ""

def get_access_token():

    """
    use AK, SK to generate access Token
    :return access_token or None(error)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params ={
        "grant_type": "client_credentials",
        #"client_id": os.getenv('ERNIE_LIENT_ID'),
        "client_id": API_KEY,
        "client_secret": SECRET_KEY


    }
    return str(requests.post(url, params=params).json().get("access_token"))



#call 文心4.0 对话 接口
def get_completion(prompt):
    print ("###get_completion start")
    """封装  文心4.0 chat 接口"""
    #url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token=" + get_access_token()
    payload = json.dumps ({
        "messages":[
            {
                "role": "user",
                "content":prompt
            }
        ]
    })

    headers = {'Content-Type': 'application/json'}
    response = requests.request(
        "POST", url, headers=headers, data=payload).json()
    """
    OpenAI
    messages = context + [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    """
    #return response.choices[0].message["content"]
    print("calling completion response:",response)
    return response["result"]



#def get_embedding(text, model="text-embedding-ada-002"):
#return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
def get_embedding(prompts):
    """call 文心千帆 BGE Embedding 接口"""
    """封装 OpenAI 的 Embedding 模型接口"""
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/bge_large_en?access_token=" + get_access_token()
   
    #print("###url:",url)
    #print ("###chinaai_utilsprompt built start:", prompts)
    
    payload =json.dumps({
        #"input": [f"{prompts}"]
        "input": prompts
    })
    print("###begin to Embedding process payload:")
    
    headers = {'Content-Type': 'application/json'}
    response = requests.request(
        "POST", url, headers=headers, data=payload).json()
    print("###PDF_response:")
    data= response["data"]
    #checkEmbedding=[x["embedding"] for x in data]
    """
    if not np.issubdtype(checkEmbedding.dtype, np.integer) and not np.issubdtype(checkEmbedding.dtype, np.floating):  
        print("嵌入向量的数据类型不正确。应该是整数或浮点数。")
    """
    #print("###get_embedding return:", [x["embedding"] for x in data])
    return [x["embedding"] for x in data]
