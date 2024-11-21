import numpy as np
import config
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

OPENAI_API_KEY = config.OPENAI_API_KEY
#text-embedding-ada-002 model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
model = "gpt-4o-mini"

# 사용자 입력 질문을 벡터화하고 FAISS에서 검색
def search_similar_question(user_question, faiss_index, data_list):
    # 사용자 질문 벡터화
    query_vector = np.array([embeddings.embed_query(user_question)], dtype="float32")
    
    # FAISS에서 검색
    k = 1  # 상위 1개의 결과 검색
    distances, indices = faiss_index.search(query_vector, k)
    
    # 가장 유사한 질문의 인덱스
    closest_index = indices[0][0]
    similar_question = data_list[closest_index]["question"]
    answer = data_list[closest_index]["answer"]
    
    return similar_question, answer

def chatgpt_generate_answer(query):
    messages = [{
        "role": "system",
        "content" : "You are a helpful assistant that generates awnser based on searched FAQ and user question."},
        {
            "role": "user",
            "content": query
        }]
    response = openai_client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer