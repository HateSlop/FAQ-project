import openai
import pickle
import faiss
import numpy as np

# OpenAI API 설정
openai.api_key = "YOUR_OPENAI_API_KEY"

def load_faiss_database(file_path: str):
    """
    FAISS 데이터베이스를 로드합니다.
    """
    return faiss.read_index(file_path)

def get_embedding(text, model="text-embedding-ada-002"):
    """
    텍스트를 벡터로 변환합니다.
    """
    response = openai.Embedding.create(input=[text], model=model)
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

def search_similar_question(user_question, faiss_db, k=3):
    """
    사용자 질문에 대해 FAISS 데이터베이스에서 가장 유사한 질문을 검색합니다.
    """
    query_embedding = get_embedding(user_question).reshape(1, -1)
    distances, indices = faiss_db.search(query_embedding, k)
    return indices[0], distances[0]

def generate_answer(user_question, matched_question, matched_answer):
    """
    OpenAI 모델을 사용하여 최종 답변을 생성합니다.
    """
    if matched_answer:
        prompt = f"""
        사용자 질문: "{user_question}"
        매칭된 질문: "{matched_question}"
        매칭된 답변: "{matched_answer}"

        위 정보를 바탕으로 사용자에게 간단한 답변을 제공하세요.
        """
    else:
        prompt = f"""
        사용자 질문: "{user_question}"

        관련 정보를 찾을 수 없습니다. 사용자에게 답변을 제안하세요.
        """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()
