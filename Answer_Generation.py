import os
import openai
import faiss 
import numpy as np
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def load_faiss_database(file_path: str):
    return faiss.read_index(file_path)

def get_embedding(texts, model="text-embedding-3-small"):
    """
    OpenAI API를 사용하여 텍스트 임베딩을 생성하는 함수.
    """
    response = openai.Embedding.create(input=texts, model=model)
    embeddings = [item['embedding'] for item in response['data']]
    return np.array(embeddings)



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
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
            ]
    )
    answer = response.choices[0].message.content
    return answer


if __name__ == "__main__":
    # FAISS 데이터베이스 파일 경로
    faiss_db_path = "faiss_db.index" 

    try:
        # FAISS 데이터베이스 로드
        print("FAISS 데이터베이스를 로드 중...")
        faiss_db = load_faiss_database(faiss_db_path)
        print("FAISS 데이터베이스가 성공적으로 로드되었습니다.")

        # 테스트 사용자 질문
        user_question = "안녕"

        # FAISS에서 유사한 질문 검색
        print("유사한 질문을 검색 중...")
        indices, distances = search_similar_question(user_question, faiss_db)

        # 매칭된 질문과 답변 출력
        if len(indices) > 0:
            # 여기에 실제 데이터베이스에서 질문 및 답변을 가져오는 로직 필요
            matched_question = f"매칭된 질문 {indices[0]}"
            matched_answer = f"매칭된 답변 {indices[0]}"
            print(f"매칭된 질문: {matched_question}")
            print(f"매칭된 답변: {matched_answer}")
        else:
            matched_question = None
            matched_answer = None

        # OpenAI 모델을 사용해 최종 답변 생성
        print("답변을 생성 중...")
        answer = generate_answer(user_question, matched_question, matched_answer)
        print(f"생성된 답변: {answer}")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")