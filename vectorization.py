import os
import pickle
import numpy as np
import openai
import faiss
from dotenv import load_dotenv
from pydantic import BaseModel



load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Pydantic 데이터 모델 정의
class FAQItem(BaseModel):
    index: int
    question: str
    answer: str

# OpenAI 임베딩 생성 함수
def get_embedding(texts, model="text-embedding-3-small"):
    """
    OpenAI API를 사용하여 텍스트 임베딩을 생성하는 함수.
    """
    response = openai.Embedding.create(input=texts, model=model)
    embeddings = [item['embedding'] for item in response['data']]
    return embeddings

# FAQ 리스트에서 벡터화된 벡터를 생성하고 FAISS 벡터 데이터베이스로 저장하는 함수
def generate_faiss_vector(faq_items, model="text-embedding-3-small"):
    # FAQ 질문 텍스트 추출
    questions = [item.question for item in faq_items]

    # OpenAI를 이용해 질문을 벡터화
    embeddings = get_embedding(questions, model=model)

    # 벡터화된 질문들을 numpy 배열로 변환
    embeddings_array = np.array(embeddings)

    # FAISS 인덱스 생성 (유사도 검색을 위한)
    faiss_index = faiss.IndexFlatL2(embeddings_array.shape[1])  # L2 거리 기반 인덱스
    faiss_index.add(embeddings_array)  # 벡터들을 인덱스에 추가

    # 벡터화된 인덱스 파일로 저장
    faiss.write_index(faiss_index, 'faiss_db.index')

    # FAQ 데이터 pickle로 저장
    with open('faq_items.pkl', 'wb') as f:
        pickle.dump([item.dict() for item in faq_items], f)

    print("FAISS 벡터스토어 및 FAQ 데이터 저장 완료")


# 테스트용 코드
if __name__ == "__main__":
    # 예시 FAQ 데이터 생성
    faq_data = [
        FAQItem(index=0, question="What is Python?", answer="Python is a programming language."),
        FAQItem(index=1, question="How to install Python?", answer="You can install Python from python.org."),
        FAQItem(index=2, question="What are Python's key features?", answer="Python supports readability, simplicity, and flexibility."),
    ]

    print("FAQ 데이터를 FAISS로 변환하고 저장 중...")

    # 벡터 데이터 생성 및 저장
    generate_faiss_vector(faq_data)

    # 저장된 데이터 확인
    if os.path.exists('faiss_db.index') and os.path.exists('faq_items.pkl'):
        print("FAISS 인덱스와 FAQ 데이터가 성공적으로 저장되었습니다.")
        print("FAISS 인덱스 파일 경로: faiss_db.index")
        print("FAQ 데이터 파일 경로: faq_items.pkl")
    else:
        print("데이터 저장에 실패했습니다.")