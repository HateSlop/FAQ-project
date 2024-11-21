import os
import pickle
import numpy as np
import openai
import faiss
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pydantic 데이터 모델 정의
class FAQItem(BaseModel):
    index: int
    question: str
    answer: str

# OpenAI 임베딩 생성 함수
def get_embeddings(texts: List[str], model="text-embedding-ada-002"):
    """
    OpenAI API를 사용하여 텍스트 임베딩을 생성하는 함수.
    """
    response = openai.Embedding.create(input=texts, model=model)
    embeddings = [item['embedding'] for item in response['data']]
    return embeddings

# FAQ 리스트에서 벡터화된 벡터를 생성하고 FAISS 벡터 데이터베이스로 저장하는 함수
def generate_faiss_vector(faq_items: List[FAQItem], model="text-embedding-ada-002"):
    # FAQ 질문 텍스트 추출
    questions = [item.question for item in faq_items]

    # OpenAI를 이용해 질문을 벡터화
    embeddings = get_embeddings(questions, model=model)

    # 벡터화된 질문들을 numpy 배열로 변환
    embeddings_array = np.array(embeddings).astype(np.float32)

    # FAISS 인덱스 생성 (유사도 검색을 위한)
    faiss_index = faiss.IndexFlatL2(embeddings_array.shape[1])  # L2 거리 기반 인덱스
    faiss_index.add(embeddings_array)  # 벡터들을 인덱스에 추가

    # 벡터화된 인덱스 파일로 저장
    faiss.write_index(faiss_index, 'faiss_db.index')

    # FAQ 데이터 pickle로 저장
    with open('faq_items.pkl', 'wb') as f:
        pickle.dump([item.dict() for item in faq_items], f)

    print("FAISS 벡터스토어 및 FAQ 데이터 저장 완료")
