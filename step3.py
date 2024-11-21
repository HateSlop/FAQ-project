import config
from langchain.embeddings.openai import OpenAIEmbeddings
from step2 import load_json
import faiss
import numpy as np

OPENAI_API_KEY = config.OPENAI_API_KEY
#text-embedding-ada-002 model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 질문 리스트를 벡터화
def vectorize_questions(questions):
    return embeddings.embed_documents(questions)

def create_faiss_index_from_vectors(vectors):
    dimension = len(vectors[0])  # 벡터 차원
    index = faiss.IndexFlatL2(dimension)  # L2 거리 기반 인덱스 생성
    index.add(np.array(vectors, dtype="float32"))  # 벡터 추가
    return index

def save_faiss_index(index, file_path):
    try:
        faiss.write_index(index, file_path)
    except Exception as e:
        print(f"FAISS 인덱스 저장 실패: {e}")