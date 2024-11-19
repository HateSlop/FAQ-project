from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import json
import os
import config

def create_vector_store():
    try:
        # FAQ 데이터 로드
        with open('news-project/faqs.json', 'r', encoding='utf-8') as f:
            faqs = json.load(f)
        
        print(f"Loaded {len(faqs)} FAQs")
        
        # 문서 객체 생성
        documents = [
            Document(
                page_content=faq['question'],
                metadata={'answer': faq['answer']}
            )
            for faq in faqs
        ]
        
        print("Converting FAQs to document objects...")
        
        # OpenAI 임베딩 초기화
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=config.OPENAI_API_KEY
        )
        
        print("Creating vector store...")
        
        # FAISS 벡터 스토어 생성
        vector_store = FAISS.from_documents(
            documents,
            embeddings
        )
        
        print("Vector store created successfully")
        
        # 벡터 스토어를 로컬에 저장
        vector_store.save_local('news-project/faiss_store')
        
        print("Vector store saved to disk")
        
        return vector_store
        
    except Exception as e:
        print(f"Error in create_vector_store: {e}")
        raise

def load_vector_store():
    """저장된 벡터 스토어를 로드합니다."""
    try:
        # OpenAI 임베딩 초기화
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=config.OPENAI_API_KEY
        )
        
        # 안전한 역직렬화 옵션 추가
        vector_store = FAISS.load_local(
            'news-project/faiss_store', 
            embeddings,
            allow_dangerous_deserialization=True  # 안전한 소스에서만 True로 설정
        )
        print("Vector store loaded successfully")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def test_search(vector_store, query="수능 시험은 언제 치나요?", k=3):
    """벡터 스토어 검색을 테스트합니다."""
    try:
        results = vector_store.similarity_search(query, k=k)
        
        print(f"\nTest Search Results for: {query}")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Question: {doc.page_content}")
            print(f"Answer: {doc.metadata['answer']}")
            
    except Exception as e:
        print(f"Error during search: {e}")

def main():
    try:
        # 벡터 스토어 생성
        vector_store = create_vector_store()
        
        # 생성된 벡터 스토어로 바로 테스트
        test_search(vector_store)
        
        print("\nTesting loaded vector store...")
        # 저장된 벡터 스토어 로드 및 테스트
        loaded_store = load_vector_store()
        if loaded_store:
            test_search(loaded_store, "수능 준비는 어떻게 해야 하나요?")
                
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()