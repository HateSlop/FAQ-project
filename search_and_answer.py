from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
import config

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

def load_vector_store():
    """저장된 벡터 스토어를 로드합니다."""
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=config.OPENAI_API_KEY
        )
        
        vector_store = FAISS.load_local(
            'news-project/faiss_store',
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def generate_answer(user_question, retrieved_faqs):
    """검색된 FAQ를 바탕으로 답변을 생성합니다."""
    
    # FAQ 컨텍스트 생성
    faq_context = "\n\n".join([
        f"질문: {faq.page_content}\n답변: {faq.metadata['answer']}"
        for faq in retrieved_faqs
    ])
    
    prompt = f"""당신은 수능과 대입 전문가입니다. 
다음의 FAQ 데이터베이스를 참고하여 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요.

참고할 FAQ:
{faq_context}

사용자 질문: {user_question}

답변 작성 시 주의사항:
1. FAQ의 정보를 기반으로 하되, 자연스럽게 통합하여 답변하세요.
2. 명확하고 구체적으로 설명하세요.
3. 필요한 경우 예시를 들어 설명하세요.
4. 정확한 정보만을 포함하세요.
5. 친절하고 이해하기 쉬운 톤으로 답변하세요.

답변:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",  # 또는 gpt-4-turbo
            messages=[
                {"role": "system", "content": "당신은 수능과 대입 전문가입니다. 정확하고 도움이 되는 답변을 제공해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "죄송합니다. 답변 생성 중 오류가 발생했습니다."

def search_and_answer(question, k=3):
    """사용자 질문에 대한 검색 및 답변 생성을 수행합니다."""
    try:
        # 벡터 스토어 로드
        vector_store = load_vector_store()
        if not vector_store:
            return "죄송합니다. 데이터베이스 로드에 실패했습니다."
        
        # 유사 질문 검색
        similar_faqs = vector_store.similarity_search(question, k=k)
        
        # 검색된 FAQ 출력 (디버깅용)
        print("\n유사한 FAQ:")
        for i, faq in enumerate(similar_faqs, 1):
            print(f"\n{i}. 질문: {faq.page_content}")
            print(f"   답변: {faq.metadata['answer']}")
        
        # 답변 생성
        answer = generate_answer(question, similar_faqs)
        
        return answer
        
    except Exception as e:
        print(f"Error in search_and_answer: {e}")
        return "죄송합니다. 처리 중 오류가 발생했습니다."

def main():
    print("수능 FAQ 검색 시스템입니다. 종료하려면 'q'를 입력하세요.")
    
    while True:
        question = input("\n질문을 입력하세요: ").strip()
        
        if question.lower() == 'q':
            print("프로그램을 종료합니다.")
            break
            
        if not question:
            print("질문을 입력해주세요.")
            continue
        
        print("\n답변을 생성 중입니다...")
        answer = search_and_answer(question)
        print("\n답변:")
        print(answer)

if __name__ == "__main__":
    main()