import os
import openai
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

from data_collection import load_and_transform
from faq_generation import generate_faq
from vectorization import vectorize_faq
from question_answering import generate_answer

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY  

# 임베딩 모델 초기화
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY  
)

if __name__ == "__main__":
    # 1. 데이터 수집 및 텍스트 변환
    url = 'https://namu.wiki/w/현대자동차'
    print("데이터를 수집하고 있습니다...")
    text = load_and_transform(url)
    print("데이터 수집 및 텍스트 변환 완료")

    # 2. FAQ 생성
    print("FAQ를 생성하고 있습니다...")
    faq_items = generate_faq(text, num_faqs=10)
    print("FAQ 생성 완료")

    # 생성된 FAQ 출력 
    # for idx, item in enumerate(faq_items):
    #     print(f"\nFAQ {idx+1}")
    #     print(f"Q: {item.question}")
    #     print(f"A: {item.answer}")

    # 3. 벡터화 및 데이터베이스 생성
    print("\n벡터화 및 데이터베이스를 생성하고 있습니다...")
    vectorize_faq(faq_items, embedding_model)
    print("벡터화 및 데이터베이스 생성 완료")

    # 4. 사용자 질문에 대한 답변 생성
    while True:
        user_question = input("\n질문을 입력하세요 (종료하려면 'exit' 입력): ")
        if user_question.lower() == 'exit':
            break
        print("답변을 생성하고 있습니다...")
        answer = generate_answer(user_question, embedding_model)
        print(f"답변: {answer}")