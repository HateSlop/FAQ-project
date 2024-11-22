import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import List
import openai
import pickle

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Pydantic 데이터 모델 정의
class FAQItem(BaseModel):
    index: int
    question: str
    answer: str

# OpenAI를 이용한 FAQ 생성
def generate_faq_from_text(context, num_faqs: int = 10) -> List[FAQItem]:
    """
    GPT-4o 모델을 사용하여 입력 텍스트로부터 FAQ를 생성합니다.
    """
    prompt = f"""
    출력 형식은 반드시 아래의 JSON 형식을 따르세요:
    [
    {{
        "index": <인덱스 번호>,
        "question": "<질문>",
        "answer": "<답변>"
    }},
    ...
    ]

    다음 텍스트에서 {num_faqs}개의 FAQ를 생성해주세요.
    이 형식을 벗어나지 말고, 다른 텍스트를 추가하지 마세요.

    텍스트:
    {context}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
                ]
        )
        
        print(response)
        answer = response.choices[0].message.content

        cleaned_content = answer.strip('```json\n').strip('```')

        # JSON 변환 및 데이터 검증
        faq_list = json.loads(cleaned_content)
        faq_items = [FAQItem(**item) for item in faq_list]
        return faq_items
    
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"FAQ 생성 중 오류 발생: {e}")
        return []
    except Exception as e:
        print(f"OpenAI API 호출 중 오류 발생: {e}")
        return []

# FAQ 데이터를 JSON 파일로 저장하는 함수
def save_faq_to_json(faqs: List[FAQItem], filename="faq.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump([faq.model_dump() for faq in faqs], f, ensure_ascii=False, indent=4)

# FAQ 데이터를 Pickle 파일로 저장하는 함수
def save_faq_to_pickle(faqs: List[FAQItem], filename="faq.pkl"):
    with open(filename, "wb") as f:
        pickle.dump([faq.model_dump() for faq in faqs], f)
'''    
if __name__ == "__main__":
    # 테스트용 입력 텍스트
    context = """
    Python은 널리 사용되는 고급 프로그래밍 언어로, 읽기 쉬운 문법과 다양한 라이브러리를 제공합니다.
    Python의 주요 특징으로는 간단한 문법, 플랫폼 독립성, 방대한 표준 라이브러리, 그리고 객체 지향 프로그래밍 지원이 있습니다.
    Python은 웹 개발, 데이터 분석, 인공지능, 스크립트 작성 등 다양한 분야에서 활용됩니다.
    """
    
    # FAQ 생성
    print("FAQ 생성 중...")
    faq_items = generate_faq_from_text(context, num_faqs=10)
    
    if faq_items:
        # 생성된 FAQ 출력
        print("생성된 FAQ:")
        for item in faq_items:
            print(f"{item.index}. {item.question}")
            print(f"   {item.answer}")
        
        # JSON 저장
        save_faq_to_json(faq_items)
        print("FAQ가 JSON 파일로 저장되었습니다: faq.json")

        # Pickle 저장
        save_faq_to_pickle(faq_items)
        print("FAQ가 Pickle 파일로 저장되었습니다: faq.pkl")
    else:
        print("FAQ 생성에 실패했습니다.")
'''  