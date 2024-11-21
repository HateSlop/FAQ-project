import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import List
import openai
import pickle

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

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
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7
        )

        # API 응답 데이터 추출
        answer = response['choices'][0]['message']['content']

        # JSON 변환 및 데이터 검증
        faq_list = json.loads(answer)
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
    """
    FAQ 데이터를 JSON 파일로 저장합니다.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump([faq.model_dump() for faq in faqs], f, ensure_ascii=False, indent=4)

# FAQ 데이터를 Pickle 파일로 저장하는 함수
def save_faq_to_pickle(faqs: List[FAQItem], filename="faq.pkl"):
    """
    FAQ 데이터를 Pickle 파일로 저장합니다.
    """
    with open(filename, "wb") as f:
        pickle.dump([faq.model_dump() for faq in faqs], f)
