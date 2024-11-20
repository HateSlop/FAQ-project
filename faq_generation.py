import os
import openai
from dotenv import load_dotenv
import json
from pydantic import BaseModel
from typing import List

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

class FAQItem(BaseModel):
    question: str
    answer: str

def generate_faq(text, num_faqs=10):
    """
    주어진 텍스트를 기반으로 FAQ를 생성합니다.
    """
    prompt = f"""
아래의 내용을 읽고 {num_faqs}개의 구체적인 FAQ를 생성해주세요.
출력 형식은 다음과 같습니다:
[
  {{
    "question": "질문1",
    "answer": "답변1"
  }},
  ...
]
다른 텍스트는 생성하지 말고, 반드시 지정된 출력 형식만 따르세요.

내용:
\"\"\"{text}\"\"\"
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    faq_list_str = response['choices'][0]['message']['content']

    # 문자열을 리스트로 변환
    try:
        faq_list = json.loads(faq_list_str)
    except json.JSONDecodeError:
        print("JSON 디코딩 오류 발생. 출력 형식을 확인하세요.")
        return None

    # Pydantic을 사용하여 데이터 검증
    try:
        faq_items = [FAQItem(**item) for item in faq_list]
    except Exception as e:
        print(f"FAQ 데이터 검증 오류 발생: {e}")
        return None

    return faq_items