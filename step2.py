import config
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import List
import json

# 질문과 답변을 위한 Pydantic 모델 정의
class QA(BaseModel):
    question: str
    answer: str

class FAQ(BaseModel):
    faqs: List[QA]

OPENAI_API_KEY = config.OPENAI_API_KEY
openai_client = OpenAI(api_key=OPENAI_API_KEY)
model = "gpt-4o"

def chatgpt_generate_faq(query):
    messages = [{
        "role": "system",
        "content" : "You are a helpful assistant that generates FAQ from input text."},
        {
            "role": "user",
            "content": query
        }]
    response = openai_client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer

# GPT 응답을 Pydantic 모델로 구조화
def structure_faq_data(raw_faq_text: str):
    faqs = []
    lines = [line.strip() for line in raw_faq_text.strip().split("\n") if line.strip()]

    for i in range(0, len(lines), 2):  # 두 줄씩 처리
        if lines[i].startswith("Q:") and lines[i + 1].startswith("A:"):
            question = lines[i][2:].strip()
            answer = lines[i + 1][2:].strip()
            faqs.append({"question": question, "answer": answer})

    # Pydantic 검증
    try:
        faq_model = FAQ(faqs=faqs)
        return faq_model.dict()
    except ValidationError as e:
        print("FAQ 데이터 검증 실패:", e)
        return None

# JSON 데이터를 파일로 저장하는 함수
def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return filename

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data