from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
import config
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import List
import json
import pickle
import os

OPENAI_API_KEY = config.OPENAI_API_KEY
openai_client = OpenAI(api_key=OPENAI_API_KEY)
model = "gpt-4o"
url = ["https://namu.wiki/w/%EB%8C%80%ED%95%99%EC%88%98%ED%95%99%EB%8A%A5%EB%A0%A5%EC%8B%9C%ED%97%98"]

# Pydantic 데이터 모델 정의
class QAItem(BaseModel):
    index: int
    question: str
    answer: str

class QAList(BaseModel):
    qnas: List[QAItem]


def url_loader(url: str):
    loader = AsyncHtmlLoader(url)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    return docs_transformed

def save_as_json_and_pickle(qna:str):
        # JSON 문자열 처리
    try:
        # "```json"과 마지막 "```" 제거
        qna = qna.strip("```json").strip("```").strip()
        # Python 리스트로 변환
        qna_list = json.loads(qna)  # JSON 리스트로 파싱
        # Pydantic 모델로 데이터 검증 
        qna_items = [QAItem(**item) for item in qna_list]
        qna_as_list = QAList(qnas=qna_list)
        #json 파일 저장
        if not os.path.exists("res"):
            os.makedirs("res")
        with open("res/qna_items.json", "w", encoding="utf-8") as f:
            json.dump([item.dict() for item in qna_items], f, ensure_ascii=False, indent=4)
        print("JSON 파일이 생성되었습니다: qna_items.json")
        # Pickle 파일 저장
        with open("res/qna_items.pkl","wb") as f:
            pickle.dump(qna_as_list, f)
        print("Pickle 파일이 생성되었습니다: res/qna_items.pkl")
    except json.JSONDecodeError as e:
        print("JSON 디코딩 에러:", e)
    except ValidationError as e:
        print("Pydantic 검증 에러:", e)