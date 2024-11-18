from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from utils import call_openai
from pydantic import BaseModel, ValidationError
from typing import List
import json
import pickle

class FAQItem(BaseModel):
    index: int
    question: str
    answer: str

class FAQList(BaseModel):
    faqs: List[FAQItem]

def generate_faq(url: str, num_faqs: int):

    # HTML 데이터 로드
    loader = AsyncHtmlLoader(
    [url],
    header_template={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    })

    docs = loader.load()

    # 텍스트 변환
    transformer = Html2TextTransformer()
    plain = transformer.transform_documents(docs)
    plain_text = plain[0].page_content

    # GPT 모델에 전달할 프롬프트 생성
    prompt = """개의 구체적인 FAQ를 아래의 context를 보고 생성하시오.
    출력포맷은 리스트이며, 세부 내용은 다음과 같습니다.
    반드시 출력포맷을 지켜서 생성하고, 다른 텍스트를 생성하거나 json으로 시작하지 마시오.

    출력포맷:
    [{"index": <인덱스 번호>, "question": <질문>, "answer": <답변>}, ...]

    Context:
    """

    # OpenAI API 호출 및 FAQ 생성
    FAQ = call_openai(str(num_faqs) + prompt + plain_text, model='gpt-4o-2024-05-13')
    FAQ_data = json.loads(FAQ)

    # Pydantic 모델로 데이터 구조화
    try:
        structured_faq = FAQList(faqs=FAQ_data)
        with open('./res/FAQ_data.pkl', 'wb') as f:
            pickle.dump(structured_faq, f)
        print("데이터 생성 완료")
        return structured_faq
    except ValidationError as e:
        print(f"검증 오류 발생: {e}")
        return None

