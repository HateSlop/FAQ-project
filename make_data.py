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
    """
    주어진 URL에서 HTML 데이터를 로드하여 FAQ 데이터를 생성합니다.

    Parameters:
    - url (str): 데이터를 가져올 웹 페이지 URL
    - num_faqs (int): 생성할 FAQ 개수

    Returns:
    - FAQList: 생성된 FAQ 데이터 리스트
    """

    # HTML 데이터 로드
    loader = AsyncHtmlLoader(
        [url],
        header_template={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
    )

    docs = loader.load()

    # 텍스트 변환
    transformer = Html2TextTransformer()
    plain = transformer.transform_documents(docs)
    plain_text = plain[0].page_content

    # GPT 모델에 전달할 프롬프트 생성
    prompt = f"""
아래의 컨텍스트에서 리그 오브 레전드와 관련된 {num_faqs}개의 구체적인 FAQ를 생성하십시오.
질문은 리그 오브 레전드의 게임 특징, 주요 모드, 개발자 정보, 플레이 방식, 밈 등 다양한 주제를 포함해야 합니다.

출력 포맷은 다음과 같습니다:
[{{"index": <인덱스 번호>, "question": <질문>, "answer": <답변>}}, ...]

아래는 FAQ 생성을 위한 컨텍스트입니다:
{plain_text}
"""


    # OpenAI API 호출 및 FAQ 생성
    try:
        FAQ = call_openai(prompt, model='gpt-4o-2024-05-13')
        FAQ_data = json.loads(FAQ)

        # Pydantic 모델로 데이터 구조화
        structured_faq = FAQList(faqs=FAQ_data)

        # 생성된 데이터를 Pickle로 저장
        with open('./res/FAQ_data.pkl', 'wb') as f:
            pickle.dump(structured_faq, f)

        print("데이터 생성 완료")
        return structured_faq

    except ValidationError as e:
        print(f"검증 오류 발생: {e}")
        return None

    except json.JSONDecodeError as e:
        print(f"JSON 디코딩 오류 발생: {e}")
        return None
