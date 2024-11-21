from make_data import generate_faq  # FAQ 생성 함수
from make_vec import generate_vec  # 벡터화 및 데이터베이스 생성 함수
from utils import call_openai, get_embedding  # OpenAI 호출 및 임베딩 생성 함수
import numpy as np

# 검색 함수
def search(query, FAQ, index):
    """
    사용자의 질문을 FAQ 데이터베이스에서 검색하고 적절한 답변을 생성.
    
    Parameters:
    - query: 사용자의 질문 (문자열)
    - FAQ: FAQ 데이터 (generate_faq로 생성된 데이터)
    - index: FAISS 벡터화 데이터베이스

    Returns:
    - OpenAI API를 통해 생성된 최종 답변 (문자열)
    """

    # FAQ 데이터를 ID를 키로 하는 딕셔너리로 변환
    faq_dict = {item.index: item for item in FAQ.faqs}

    # 사용자의 질문을 벡터화
    query_vector = np.array(get_embedding(query), dtype=np.float32).reshape(1, -1)
    
    # 상위 3개의 FAQ를 검색
    k = 3
    distances, indices = index.search(query_vector, k)
    
    # 검색 결과를 FAQ 데이터에서 가져옴
    results = [(faq_dict[_id].question, faq_dict[_id].answer) for _id in indices[0] if _id != -1]
    
    # 검색된 FAQ 출력
    print("관련 FAQ:")
    for idx, (question, answer) in enumerate(results, 1):
        print(f"{idx}. 질문: {question}")
        print(f"   답변: {answer}")

    prompt_faq = f"""제시된 FAQ 데이터를 바탕으로 아래의 질문에 적절한 답변을 작성하시오.
                FAQ는 리그 오브 레전드와 관련된 정보를 제공하기 위해 작성되었습니다.

                FAQ 데이터:
                {results}

                사용자 질문:
                {query}

                답변:
"""
    final_result = call_openai(prompt_faq, temperature=0.7, model='gpt-4o-mini')


    return final_result


def main(query):
    """
    리그 오브 레전드 챗봇의 전체 실행 흐름을 관리하는 함수.
    
    Parameters:
    - query: 사용자가 입력한 질문 (문자열)
    """

    # 리그 오브 레전드 나무위키 URL
    url = "https://namu.wiki/w/%EB%A6%AC%EA%B7%B8%20%EC%98%A4%EB%B8%8C%20%EB%A0%88%EC%A0%84%EB%93%9C"

    # FAQ 데이터 생성
    num_faqs = 20  # 생성할 FAQ 수
    FAQ = generate_faq(url, num_faqs)

    # 데이터 벡터화 및 인덱스 생성
    index = generate_vec()

    # 사용자 입력 출력
    print(f"입력 질문: {query}")

    # 검색 및 최종 답변 생성
    results = search(query, FAQ, index)

    # 최종 결과 출력
    print(f"최종 답변: {results}")


# 실행 예시
query = "리그오브레전드는 어떤 게임인가요?"
final_result = main(query)
