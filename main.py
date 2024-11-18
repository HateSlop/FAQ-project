from make_data import generate_faq
from make_vec import generate_vec
from utils import call_openai, get_embedding
import numpy as np


def search(query, FAQ, index):

    # ID를 키로 FAQItem 객체를 저장하는 딕셔너리 생성
    faq_dict = {item.index: item for item in FAQ.faqs}

    # 검색할 쿼리의 임베딩 생성
    query_vector = np.array(get_embedding(query), dtype=np.float32).reshape(1, -1)
    
    # 상위 K개의 결과 검색
    k = 3
    distances, indices = index.search(query_vector, k)
    
    # ID를 통해 structured_faq.faqs에서 question과 answer 추출
    results = [(faq_dict[_id].question, faq_dict[_id].answer) for _id in indices[0] if _id != -1]
    
    # 검색 결과 출력
    print("관련 FAQ:")
    for idx, (question, answer) in enumerate(results, 1):
        print(f"{idx}. 질문: {question}")
        print(f"   답변: {answer}")

    prompt_faq = f"""제시된 관련있는 FAQ를 참고하여 아래의 질문에 적절한 대답을 하시오.
    만약 제시된 FAQ와 질문이 관련성이 없다면 관련된 FAQ가 존재하지 않는다고 답하시오.

    FAQ:
    {results}
    Question:
    {query}
    """

    final_result = call_openai(prompt_faq, model='gpt-4o-mini')

    return final_result

def main(query):

    #검색 데이터 url
    url = "https://namu.wiki/w/%EB%93%9C%EB%9E%98%EA%B3%A4%EB%B3%BC"

    #생성 FAQ 수
    num_faqs = 20

    #데이터 생성 및 정렬화
    FAQ = generate_faq(url, num_faqs)

    #데이터 임베딩
    index = generate_vec()

    print(f"입력 질문: {query}")

    #최종 결과 출력
    results = search(query, FAQ, index)

    print(f"최종 답변: {results}")



#입력 질문
query = "드래곤볼은 언제 시작되었나요?"
final_result = main(query)