import pickle
import numpy as np
from utils import get_embedding


def make_final_prompt(query: str, index):
    # Pickle 파일에서 데이터 읽기
    with open("res/qna_items.pkl", "rb") as file:
        qna_as_list = pickle.load(file)

    # qna_dict 생성
    qna_dict = {item.index: item for item in qna_as_list.qnas}

    # Query 임베딩 생성
    query_embedding = np.array(get_embedding(query)).reshape(1, -1)

    # FAISS 인덱스에서 검색
    k = 3
    d, i = index.search(query_embedding, k)

    # 검색 결과 생성
    results = [
        (qna_dict[id].question, qna_dict[id].answer)
        for id in i[0]
        if id != -1
    ]

    # FAQ 출력 형식 생성
    if results:
        faq_output = "\n".join(
            [f"{idx + 1}. Question: {q}, Answer: {a}" for idx, (q, a) in enumerate(results)]
        )
    else:
        faq_output = "관련된 FAQ가 존재하지 않습니다."

    # 프롬프트 생성
    prompt_qna = f"""제시된 관련있는 FAQ를 참고하여 아래의 질문에 적절한 대답을 하시오.
    만약 제시된 FAQ와 질문이 관련성이 없다면 관련된 FAQ가 존재하지 않는다고 답하시오.
    FAQ:
    {faq_output}
    Question:
    {query}
    """
    #print(prompt_qna)
    return prompt_qna

