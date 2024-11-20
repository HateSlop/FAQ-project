import openai
import pickle

def generate_answer(user_question, embedding_model):
    # 데이터베이스 로드
    with open('faiss_db.pkl', 'rb') as f:
        faiss_db = pickle.load(f)

    # 유사한 질문 검색
    results = faiss_db.similarity_search(user_question, k=1)

    if results:
        matched_result = results[0]
        matched_question = matched_result.page_content
        matched_answer = matched_result.metadata.get('answer', '')
    else:
        matched_question = None
        matched_answer = None

    # 최종 답변 생성
    if matched_answer:
        prompt = f"""
        사용자 질문: "{user_question}"
        매칭된 FAQ 질문: "{matched_question}"
        매칭된 FAQ 답변: "{matched_answer}"

        위 정보를 바탕으로 사용자에게 친절하고 정확한 답변을 100자 이내로 제공해주세요.
        """
    else:
        prompt = f"""
        사용자 질문: "{user_question}"

        해당 질문에 대한 정보를 찾을 수 없습니다. 사용자에게 정중하게 답변해주세요.
        """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )

    final_answer = response['choices'][0]['message']['content'].strip()
    return final_answer