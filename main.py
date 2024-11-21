import asyncio
import pickle
from prompt_template import generate_faq_prompt, generate_answer_prompt
from step1 import fetch_html, html_to_text
from step2 import chatgpt_generate_faq, structure_faq_data, save_to_json, load_json
from step3 import vectorize_questions, create_faiss_index_from_vectors, save_faiss_index
from step4 import search_similar_question, chatgpt_generate_answer

def save_faq_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_faq_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

async def main():
    # URL에서 HTML 데이터 가져오기
    url = "https://namu.wiki/w/%EC%A0%9C2%EC%B0%A8%20%EC%84%B8%EA%B3%84%20%EB%8C%80%EC%A0%84"
    html_content = await fetch_html(url)
    
    # HTML 데이터를 텍스트로 변환
    text_content = html_to_text(html_content)

    # GPT 모델에 텍스트 입력 및 FAQ 생성
    faq_query = generate_faq_prompt + text_content
    raw_faq_text = chatgpt_generate_faq(faq_query)

    structured_faq = structure_faq_data(raw_faq_text)
    file_path = save_to_json(structured_faq, "faq_data.json")

    faq_data = load_json(file_path)
    save_faq_data(faq_data, "faq_data.pkl")

    questions =  [item["question"] for item in faq_data["faqs"]]
    question_vectors = vectorize_questions(questions)
    faiss_index = create_faiss_index_from_vectors(question_vectors)
    save_faiss_index(faiss_index, "faiss_question_index.bin")

    user_question = input("==========================================================\n제 2차 세계대전에 대해 궁금한 점은 무엇이든지 물어보세요!\n\n질문: ")

    # 유사한 질문 검색
    faq_question, faq_answer = search_similar_question(user_question, faiss_index, faq_data["faqs"])

    answer_query = generate_answer_prompt.format(user_question=user_question,
    faq_question=faq_question,
    faq_answer=faq_answer)
    answer = chatgpt_generate_answer(answer_query)
    print("\n답변: ", answer, "\n==========================================================")

if __name__ == "__main__":
    asyncio.run(main())