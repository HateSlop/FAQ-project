import asyncio
from data_collection import load_and_transform
from FAQ_Generation import generate_faq_from_text, save_faq_to_json, save_faq_to_pickle
from vectorization import generate_faiss_vector
from Answer_Generation import load_faiss_database, search_similar_question, generate_answer

async def main():
    # 1. 웹 데이터 수집
    url = "https://namu.wiki/w/%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD"
    print("웹 데이터 로드 및 텍스트 변환 중...")
    text_data = await load_and_transform(url)

    if not text_data:
        print("데이터를 가져오지 못했습니다. 프로그램을 종료합니다.")
        return

    # 2. FAQ 생성
    print("FAQ 데이터 생성 중...")
    faq_items = generate_faq_from_text(context=text_data, num_faqs=10)

    if not faq_items:
        print("FAQ 생성을 실패했습니다. 프로그램을 종료합니다.")
        return

    # 3. FAQ 데이터 저장
    print("FAQ 데이터를 파일로 저장 중...")
    save_faq_to_json(faq_items, "faq.json")
    save_faq_to_pickle(faq_items, "faq.pkl")

    # 4. FAISS 벡터스토어 생성 및 저장
    print("FAISS 벡터스토어 생성 중...")
    generate_faiss_vector(faq_items)

    # 5. FAISS 데이터베이스 로드
    print("FAISS 데이터베이스 로드 중...")
    faiss_db = load_faiss_database("faiss_db.index")

    # 6. 사용자 질문 처리
    print("사용자 질문에 대한 답변 생성 준비 완료.")
    while True:
        user_question = input("\n질문을 입력하세요 (종료하려면 'exit' 입력): ").strip()
        if user_question.lower() == "exit":
            print("프로그램을 종료합니다.")
            break

        # 7. 유사 질문 검색
        indices, distances = search_similar_question(user_question, faiss_db, k=3)

        if distances[0] < 0.5:  # 임계값: 유사도 0.5 미만
            print("유사한 질문을 찾을 수 없습니다. 답변을 생성합니다...")
            answer = generate_answer(user_question, matched_question=None, matched_answer=None)
        else:
            # 가장 유사한 질문 및 답변 매칭
            with open("faq_items.pkl", "rb") as f:
                faq_data = pickle.load(f)
            matched_question = faq_data[indices[0]]["question"]
            matched_answer = faq_data[indices[0]]["answer"]

            print(f"매칭된 질문: {matched_question}")
            print(f"매칭된 답변: {matched_answer}")
            answer = generate_answer(user_question, matched_question, matched_answer)

        # 8. 최종 답변 출력
        print(f"\nAI 답변: {answer}")

if __name__ == "__main__":
    asyncio.run(main())
