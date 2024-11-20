from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle

def vectorize_faq(faq_items, embedding_model):
    # FAQ 질문 및 답변 추출
    questions = [item.question for item in faq_items]
    answers = [item.answer for item in faq_items]
    # 메타데이터 생성
    metadatas = [{'answer': a} for a in answers]
    # FAISS 벡터스토어 생성 (임베딩 생성 포함)
    faiss_db = FAISS.from_texts(questions, embedding=embedding_model, metadatas=metadatas)
    # 데이터베이스와 FAQ 데이터 저장
    with open('faiss_db.pkl', 'wb') as f:
        pickle.dump(faiss_db, f)
    with open('faq_items.pkl', 'wb') as f:
        pickle.dump(faq_items, f)

    print("벡터스토어 및 FAQ 데이터 저장 완료")
