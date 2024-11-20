from utils import get_embeddings
import faiss
import numpy as np
import pickle

def generate_vec():
    # 저장된 파일 경로
    pickle_file_path = "./res/FAQ_data.pkl"

    # Pickle 파일에서 데이터를 불러오기
    with open(pickle_file_path, "rb") as f:
        structured_faq = pickle.load(f)

    questions = [item.question for item in structured_faq.faqs]

    embqs = np.array(get_embeddings(questions))

    index = faiss.IndexIDMap(faiss.IndexFlatIP(embqs.shape[1]))
    ids = np.array([item.index for item in structured_faq.faqs])
    index.add_with_ids(embqs, ids)

    faiss.write_index(index, './res/FAQ_vec.index')
    print("벡터 생성 완료")
    return index