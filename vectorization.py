#from utils import get_embeddings
import faiss
import numpy as np
import pickle
import config
from openai import OpenAI

OPENAI_API_KEY = config.OPENAI_API_KEY
openai_client = OpenAI(api_key=OPENAI_API_KEY)
model = "gpt-4o"

def get_embeddings(text, model='text-embedding-3-small'):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=text,
        model=model
    )
    output = []
    for i in range(len(response.data)):
        output.append(response.data[i].embedding)
    return output

def generate_vector():
    # 저장된 파일 경로
    pickle_file_path = "res/qna_items.pkl"

    # Pickle 파일에서 데이터를 불러오기
    with open(pickle_file_path, "rb") as f:
        qna_as_list = pickle.load(f)

    questions = [item.question for item in qna_as_list.qnas]

    embedding_questions = np.array(get_embeddings(questions))

    index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_questions.shape[1]))
    ids = np.array([item.index for item in qna_as_list.qnas])
    index.add_with_ids(embedding_questions, ids)

    faiss.write_index(index, 'res/qna_vec.index')
    print("벡터 생성 완료")
    return index
