import pickle
import config

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from download_data import get_data
from inference import inference_json




def main():
    # html -> text -> faq
    product_detail = get_data()
    result_text = inference_json(product_detail)

    result = result_text["qa_list"] 
    print(result)

    with open("qas.pkl", "wb") as f:
        pickle.dump(result, f)

    result_questions = [row['question'] for row in result]

    db = FAISS.from_texts(
        result_questions,
        embedding=OpenAIEmbeddings(api_key=config.OPENAI_API_KEY),
        metadatas=result
    )

    db.save_local("qas.index")


if __name__ == '__main__':
    main()