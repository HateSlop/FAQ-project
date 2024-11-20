from qna_collection import url_loader, save_as_json_and_pickle
from vectorization import generate_vector
from utils import generate_answer, prompt
from make_final_prompt import make_final_prompt

url = ["https://namu.wiki/w/%EB%8C%80%ED%95%99%EC%88%98%ED%95%99%EB%8A%A5%EB%A0%A5%EC%8B%9C%ED%97%98"]

def setting(url):
    url = url
    #Qna Collection
    docs_transformed = url_loader(url)
    qna = generate_answer(prompt + str(docs_transformed), model= "gpt-4o")
    save_as_json_and_pickle(qna)
    #Collection to vector
    index = generate_vector()
    return index

def main(index):
    query = input("질문을 입력하세요:")
    final_prompt = make_final_prompt(query, index)
    answer = generate_answer(final_prompt,model="gpt-4o-mini")
    print(answer)
    return answer

index = setting(url)
main(index)
