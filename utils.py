from openai import OpenAI
import config

OPENAI_API_KEY = config.OPENAI_API_KEY
openai_client = OpenAI(api_key=OPENAI_API_KEY)

prompt = """다음에 오는 텍스트에서 FAQ 형태의 질문과 답변을 10개 이상 생성하시오.
생성된 FAQ는 검색 시스템에서 사용될 수 있도록 
반드시 다음의 출력 포맷을 맞추어 문자열의 리스트 형태로 구조화하시오.
이 외에 다른 텍스트를 절대 생성하지 마시오. ```json으로 시작하지도 마시오.
출력 포맷:
[{"index": <인덱스 번호>, "question": <질문>, "answer": <답변>},...]
"""

def generate_answer(prompt, model):
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."},
        {
            "role": "system",
            "content": prompt
        }]
    response = openai_client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer

def get_embedding(text, model='text-embedding-3-small'):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


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


