from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from prompt_template import prompt_template_question
import config

api_key = config.OPENAI_API_KEY
client= OpenAI(api_key=api_key)

def search(question):
    db = FAISS.load_local(
        "qas.index",
        OpenAIEmbeddings(openai_api_key=api_key),
        allow_dangerous_deserialization=True
    )

    result = db.search(question, search_type="similarity")

    return result[0].metadata

def generate_answer(context, question):
    context_join = f"""Q: {context['question']}
A: {context['answer']}"""
    prompt = prompt_template_question.format(context=context_join, question=question)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    output = response.choices[0].message.content
    return output

if __name__ == '__main__':
    question = "수능과 훔바 댄스의 관계에 대해서 알려줘?"
    qa = search(question)
    # print(qa['question'])
    # print(qa['answer'])
    print()
    print(question)
    print(generate_answer(qa, question))