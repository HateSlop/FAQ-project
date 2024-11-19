from openai import OpenAI
from pydantic import BaseModel
import json
import config

class QA(BaseModel):
    question: str
    answer: str

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

def chunk_text(text, max_length=3000):
    """텍스트를 청크로 나눕니다."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def generate_faqs(text_data):
    chunks = chunk_text(text_data)
    all_faqs = []
    
    for i, chunk in enumerate(chunks[:3]):  # 처음 3개 청크만 처리
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """당신은 교육 전문가입니다. 
                    주어진 텍스트를 바탕으로 FAQ를 JSON 형식으로 생성해주세요."""},
                    {"role": "user", "content": f"""다음 텍스트를 바탕으로 5개의 FAQ를 생성해주세요.
                    반드시 다음의 JSON 형식으로 출력해주세요:
                    
                    텍스트: {chunk}
                    
                    출력 형식:
                    [
                        {{"question": "첫 번째 질문", "answer": "첫 번째 답변"}},
                        {{"question": "두 번째 질문", "answer": "두 번째 답변"}},
                        ...
                    ]"""}
                ],
                temperature=0.7
            )
            
            # 응답 출력하여 디버깅
            print(f"\nAPI Response for chunk {i+1}:")
            print(response.choices[0].message.content)
            
            # JSON 파싱
            faqs_json = json.loads(response.choices[0].message.content)
            chunk_faqs = [QA(**qa) for qa in faqs_json]
            all_faqs.extend(chunk_faqs)
            print(f"Chunk {i+1}/{len(chunks[:3])} processed: Generated {len(chunk_faqs)} FAQs")
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in chunk {i+1}: {e}")
            continue
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            continue
    
    # 중복 제거
    unique_faqs = []
    seen_questions = set()
    for faq in all_faqs:
        if faq.question not in seen_questions:
            seen_questions.add(faq.question)
            unique_faqs.append(faq)
    
    return unique_faqs

def main():
    try:
        # 텍스트 데이터 로드
        with open('news-project/text_data.txt', 'r', encoding='utf-8') as f:
            text_data = f.read()
        
        # FAQ 생성
        faqs = generate_faqs(text_data)
        print(f"\nTotal unique FAQs generated: {len(faqs)}")
        
        # 결과 저장
        with open('news-project/faqs.json', 'w', encoding='utf-8') as f:
            json.dump([{"question": faq.question, "answer": faq.answer} for faq in faqs], 
                     f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()