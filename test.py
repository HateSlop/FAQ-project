import openai
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 키 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")

# 테스트 함수
def test_openai_api():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 사용할 모델
            messages=[
                {"role": "user", "content": "Hello! Can you confirm this API is working?"}
            ],
            max_tokens=50,
            temperature=0.7
        )
        # 응답 출력
        print("응답 내용:", response['choices'][0]['message']['content'])
    except Exception as e:
        print("오류 발생:", e)

# 테스트 실행
if __name__ == "__main__":
    test_openai_api()
