import asyncio
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from bs4 import BeautifulSoup

async def load_and_transform():
    url = "https://namu.wiki/w/대학수학능력시험"
    
    try:
        # HTML 로드
        loader = AsyncHtmlLoader(
            [url],  # 리스트 형태로 전달
            requests_per_second=2
        )
        docs = await loader.aload()
        
        if not docs:
            raise Exception("No documents loaded")
        
        # HTML을 텍스트로 변환
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        
        # 텍스트 정제
        cleaned_text = ""
        for doc in docs_transformed:
            soup = BeautifulSoup(doc.page_content, 'html.parser')
            
            # 불필요한 태그 제거
            for tag in soup(['script', 'style', 'meta', 'link', 'header', 'footer', 'nav']):
                tag.decompose()
            
            # 텍스트 추출 및 정제
            text = soup.get_text(separator='\n', strip=True)
            cleaned_text += '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        
        # 정제된 텍스트 저장
        with open('news-project/text_data.txt', 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
            
        print("Text extraction completed successfully")
        
    except Exception as e:
        print(f"Error during text extraction: {e}")

async def main():
    await load_and_transform()

if __name__ == "__main__":
    asyncio.run(main())