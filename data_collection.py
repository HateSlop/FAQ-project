from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
import asyncio


async def load_and_transform(url: str) -> str:
    """
    Args:
        url (str): 웹 페이지 URL.

    Returns:
        str: 텍스트로 변환된 웹 페이지 내용.
    """
    try:
        # HTML 데이터 로드
        loader = AsyncHtmlLoader(
            [url],
            header_template={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            }
        )
        print(f"URL에서 HTML 데이터를 로드 중: {url}")
        docs = await loader.aload()  # 비동기 방식으로 HTML 로드

        # HTML 데이터를 텍스트로 변환
        transformer = Html2TextTransformer()
        transformed_docs = transformer.transform_documents(docs)
        plain_text = transformed_docs[0].page_content
        print("텍스트로 변환 완료.")

        return plain_text
    except Exception as e:
        print(f"데이터 로드 및 변환 중 오류 발생: {e}")
        return ""
