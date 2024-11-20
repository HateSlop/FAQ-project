from langchain.document_loaders import UnstructuredURLLoader

def load_and_transform(url):
    # UnstructuredURLLoader를 사용하여 URL의 콘텐츠 로드
    loader = UnstructuredURLLoader(urls=[url])
    documents = loader.load()
    # 텍스트 추출
    text = documents[0].page_content
    return text