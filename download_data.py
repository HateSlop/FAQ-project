from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
def get_data():
  url = "https://namu.wiki/w/%EB%8C%80%ED%95%99%EC%88%98%ED%95%99%EB%8A%A5%EB%A0%A5%EC%8B%9C%ED%97%98?from=%EC%88%98%EB%8A%A5"

  loader = AsyncHtmlLoader(url, verify_ssl=False)
  docs = loader.load()

  html2text = Html2TextTransformer()
  docs_transformed = html2text.transform_documents(docs)

  content = docs_transformed[0].page_content
  return content

if __name__ == '__main__':
  content = get_data()
