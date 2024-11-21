import html2text
import aiohttp

# 비동기적으로 HTML 데이터 로드
async def fetch_html(url):
    headers = {"User-Agent": "MyCrawler/1.0"}  # User-Agent 설정
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            return await response.text()

# HTML을 텍스트로 변환
def html_to_text(html_content):
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True  # 링크 포함 여부
    text_maker.ignore_images = True  # 이미지 태그 무시
    return text_maker.handle(html_content)

