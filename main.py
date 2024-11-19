import asyncio
import data_collection
import faq_generation

async def main():
    # 데이터 수집
    await data_collection.main()
    
    # FAQ 생성
    faq_generation.main()

if __name__ == "__main__":
    asyncio.run(main())