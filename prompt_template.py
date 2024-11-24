prompt_template = """다음 내용을 읽고 FAQ를 열 가지 이상 만들어 주세요. 

```
{product_detail}
```
"""

prompt_template_json = """다음 내용을 읽고 FAQ를 10개 만들어주세요.

{format_instructions}

```
{product_detail}
```
"""

prompt_template_question = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: korean
"""