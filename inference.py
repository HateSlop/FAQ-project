import config
import json
from typing import List

from openai import OpenAI
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser  

from download_data import get_data
from prompt_template import prompt_template, prompt_template_json

client = OpenAI(api_key=config.OPENAI_API_KEY)
def inference(product_detail):
  prompt = prompt_template.format(product_detail=product_detail)
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
      "role": "system", 
      "content": "You are a helpful assistant."
    }, {
      "role": "user", 
      "content": prompt
    }], 
    temperature=0
  )
  output = response.choices[0].message.content
  return output

class QA(BaseModel):
  question: str
  answer: str

class Output(BaseModel):
  qa_list: List[QA]

output_parser = PydanticOutputParser(pydantic_object=Output)

def inference_json(product_detail):
  prompt = prompt_template_json.format(
    format_instructions=output_parser.get_format_instructions(),
    product_detail=product_detail
  )
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
      "role": "system", 
      "content": "You are a helpful assistant."
    }, {
      "role": "user", 
      "content": prompt
    }], 
    temperature=0, 
    response_format={"type": "json_object"}
  )
  output = response.choices[0].message.content
  output_json = json.loads(output)
  return output_json

if __name__ == '__main__':
  product_detail = get_data()
  result = inference_json(product_detail)
  print(json.dumps(result, indent=2, ensure_ascii=False))
