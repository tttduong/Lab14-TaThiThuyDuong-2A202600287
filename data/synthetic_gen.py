# -*- coding: utf-8 -*-
import json
import asyncio
import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_qa_from_text(text: str, num_pairs: int = 50) -> List[Dict]:
    print(f"Generating {num_pairs} QA pairs from text using LLM...")
    
    prompt = f"""
Dựa trên văn bản sau đây về AI Evaluation Benchmarking:

{text}

Hãy sinh ra {num_pairs} cặp câu hỏi - câu trả lời chất lượng. Mỗi cặp phải bao gồm:
- question: Câu hỏi từ người dùng, đa dạng và tự nhiên.
- expected_answer: Câu trả lời lý tưởng (Ground Truth), chính xác và dựa trên văn bản.
- context: Đoạn văn bản ngắn chứa câu trả lời (không quá 200 từ).
- expected_ids: Danh sách ID tài liệu/chunk giả (ví dụ: ["doc_1", "chunk_5"]).

Yêu cầu:
- Ít nhất 5 câu hỏi phải là "Red Teaming": hỏi về thông tin không có trong văn bản để kiểm tra khả năng hallucination của hệ thống RAG.
- Các câu hỏi khác phải dựa trực tiếp trên nội dung văn bản.
- Trả về CHỈ JSON array, không có text giải thích khác. Mỗi phần tử là một object với các key trên.
- Đảm bảo câu trả lời bằng tiếng Việt có dấu.

Ví dụ format (chỉ trả về array này, không có markdown):
[
  {{
    "question": "AI Evaluation là gì?",
    "expected_answer": "AI Evaluation là quy trình đánh giá hiệu suất của các hệ thống trí tuệ nhân tạo.",
    "context": "AI Evaluation Benchmarking là một quy trình kỹ thuật quan trọng trong việc đánh giá hiệu suất của các hệ thống trí tuệ nhân tạo...",
    "expected_ids": ["doc_1"]
  }},
  ...
]
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Hoặc model khác nếu cần
            messages=[
                {"role": "system", "content": "Bạn là một chuyên gia tạo dữ liệu tổng hợp cho đánh giá AI."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.7
        )
        
        content = response.choices[0].message.content.strip()
        # Loại bỏ markdown nếu có
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        dataset = json.loads(content)
        print(f"Generated {len(dataset)} QA pairs.")
        return dataset[:num_pairs]  # Đảm bảo không quá num_pairs
    
    except Exception as e:
        print(f"Error generating QA pairs: {e}")
        # Fallback to dummy data
        return [
            {
                "question": f"Câu hỏi mẫu {i}",
                "expected_answer": f"Câu trả lời mẫu {i}",
                "context": "Đoạn văn mẫu.",
                "expected_ids": [f"doc_{i}"]
            } for i in range(1, num_pairs + 1)
        ]

async def main():
    raw_text = """
AI Evaluation Benchmarking là quy trình đánh giá hiệu suất của hệ thống trí tuệ nhân tạo, đặc biệt là mô hình ngôn ngữ lớn và hệ thống RAG. 

Quy trình bao gồm tạo Golden Dataset để đo lường độ chính xác, tin cậy và khả năng chống red teaming. Dataset gồm hàng trăm cặp câu hỏi - câu trả lời, tạo tự động bằng AI.

Đánh giá retrieval: tìm kiếm thông tin từ kho dữ liệu lớn. Chỉ số: Hit Rate, MRR, NDCG.

Red teaming: kiểm tra bảo mật, tạo câu hỏi đánh lừa để phát hiện hallucination.

Công cụ: RAGAS, LangChain, framework đánh giá tự động.

Benchmarking cải thiện chất lượng, đảm bảo công bằng khi so sánh mô hình AI.
"""
    
    qa_pairs = await generate_qa_from_text(raw_text, 50)
    
    os.makedirs("data", exist_ok=True)
    with open("data/golden_set.jsonl", "a", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Hoàn thành! Đã thêm {len(qa_pairs)} cặp QA vào data/golden_set.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
