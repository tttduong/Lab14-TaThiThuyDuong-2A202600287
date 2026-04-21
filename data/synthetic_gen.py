# -*- coding: utf-8 -*-
import asyncio
import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY. Create .env and add OPENAI_API_KEY=your_key."
        )
    return OpenAI(api_key=api_key)


def _clean_json_payload(content: str) -> str:
    payload = content.strip()
    if payload.startswith("```json"):
        payload = payload[7:]
    elif payload.startswith("```"):
        payload = payload[3:]
    if payload.endswith("```"):
        payload = payload[:-3]
    return payload.strip()


async def generate_qa_batch(client: OpenAI, text: str, batch_size: int, batch_idx: int) -> List[Dict]:
    print(f"Generating batch {batch_idx} ({batch_size} questions)...")
    prompt = f"""
Dựa trên văn bản sau đây về AI Evaluation Benchmarking:

{text}

Hãy sinh ra {batch_size} cặp câu hỏi - câu trả lời chất lượng. Mỗi cặp phải bao gồm:
- question: Câu hỏi từ người dùng, đa dạng và tự nhiên.
- expected_answer: Câu trả lời lý tưởng (Ground Truth), chính xác và dựa trên văn bản.
- context: Đoạn văn bản ngắn chứa câu trả lời (không quá 200 từ).
- expected_ids: Danh sách ID tài liệu/chunk giả (ví dụ: ["doc_1", "chunk_5"]).

Yêu cầu:
- Tạo ít nhất 1 câu "Red Teaming" trong batch này: hỏi thông tin không có trong văn bản để kiểm tra hallucination.
- Các câu hỏi còn lại phải dựa trực tiếp vào văn bản.
- Trả về CHỈ JSON array, không có text giải thích khác.
- Đảm bảo tiếng Việt có dấu.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là chuyên gia tạo dữ liệu tổng hợp cho đánh giá AI."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1800,
        temperature=0.7,
    )

    content = response.choices[0].message.content or ""
    payload = _clean_json_payload(content)
    try:
        dataset = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Batch {batch_idx} returned invalid JSON: {exc}") from exc

    if not isinstance(dataset, list):
        raise ValueError(f"Batch {batch_idx} did not return a valid JSON array.")
    return dataset[:batch_size]


async def generate_qa_from_text(text: str, total_pairs: int = 50, batch_size: int = 10) -> List[Dict]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if total_pairs <= 0:
        raise ValueError("total_pairs must be greater than 0.")

    client = _get_client()
    all_pairs: List[Dict] = []
    batch_idx = 1

    while len(all_pairs) < total_pairs:
        remaining = total_pairs - len(all_pairs)
        current_batch_size = min(batch_size, remaining)
        batch_pairs = await generate_qa_batch(client, text, current_batch_size, batch_idx)
        all_pairs.extend(batch_pairs)
        print(f"Batch {batch_idx} done: +{len(batch_pairs)} QA (total {len(all_pairs)}/{total_pairs})")
        batch_idx += 1

    return all_pairs[:total_pairs]


async def main() -> None:
    raw_text = """
AI Evaluation Benchmarking là quy trình đánh giá hiệu suất của hệ thống trí tuệ nhân tạo, đặc biệt là mô hình ngôn ngữ lớn và hệ thống RAG.

Quy trình bao gồm tạo Golden Dataset để đo lường độ chính xác, tin cậy và khả năng chống red teaming. Dataset gồm hàng trăm cặp câu hỏi - câu trả lời, tạo tự động bằng AI.

Đánh giá retrieval: tìm kiếm thông tin từ kho dữ liệu lớn. Chỉ số: Hit Rate, MRR, NDCG.

Red teaming: kiểm tra bảo mật, tạo câu hỏi đánh lừa để phát hiện hallucination.

Công cụ: RAGAS, LangChain, framework đánh giá tự động.

Benchmarking cải thiện chất lượng, đảm bảo công bằng khi so sánh mô hình AI.
"""

    qa_pairs = await generate_qa_from_text(raw_text, total_pairs=50, batch_size=10)

    output_path = "data/golden_set.jsonl"
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Done! Wrote {len(qa_pairs)} QA pairs to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
