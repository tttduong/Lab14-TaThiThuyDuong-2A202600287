# -*- coding: utf-8 -*-
import asyncio
import json
import os
import random
import time
from typing import Any, Dict, List

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


async def generate_qa_batch_with_retry(
    client: OpenAI, text: str, batch_size: int, batch_idx: int, max_retries: int = 3
) -> List[Dict[str, Any]]:
    """
    Sinh một nhóm QA pairs với cơ chế thử lại (Retry) nếu gặp lỗi API hoặc JSON.
    """
    for attempt in range(max_retries):
        try:
            print(f"Generating batch {batch_idx} (attempt {attempt + 1}/{max_retries})...")
            
            # Thay đổi seed/nhiễu nhẹ bằng cách thêm hướng dẫn đa dạng cho mỗi batch
            diversity_hints = [
                "Tập trung vào các câu hỏi tính toán và chỉ số (Hit Rate, MRR, NDCG).",
                "Tập trung vào các khái niệm hallucination và cách phòng tránh.",
                "Tập trung vào sự khác biệt giữa các framework RAGAS, LangChain.",
                "Hỏi về quy trình thực tế khi triển khai Benchmarking trong doanh nghiệp.",
                "Tập trung vào các trường hợp biên (Edge cases) và lỗi hệ thống."
            ]
            hint = diversity_hints[batch_idx % len(diversity_hints)]

            prompt = f"""
Dựa trên tài liệu kỹ thuật về AI Evaluation dưới đây:

{text}

Nhiệm vụ: Sinh ra {batch_size} cặp (Câu hỏi, Trả lời, Ngữ cảnh) chất lượng cao.
Yêu cầu chuyên môn:
1. Mỗi cặp gồm: 'question', 'expected_answer', 'context', 'expected_ids'.
2. ĐỘ ĐA DẠNG: {hint} Đảm bảo không trùng lặp các câu hỏi đã sinh ở batch trước.
3. RED TEAMING: Ít nhất 1-2 câu hỏi trong mỗi đợt phải là 'Adversarial' (hỏi về thông tin sai lệch hoặc không có trong text) để test tính trung thực (Faithfulness).
4. KHUNG DỮ LIỆU: Chỉ trả về JSON array. Không kèm giải thích.
5. NGÔN NGỮ: Tiếng Việt chuyên nghiệp, tự nhiên.

Ví dụ format:
[
  {{
    "question": "MRR là gì và tại sao nó quan trọng?",
    "expected_answer": "MRR (Mean Reciprocal Rank) là chỉ số đo lường vị trí của tài liệu đúng đầu tiên. Nó quan trọng vì phản ánh trải nghiệm người dùng khi tìm kiếm.",
    "context": "Mean Reciprocal Rank (MRR) tập trung vào vị trí của tài liệu liên quan đầu tiên...",
    "expected_ids": ["doc_eval_01"]
  }}
]
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia AI Engineering hàng đầu, chuyên tạo dữ liệu benchmark chuẩn quốc tế."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2500,
                temperature=0.8, # Tăng temperature để đa dạng hóa câu trả lời
            )

            content = response.choices[0].message.content or ""
            payload = _clean_json_payload(content)
            dataset = json.loads(payload)

            if isinstance(dataset, list) and len(dataset) > 0:
                return dataset[:batch_size]
            
        except Exception as e:
            print(f"⚠️ Batch {batch_idx} failed (Attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.random()
                time.sleep(wait_time)
            else:
                print(f"❌ Failed batch {batch_idx} after {max_retries} attempts.")
                
    return []


async def generate_qa_from_text(text: str, total_pairs: int = 50, batch_size: int = 10) -> List[Dict]:
    client = _get_client()
    all_pairs: List[Dict] = []
    batch_idx = 1

    while len(all_pairs) < total_pairs:
        remaining = total_pairs - len(all_pairs)
        current_batch_size = min(batch_size, remaining)
        
        batch_pairs = await generate_qa_batch_with_retry(
            client, text, current_batch_size, batch_idx
        )
        
        if not batch_pairs:
            # Nếu một batch hoàn toàn thất bại sau các lần thử, dừng lại để tránh loop vô hạn
            print("Stopping due to consecutive failures.")
            break

        all_pairs.extend(batch_pairs)
        print(f"✅ Batch {batch_idx} hoàn tất: +{len(batch_pairs)} QA (Tổng {len(all_pairs)}/{total_pairs})")
        batch_idx += 1
        
        # Nghỉ ngắn giữa các batch để tránh Rate Limit của OpenAI
        await asyncio.sleep(1)

    return all_pairs[:total_pairs]


async def main() -> None:
    # Tài liệu kỹ thuật chi tiết hơn để AI sinh câu hỏi chất lượng
    enriched_text = """
Evaluating a Retrieval-Augmented Generation (RAG) system requires assessing both the retrieval stage and the generation stage.
1. Retrieval Metrics:
- Hit Rate (@K): Checks if at least one relevant document appears in the top K results.
- Mean Reciprocal Rank (MRR): Position of the first relevant document (1/rank). High MRR means correct info is at the top.
- Normalized Discounted Cumulative Gain (nDCG): Measures ranking quality by rewarding highly relevant documents at the top. It handles graded relevance.
2. RAGAS Framework:
- Faithfulness: Does the answer derive only from the context? Detects hallucinations.
- Answer Relevancy: How pertinent is the answer to the query?
- Context Precision & Recall: Precision measures the ratio of relevant chunks. Recall measures if all necessary info was retrieved.
- Answer Correctness: Compares the result against ground truth labels.
Best practices involve creating a 'Golden Dataset' of Q/A/Context triplets. Systems should be evaluated for latency and cost efficiency alongside accuracy.
Red Teaming involves adversarial prompts like prompt injection or asking out-of-context questions to test system guardrails.
    """

    print("🚀 Bắt đầu quy trình sinh Golden Dataset mới...")
    qa_pairs = await generate_qa_from_text(enriched_text, total_pairs=50, batch_size=10)

    if not qa_pairs:
        print("❌ Không tạo được dữ liệu. Vui lòng kiểm tra API Key hoặc kết nối mạng.")
        return

    output_path = "data/golden_set.jsonl"
    os.makedirs("data", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            
    print(f"\n✨ THÀNH CÔNG! Đã ghi {len(qa_pairs)} cặp QA vào {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
