import math
from typing import Any, Dict, List


class RetrievalEvaluator:
    """
    Hệ thống đánh giá hiệu suất của Vector Database/Retrieval Stage.
    Cung cấp các chỉ số chuẩn: Hit Rate, MRR, và NDCG.
    """

    def __init__(self):
        pass

    def calculate_hit_rate(
        self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3
    ) -> float:
        """
        Tính toán Hit Rate @K.
        Trả về 1.0 nếu ít nhất một ID kỳ vọng nằm trong Top K tài liệu được lấy ra.
        """
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Tính toán Mean Reciprocal Rank (MRR).
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids (1-indexed).
        MRR = 1 / vị trí.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def calculate_ndcg(
        self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3
    ) -> float:
        """
        Tính toán Normalized Discounted Cumulative Gain (NDCG) @K.
        Giả định độ liên quan nhị phân (1 cho hit, 0 cho miss).
        """
        actual_relevance = [
            1.0 if doc_id in expected_ids else 0.0 for doc_id in retrieved_ids[:top_k]
        ]
        
        # DCG = sum(rel_i / log2(i + 1 + 1))
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(actual_relevance))
        
        # IDCG (Lý tưởng): Sắp xếp tất cả các hit lên đầu
        hits = sum(actual_relevance)
        if hits == 0:
            return 0.0
            
        idcg = sum(1.0 / math.log2(i + 2) for i in range(int(hits)))
        
        return dcg / idcg

    async def evaluate_batch(self, dataset: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, float]:
        """
        Đánh giá toàn bộ bộ dữ liệu và trả về các chỉ số trung bình.
        
        Args:
            dataset: Danh sách các dict, mỗi dict chứa 'expected_ids' và 'retrieved_ids'.
            top_k: Số lượng tài liệu hàng đầu để tính toán Hit Rate và NDCG.
        """
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "avg_ndcg": 0.0}

        hit_rates = []
        mrr_scores = []
        ndcg_scores = []

        for item in dataset:
            # Hỗ trợ nhiều cách đặt tên key để tăng tính tương thích
            expected_ids = (
                item.get("expected_ids")
                or item.get("ground_truth_ids")
                or item.get("expected_retrieval_ids")
                or []
            )
            retrieved_ids = item.get("retrieved_ids", [])

            hit_rates.append(self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=top_k))
            mrr_scores.append(self.calculate_mrr(expected_ids, retrieved_ids))
            ndcg_scores.append(self.calculate_ndcg(expected_ids, retrieved_ids, top_k=top_k))

        return {
            "avg_hit_rate": sum(hit_rates) / len(hit_rates),
            "avg_mrr": sum(mrr_scores) / len(mrr_scores),
            "avg_ndcg": sum(ndcg_scores) / len(ndcg_scores),
        }


if __name__ == "__main__":
    import asyncio

    evaluator = RetrievalEvaluator()

    # --- Test Cases ---
    test_expected = ["doc1"]
    test_retrieved = ["doc2", "doc1", "doc3"] # Hit ở vị trí 2

    print("--- Testing Single Case ---")
    print(f"Hit@3: {evaluator.calculate_hit_rate(test_expected, test_retrieved, top_k=3)}")
    print(f"MRR:   {evaluator.calculate_mrr(test_expected, test_retrieved)}")
    print(f"NDCG@3: {evaluator.calculate_ndcg(test_expected, test_retrieved, top_k=3):.4f}")

    # --- Testing Batch ---
    dataset = [
        {"expected_ids": ["doc1"], "retrieved_ids": ["doc1", "doc2"]}, # Rank 1
        {"expected_ids": ["doc2"], "retrieved_ids": ["doc1", "doc2"]}, # Rank 2
        {"expected_ids": ["doc3"], "retrieved_ids": ["doc1", "doc2"]}, # Miss
    ]
    
    async def run_test():
        results = await evaluator.evaluate_batch(dataset)
        print("\n--- Testing Batch ---")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
            
    asyncio.run(run_test())
