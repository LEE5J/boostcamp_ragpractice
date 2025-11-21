import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.models import CountRequest

# -------------------------
# 1. 임베딩 모델 / Qdrant 클라이언트 초기화
# -------------------------
MODEL_NAME = "google/embeddinggemma-300m"

embed_model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(host="localhost", port=6333)

count = client.count(
    collection_name="park_data",
    exact=True
).count

# print("documents 컬렉션 point 개수:", count)

def embed_query(text: str) -> np.ndarray:
    vec = embed_model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    return vec

# -------------------------
# 2. 검색 함수
# -------------------------
def search_parks(query: str, region: str | None = None, top_k: int = 5):
    """
    query: 사용자 자연어 질의
    region: "성동구", "강남구" 등 특정 지역으로 필터링하고 싶을 때
    top_k: 몇 개까지 가져올지
    """
    q_vec = embed_query(query)

    must_conditions = []

    if region is not None:
        must_conditions.append(
            FieldCondition(
                key="extra_meta.region",
                match=MatchValue(value=region),
            )
        )

    q_filter = Filter(must=must_conditions)

    results = client.query_points(
        collection_name="documents",
        query=q_vec,
        query_filter=q_filter,
        limit=top_k,
    ).points

    return results

# -------------------------
# 3. 테스트 실행
# -------------------------
if __name__ == "__main__":
    user_query = "성동구 근처에서 강이 보이고 조용하게 산책하기 좋은 큰 공원 추천해줘."
    results = search_parks(user_query, region="성동구", top_k=5)

    for i, r in enumerate(results, start=1):
        p = r.payload
        title = p.get("title")
        extra_meta = p.get("extra_meta", {})
        region = extra_meta.get("region", "지역 정보 없음")
        text_preview = (p.get("text") or "")[:120].replace("\n", " ")
        print(f"[{i}] score={r.score:.4f} | {title} ({region})")
        print(f"    {text_preview}...")
        print()
