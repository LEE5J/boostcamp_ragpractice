import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

with open('../data/docs_emb.json', 'r', encoding='utf-8') as f:
    docs = json.load(f)

embeddings = np.load('../data/embeddings_.npy')

EMBED_DIM = embeddings.shape[1]

## Qdrant Client 생성
client = QdrantClient(host='localhost', port=6333)
## Collection 생성
client.recreate_collection(
    collection_name='park_data',
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
)

## Qdrant 에 데이터 적재
points = []
for i, (doc, vec) in enumerate(zip(docs, embeddings), start = 1):
    # payload 구성
    payload = {
        "doc_type": doc.get("doc_type"),
        "ref_id": i,
        "title": doc.get("title"),
        "text": doc.get("text"),
        "tags": doc.get("tags", []),
        "extra_meta": doc.get("extra_meta", {}),
    }

    extra = doc.get("extra_meta", {})
    region = extra.get("region")

    point = PointStruct(
        id=i,
        vector=vec.tolist(),
        payload=payload
    )

    points.append(point)

client.upsert(
    collection_name='park_data',
    points=points
)

print(f"Upserted {len(points)} points into 'park_data' collection.")