"""
벡터 DB에서 정보를 가져오는 테스트 코드
직접 정의한 쿼리 문장으로 임베딩 검색을 수행합니다.
"""
from typing import List, Dict

from pymilvus import (
    connections,
    Collection,
)
from sentence_transformers import SentenceTransformer
import torch


# 설정 (insert2db.py와 동일)
MILVUS_HOST = "192.168.50.20"
MILVUS_PORT = "19530"
COLLECTION_NAME = "legal_documents"
EMBEDDING_MODEL = "google/embeddinggemma-300m"
DIMENSION = 768
USE_CPU = False  # True로 설정하면 CPU 사용

# 테스트할 쿼리 문장 (직접 정의)
TEST_QUERIES = [
    "동업 조합 해산 후, 일방 청산인이 청산 절차에 비협조한다는 이유로 다른 청산인이 그에 대한 직무집행정지 가처분을 신청한 사건입니다. 법원은 민법상 '조합'의 청산인은 법인의 청산인과 달리 법원에 해임을 청구할 법적 근거가 없으므로, 해당 가처분 신청은 부적법하다고 보아 각하했습니다.",
    "배당요구신청서라는 제목이나 인지를 첩부하지 않고 채권계산서만 제출했어도, 채권의 원인과 수액이 명시되었다면 적법한 배당요구로 보아야 한다고 판시했습니다. 또한, 부동산 압류 효력이 발생한 이후에 근저당권 등기를 마친 담보물권자도 별도의 가압류 절차 없이, 압류채권자와 채권액에 비례하여 평등하게 배당받을 수 있다",

]


def connect_to_milvus():
    """Milvus에 연결합니다."""
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        print(f"✓ Milvus에 연결되었습니다: {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        print(f"✗ Milvus 연결 실패: {e}")
        raise


def search_similar_chunks(collection: Collection, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
    """
    Milvus에서 유사한 청크를 검색합니다.
    
    Args:
        collection: Milvus 컬렉션 객체
        query_embedding: 검색 쿼리 임베딩 벡터
        top_k: 반환할 상위 결과 개수
        
    Returns:
        검색 결과 리스트
    """
    # 컬렉션이 로드되지 않았으면 로드
    if not collection.has_index():
        print("⚠️  컬렉션에 인덱스가 없습니다.")
        return []
    
    try:
        collection.load()
    except Exception:
        pass
    
    # 검색 파라미터
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    
    # 검색 수행
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["file_path", "category", "name", "chunk_text", "chunk_index"]
    )
    
    # 결과 파싱 (원본 데이터 그대로 저장)
    search_results = []
    if results and len(results) > 0:
        for hit in results[0]:
            # 원본 데이터 그대로 저장
            search_results.append({
                "id": hit.id,
                "distance": hit.distance,
                "file_path": hit.entity.get("file_path", ""),
                "category": hit.entity.get("category", ""),
                "name": hit.entity.get("name", ""),
                "chunk_text": hit.entity.get("chunk_text", ""),
                "chunk_index": hit.entity.get("chunk_index", -1),
            })
    
    return search_results


def main():
    """메인 함수"""
    print("=" * 80)
    print("벡터 DB 검색 테스트")
    print("=" * 80)
    
    # 1. Milvus 연결
    print("\n[1/4] Milvus 연결 중...")
    connect_to_milvus()
    
    # 2. 컬렉션 로드
    print("\n[2/4] 컬렉션 로드 중...")
    try:
        collection = Collection(COLLECTION_NAME)
        collection.load()
        print(f"✓ 컬렉션 '{COLLECTION_NAME}' 로드 완료")
        print(f"  총 벡터 개수: {collection.num_entities:,}개")
    except Exception as e:
        print(f"✗ 컬렉션 로드 실패: {e}")
        return
    
    # 3. 임베딩 모델 로드
    print("\n[3/4] 임베딩 모델 로드 중...")
    print(f"모델: {EMBEDDING_MODEL}")
    
    # 디바이스 설정
    if USE_CPU:
        device = 'cpu'
        print("⚠️  CPU 모드로 실행합니다")
    else:
        if torch.backends.mps.is_available():
            device = 'mps'
            print("✓ MPS (Apple Silicon GPU) 사용")
        else:
            device = 'cpu'
            print("⚠️  CPU 모드로 실행합니다 (MPS 사용 불가)")
    
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print("✓ 임베딩 모델 로드 완료")
    
    # 4. 직접 정의한 쿼리 문장으로 검색
    print("\n[4/4] 쿼리 문장으로 검색 중...")
    print("=" * 80)
    
    test_cases = []
    
    for idx, query_sentence in enumerate(TEST_QUERIES, 1):
        print(f"\n[쿼리 {idx}]")
        print(f"  🔍 검색 쿼리 문장:")
        print(f"  {query_sentence.replace(chr(10), ' ').replace(chr(13), ' ')}")
        
        # 임베딩 생성
        print(f"  ⏳ 임베딩 생성 중...")
        query_embedding = model.encode(
            query_sentence,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()
        
        # 벡터 검색
        print(f"  ⏳ 벡터 검색 중...")
        search_results = search_similar_chunks(collection, query_embedding, top_k=5)
        
        if not search_results:
            print(f"  ⚠️  검색 결과가 없습니다.")
            continue
        
        # 결과 저장
        test_cases.append({
            "query_sentence": query_sentence,
            "search_results": search_results
        })
        
        # 결과 출력
        print(f"\n  📊 검색 결과 (상위 5개):")
        print(f"  {'-' * 76}")
        for result_idx, result in enumerate(search_results, 1):
            print(f"\n  [{result_idx}] 거리: {result['distance']:.4f}")
            print(f"      파일: {result['file_path']}")
            print(f"      문서명: {result['name']}")
            print(f"      청크 인덱스: {result['chunk_index']}")
            chunk_text = result['chunk_text'].replace(chr(10), ' ').replace(chr(13), ' ')
            print(f"      청크 내용: {chunk_text}")
    
    # 전체 요약 출력
    print("\n" + "=" * 80)
    print("테스트 완료 요약")
    print("=" * 80)
    print(f"총 테스트 케이스: {len(test_cases)}개")
    for idx, case in enumerate(test_cases, 1):
        print(f"\n[테스트 케이스 {idx}]")
        query_text = case['query_sentence'].replace(chr(10), ' ').replace(chr(13), ' ')
        print(f"  쿼리: {query_text}")
        print(f"  검색 결과: {len(case['search_results'])}개")


if __name__ == "__main__":
    main()

