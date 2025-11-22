"""
raw_corpus의 XML 파일들을 임베딩하여 Milvus 벡터 DB에 삽입하는 스크립트
"""
from pathlib import Path
from typing import List, Dict
from xml_parser import extract_text_from_xml

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch


# 설정
MILVUS_HOST = "192.168.50.20"
MILVUS_PORT = "19530"  # Milvus 기본 포트
COLLECTION_NAME = "legal_documents_180K"
RAW_CORPUS_PATH = "raw_corpus"
EMBEDDING_MODEL = "google/embeddinggemma-300m"  # 한국어 임베딩 모델
DIMENSION = 768  # 임베딩 차원
CHUNK_SIZE = 500  # 텍스트 청크 크기 (문자 수)
CHUNK_OVERLAP = 50  # 청크 간 겹치는 문자 수
MAX_CHUNKS = 1000000  # 최대 삽입할 청크 개수
EMBEDDING_BATCH_SIZE = 8  # 임베딩 배치 크기 (메모리 부족 시 줄이기)
USE_CPU = False  # True로 설정하면 CPU 사용 (MPS 메모리 부족 시)
DROP_EXISTING_COLLECTION = False  # True로 설정하면 기존 컬렉션을 삭제하고 새로 생성


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    텍스트를 청크로 분할합니다.
    
    Args:
        text: 분할할 텍스트
        chunk_size: 청크 크기 (문자 수)
        overlap: 청크 간 겹치는 문자 수
        
    Returns:
        텍스트 청크 리스트
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 문장 단위로 자르기 (개선된 버전)
        if end < len(text):
            # 마지막 문장 부호나 줄바꿈을 찾아서 자르기
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            last_space = text.rfind(' ', start, end)
            
            # 가장 가까운 구분자를 찾아서 자르기
            cut_point = max(last_period, last_newline, last_space)
            if cut_point > start:
                end = cut_point + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def get_all_xml_files(root_dir: str) -> List[str]:
    """
    디렉토리에서 모든 XML 파일을 재귀적으로 찾습니다.
    
    Args:
        root_dir: 검색할 루트 디렉토리
        
    Returns:
        XML 파일 경로 리스트
    """
    xml_files = []
    root_path = Path(root_dir)
    
    for xml_file in root_path.rglob("*.xml"):
        xml_files.append(str(xml_file))
    
    return sorted(xml_files)


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


def create_collection_if_not_exists():
    """
    컬렉션이 없으면 생성합니다.
    DROP_EXISTING_COLLECTION이 True인 경우 기존 컬렉션을 삭제하고 새로 생성합니다.
    
    Returns:
        Collection 객체
    """
    # 기존 컬렉션이 있고 DROP_EXISTING_COLLECTION이 True인 경우 삭제
    if DROP_EXISTING_COLLECTION and utility.has_collection(COLLECTION_NAME):
        print(f"⚠️  기존 컬렉션 '{COLLECTION_NAME}' 삭제 중...")
        utility.drop_collection(COLLECTION_NAME)
        print(f"✓ 기존 컬렉션 '{COLLECTION_NAME}' 삭제 완료")
    
    # 기존 컬렉션이 있으면 반환
    if utility.has_collection(COLLECTION_NAME):
        print(f"컬렉션 '{COLLECTION_NAME}'이 이미 존재합니다.")
        collection = Collection(COLLECTION_NAME)
        return collection
    
    # 스키마 정의
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="법률 문서 임베딩 컬렉션"
    )
    
    # 컬렉션 생성
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    
    # 인덱스 생성
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    
    print(f"✓ 컬렉션 '{COLLECTION_NAME}' 생성 완료")
    return collection


def insert_data_to_milvus(collection: Collection, data: List[Dict], embeddings: List[List[float]]):
    """
    데이터를 Milvus에 삽입합니다.
    
    Args:
        collection: Milvus 컬렉션 객체
        data: 삽입할 데이터 리스트
        embeddings: 임베딩 벡터 리스트
    """
    # 배치 크기
    batch_size = 1000
    
    with tqdm(total=len(data), desc="Milvus에 데이터 삽입 중", unit="청크") as pbar:
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            entities = [
                [item["file_path"] for item in batch_data],
                [item["category"] for item in batch_data],
                [item["name"] for item in batch_data],
                [item["chunk_text"] for item in batch_data],
                [item["chunk_index"] for item in batch_data],
                batch_embeddings,
            ]
            
            collection.insert(entities)
            pbar.update(len(batch_data))
    
    # 데이터 플러시
    collection.flush()
    print(f"✓ 총 {len(data):,}개의 청크가 삽입되었습니다.")


def main():
    """메인 함수"""
    print("=" * 60)
    print("법률 문서 임베딩 및 Milvus 삽입 스크립트")
    print("=" * 60)
    
    # 설정 정보 출력
    print(f"\n설정 정보:")
    print(f"  - 컬렉션명: {COLLECTION_NAME}")
    print(f"  - 기존 컬렉션 삭제: {'예' if DROP_EXISTING_COLLECTION else '아니오'}")
    if DROP_EXISTING_COLLECTION:
        print(f"  ⚠️  경고: 기존 컬렉션이 삭제되고 새로 생성됩니다!")
    
    # 1. Milvus 연결
    print("\n[1/5] Milvus 연결 중...")
    connect_to_milvus()
    
    # 2. 컬렉션 생성/확인
    print("\n[2/5] 컬렉션 확인/생성 중...")
    collection = create_collection_if_not_exists()
    
    # 3. XML 파일 찾기
    print("\n[3/5] XML 파일 검색 중...")
    xml_files = get_all_xml_files(RAW_CORPUS_PATH)
    print(f"✓ 총 {len(xml_files)}개의 XML 파일을 찾았습니다.")
    
    if len(xml_files) == 0:
        print("✗ XML 파일을 찾을 수 없습니다.")
        return
    
    # 4. 텍스트 추출 및 청크 분할
    print(f"\n[4/5] 텍스트 추출 및 청크 분할 중... (최대 {MAX_CHUNKS:,}개)")
    all_chunks = []
    
    with tqdm(total=MAX_CHUNKS, desc="청크 생성 중", unit="청크") as pbar:
        for xml_file in xml_files:
            if len(all_chunks) >= MAX_CHUNKS:
                pbar.update(MAX_CHUNKS - pbar.n)
                break
                
            extracted = extract_text_from_xml(xml_file)
            if extracted and extracted["text"]:
                chunks = split_text_into_chunks(extracted["text"])
                # 문서명을 각 청크 앞에 추가
                doc_name_prefix = f"[문서명: {extracted['name']}]\n\n" if extracted.get("name") else ""
                
                for idx, chunk in enumerate(chunks):
                    if len(all_chunks) >= MAX_CHUNKS:
                        break
                    
                    # 청크 앞에 문서명 추가
                    chunk_with_name = doc_name_prefix + chunk
                    all_chunks.append({
                        "file_path": extracted["file_path"],
                        "category": extracted["category"],
                        "name": extracted["name"],
                        "chunk_text": chunk_with_name,
                        "chunk_index": idx,
                    })
                    pbar.update(1)
    
    print(f"✓ 총 {len(all_chunks):,}개의 텍스트 청크를 생성했습니다.")
    
    if len(all_chunks) == 0:
        print("✗ 처리할 텍스트 청크가 없습니다.")
        return
    
    # 5. 임베딩 생성
    print(f"\n[5/6] 임베딩 생성 중... ({len(all_chunks):,}개)")
    print(f"임베딩 모델 로딩: {EMBEDDING_MODEL}")
    
    # 디바이스 설정 (메모리 부족 시 CPU 사용)
    if USE_CPU:
        device = 'cpu'
        print("⚠️  CPU 모드로 실행합니다 (메모리 절약)")
    else:
        # MPS 메모리 부족 시 자동으로 CPU로 전환
        if torch.backends.mps.is_available():
            device = 'mps'
            print("✓ MPS (Apple Silicon GPU) 사용")
        else:
            device = 'cpu'
            print("⚠️  CPU 모드로 실행합니다 (MPS 사용 불가)")
    
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    
    texts = [chunk["chunk_text"] for chunk in all_chunks]
    
    # 메모리 부족 방지를 위해 작은 배치로 나눠서 처리
    print(f"배치 크기: {EMBEDDING_BATCH_SIZE} (메모리 절약 모드)")
    all_embeddings = []
    
    try:
        # 전체를 한 번에 처리 시도
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True
        ).tolist()
        all_embeddings = embeddings
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "MPS" in str(e):
            print(f"\n⚠️  메모리 부족 감지: 배치 크기를 더 줄여서 처리합니다...")
            # 배치 크기를 더 줄여서 재시도
            smaller_batch = max(1, EMBEDDING_BATCH_SIZE // 2)
            print(f"배치 크기를 {smaller_batch}로 줄입니다...")
            
            # CPU로 전환
            if device == 'mps':
                print("CPU로 전환합니다...")
                model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
                device = 'cpu'
            
            # 더 작은 배치로 나눠서 처리
            all_embeddings = []
            for i in tqdm(range(0, len(texts), smaller_batch), desc="임베딩 생성 (작은 배치)"):
                batch_texts = texts[i:i+smaller_batch]
                batch_embeddings = model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    batch_size=smaller_batch,
                    convert_to_numpy=True
                ).tolist()
                all_embeddings.extend(batch_embeddings)
        else:
            raise
    
    embeddings = all_embeddings
    print(f"✓ 임베딩 생성 완료: {len(embeddings):,}개")
    
    # 6. Milvus에 삽입
    print("\n[6/6] Milvus에 데이터 삽입 중...")
    insert_data_to_milvus(collection, all_chunks, embeddings)
    
    # 컬렉션 로드
    collection.load()
    print(f"✓ 컬렉션 로드 완료")
    
    # 통계 출력
    num_entities = collection.num_entities
    print(f"\n{'=' * 60}")
    print(f"완료! 총 {num_entities:,}개의 벡터가 저장되었습니다.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

