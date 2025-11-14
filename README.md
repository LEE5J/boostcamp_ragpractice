# 법률 문서 RAG 실습 프로젝트

법률 문서 XML 파일을 임베딩하여 Milvus 벡터 DB에 저장하고 검색하는 프로젝트입니다.

## 설정

### Milvus 설정
- **Host**: `192.168.50.20`
- **Port**: `19530`
- **Collection Name**: `legal_documents`

### 임베딩 모델
- **Model**: `google/embeddinggemma-300m`
- **Dimension**: `768`
- **인덱스 타입**: `IVF_FLAT`
- **거리 메트릭**: `L2`

> ⚠️ **주의**: 임베딩 모델을 사용하기 전에 Hugging Face 로그인이 필요합니다.
> 
> ```bash
> huggingface-cli login
> ```

### 벡터 DB 스키마
- `id` (INT64, Primary Key, Auto ID)
- `file_path` (VARCHAR, max_length: 500)
- `category` (VARCHAR, max_length: 200)
- `name` (VARCHAR, max_length: 500)
- `chunk_text` (VARCHAR, max_length: 65535)
- `chunk_index` (INT64)
- `embedding` (FLOAT_VECTOR, dimension: 768)

## 설치 및 실행

### uv 사용법

이 프로젝트는 `uv`를 사용하여 패키지 관리를 합니다.

```bash
# 프로젝트 설치
uv sync

# 가상환경 활성화 (필요한 경우)
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows

test_vector_search.py  부분을 참고해서 어떻게 벡터 서치를 사용하는지 알 수 있습니다.

## Milvus 접근 방법

Milvus는 `pymilvus` 라이브러리를 통해 접근합니다.

```python
from pymilvus import connections, Collection

# 연결
connections.connect(
    alias="default",
    host="192.168.50.20",
    port="19530"
)

# 컬렉션 로드
collection = Collection("legal_documents")
collection.load()
```

## XML 파일 가져오기
이것은 ai hub의 데이터를 활용했는데 "019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터" > Training > 원천데이터 > TS_1.판결문 폴터를 raw_corpus 로 바꾼후 벡터 DB 에 주입하였습니다.
시간상의 관계로 10000건의 데이터만 진행하였습니다.
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71487
여기서
XML 파일은 `raw_corpus` 디렉토리에 저장되어 있으며, `xml_parser.py`의 `extract_text_from_xml()` 함수를 사용하여 파싱합니다.


```python
from xml_parser import extract_text_from_xml

# XML 파일에서 텍스트 추출
result = extract_text_from_xml("raw_corpus/1/2019/2019나134.xml")
# 반환값: {"file_path": "...", "category": "...", "name": "...", "text": "..."}
```

XML 파일은 `raw_corpus` 디렉토리 하위에 재귀적으로 저장되어 있으며, 각 XML 파일은 다음 구조를 가집니다:
- `<file>`: 루트 요소
  - `<category>`: 카테고리 정보
  - `<name>`: 문서명
  - `<cn>`: 본문 텍스트

## 벡터 DB 사용

### 데이터 삽입
`insert2db.py`는 XML 파일들을 임베딩하여 벡터 DB에 삽입하는 스크립트입니다.

### 벡터 검색

벡터 DB에서 검색할 때는 다음 사항을 주의해야 합니다:

1. **임베딩 차원**: 검색 쿼리도 **768 차원**의 임베딩 벡터로 변환해야 합니다.
2. **인덱스 타입**: `IVF_FLAT` 인덱스를 사용하므로, 검색 시 `nprobe` 파라미터를 설정해야 합니다.
3. **검색 파라미터**:
   ```python
   search_params = {
       "metric_type": "L2",
       "params": {"nprobe": 10}
   }
   ```
4. **출력 필드**: 검색 시 다음 필드를 가져올 수 있습니다:
   - `file_path`
   - `category`
   - `name`
   - `chunk_text`
   - `chunk_index`

검색 예제는 `test_vector_search.py`를 참고하세요.

