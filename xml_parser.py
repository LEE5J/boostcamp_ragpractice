"""
XML 파일 파싱 모듈
BeautifulSoup을 사용하여 잘못된 형식의 XML도 파싱할 수 있습니다.
"""
from typing import Dict, Optional
from bs4 import BeautifulSoup


def extract_text_from_xml(xml_path: str) -> Optional[Dict[str, str]]:
    """
    XML 파일에서 텍스트를 추출합니다.
    BeautifulSoup을 사용하여 잘못된 형식의 XML도 파싱할 수 있습니다.
    
    Args:
        xml_path: XML 파일 경로
        
    Returns:
        파일명, 카테고리, 텍스트 내용을 포함한 딕셔너리
        파싱 실패 시 None 반환
    """
    try:
        with open(xml_path, 'r') as f:
            xml_content = f.read()
        
        # BeautifulSoup으로 파싱 (xml.parser 사용)
        soup = BeautifulSoup(xml_content, 'xml')
        
        category = ""
        name = ""
        text = ""
        
        # file 요소 찾기
        file_elem = soup.find('file')
        if file_elem:
            category_elem = file_elem.find('category')
            if category_elem:
                category = category_elem.get_text(strip=True)
            
            name_elem = file_elem.find('name')
            if name_elem:
                name = name_elem.get_text(strip=True)
            
            cn_elem = file_elem.find('cn')
            if cn_elem:
                text = cn_elem.get_text(strip=True)
        
        return {
            "file_path": xml_path,
            "category": category,
            "name": name,
            "text": text
        }
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None

