"""수집기의 정책을 바꾸지 않는 텍스트, HTTP, CSV 공통 처리."""

from __future__ import annotations

import csv
import re
from pathlib import Path


def normalize_whitespace(text: str) -> str:
    """연속 공백을 하나로 합치고 양끝 공백을 제거한다."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str | None) -> str:
    """비어 있는 입력을 허용하는 공백 정리 함수."""
    return normalize_whitespace(text) if text else ""


def fetch_soup(url, *, client, **request_kwargs):
    """호출자가 지정한 클라이언트와 옵션으로 HTML을 읽는다. 오류는 전파한다."""
    from bs4 import BeautifulSoup

    response = client.get(url, **request_kwargs)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def create_retry_session(headers, retry, **adapter_kwargs):
    """수집기별 Retry 설정으로 HTTP/HTTPS 연결 풀을 구성한다."""
    import requests
    from requests.adapters import HTTPAdapter

    session = requests.Session()
    session.headers.update(headers)
    adapter = HTTPAdapter(max_retries=retry, **adapter_kwargs)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def write_csv_records(
    path, records, fieldnames, *, mode="w", encoding="utf-8-sig", write_header=True
):
    """컬럼 순서, 저장 모드, 헤더 여부는 각 수집기가 결정한다."""
    with Path(path).open(mode, encoding=encoding, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(records)
