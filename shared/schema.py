from typing import TypedDict


class FedDocument(TypedDict, total=False):
    release_date: str
    release_time: str
    is_sep: bool
    doc_type: str
    label: str
    url: str
    title: str
    body_text: str
