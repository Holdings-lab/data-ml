"""
shared 뉴스 처리 패키지.

크롤러 뉴스 로드, 일자별 집계, 시장 프레임 병합 로직을 포함한다.
"""

from shared.news.features import build_daily_news_feature_table, load_news_source_table
from shared.news.merge import merge_news_features_into_market_frame

__all__ = [
    "build_daily_news_feature_table",
    "load_news_source_table",
    "merge_news_features_into_market_frame",
]
