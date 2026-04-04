"""
shared 파이프라인 패키지.

여러 기능 모듈을 조합해 실제 end-to-end 실행 흐름을 정의한다.
"""

from shared.pipelines.market_news import run_market_news_training_pipeline

__all__ = ["run_market_news_training_pipeline"]
