"""상품 목록을 키워드와 벤치마킹 후보로 변환한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "keyword",
        "rank",
        "product_name",
        "product_url",
        "price",
        "review_count",
        "delivery_type",
        "seller_name",
    }
)


class InputSchemaError(Exception):
    """필수 입력 열이 누락됐을 때 발생한다."""


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    """추천 계산에 적용할 사용자 조정 기준이다."""

    overseas_ratio_threshold: float = 0.4
    review_floor: int = 10
    review_ceiling: int = 300
    recommendation_limit: int = 3


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """키워드 요약, 모든 상품, 우선 추천 상품을 묶는다."""

    keywords: pl.DataFrame
    products: pl.DataFrame
    recommendations: pl.DataFrame


def analyze_products(products: pl.DataFrame, settings: AnalysisSettings) -> AnalysisResult:
    """입력 상품을 정규화해 키워드별 시장성과 벤치마킹 후보를 계산한다."""
    missing_columns = REQUIRED_COLUMNS.difference(products.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise InputSchemaError(f"필수 열이 없습니다: {missing}")

    normalized = (
        products.with_columns(
            pl.col("rank").cast(pl.Float64),
            pl.col("price").cast(pl.Float64),
            pl.col("review_count").cast(pl.Float64),
            pl.col("delivery_type").cast(pl.String).str.contains("해외").alias("is_overseas"),
        )
        .with_columns(
            (1.0 / pl.col("rank")).alias("rank_score"),
            pl.when(
                pl.col("review_count").is_between(settings.review_floor, settings.review_ceiling)
            )
            .then(1.0)
            .otherwise(0.0)
            .alias("review_score"),
        )
        .with_columns(
            (
                pl.when(pl.col("is_overseas")).then(0.4).otherwise(0.0)
                + pl.col("rank_score") * 0.35
                + pl.col("review_score") * 0.25
            )
            .round(3)
            .alias("benchmark_score"),
            pl.when(pl.col("is_overseas"))
            .then(pl.lit("해외배송"))
            .otherwise(pl.lit("국내배송"))
            .alias("delivery_label"),
        )
    )
    keywords = (
        normalized.group_by("keyword")
        .agg(
            pl.len().alias("product_count"),
            pl.col("is_overseas").mean().round(3).alias("overseas_ratio"),
            pl.col("review_count").mean().round(1).alias("average_reviews"),
            pl.col("price").median().round(0).alias("median_price"),
        )
        .with_columns(
            pl.when(pl.col("overseas_ratio") >= settings.overseas_ratio_threshold)
            .then(pl.lit("진입 검토"))
            .otherwise(pl.lit("보류"))
            .alias("status"),
        )
        .sort(["overseas_ratio", "average_reviews"], descending=[True, False])
    )
    recommendations = (
        normalized.filter(
            pl.col("is_overseas")
            & pl.col("review_count").is_between(settings.review_floor, settings.review_ceiling)
        )
        .join(keywords.select("keyword", "overseas_ratio"), on="keyword")
        .filter(pl.col("overseas_ratio") >= settings.overseas_ratio_threshold)
        .sort(["keyword", "benchmark_score"], descending=[False, True])
        .unique(subset=["keyword", "seller_name"], keep="first", maintain_order=True)
        .group_by("keyword", maintain_order=True)
        .head(settings.recommendation_limit)
        .sort(["keyword", "benchmark_score"], descending=[False, True])
    )
    return AnalysisResult(keywords=keywords, products=normalized, recommendations=recommendations)
