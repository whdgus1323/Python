from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, override

import polars as pl

from keyword_sourcing.analysis import AnalysisResult

SUPPORTED_INPUT_SUFFIXES: Final[frozenset[str]] = frozenset({".csv", ".xls", ".xlsx"})


@dataclass(frozen=True, slots=True)
class UnsupportedInputError(Exception):
    path: Path

    @override
    def __str__(self) -> str:
        return f"지원하지 않는 파일 형식입니다: {self.path.name}"


@dataclass(frozen=True, slots=True)
class EmptyInputDirectoryError(Exception):
    path: Path

    @override
    def __str__(self) -> str:
        return f"분석할 CSV 또는 Excel 파일이 없습니다: {self.path}"


def load_products(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    match suffix:
        case ".csv":
            products = pl.read_csv(path)
        case ".xls" | ".xlsx":
            products = pl.read_excel(path)
        case _:
            raise UnsupportedInputError(path=path)
    return products.with_columns(pl.lit(path.name).alias("source_file"))


def load_product_folder(path: Path) -> pl.DataFrame:
    source_paths = tuple(
        source_path
        for source_path in sorted(path.iterdir())
        if source_path.is_file() and source_path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    )
    if not source_paths:
        raise EmptyInputDirectoryError(path=path)
    return pl.concat([load_products(source_path) for source_path in source_paths], how="diagonal")


def export_recommendations(path: Path, result: AnalysisResult) -> None:
    suffix = path.suffix.lower()
    match suffix:
        case ".csv":
            result.recommendations.write_csv(path)
        case ".xlsx":
            result.recommendations.write_excel(path, worksheet="벤치마킹 후보")
        case _:
            raise UnsupportedInputError(path=path)
