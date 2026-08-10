from __future__ import annotations

from typing import Final, final, override

import polars as pl
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt
from PySide6.QtGui import QBrush, QColor

OVERSEAS_BRUSH: Final[QBrush] = QBrush(QColor(82, 61, 15))
CANDIDATE_BRUSH: Final[QBrush] = QBrush(QColor(16, 77, 49))
ROOT_INDEX: Final[QModelIndex] = QModelIndex()
DISPLAY_ROLE: Final[int] = int(Qt.ItemDataRole.DisplayRole)
DISPLAY_LABELS: Final[dict[str, str]] = {
    "keyword": "키워드",
    "rank": "순위",
    "product_name": "상품명",
    "product_url": "상품 링크",
    "price": "가격",
    "review_count": "리뷰 수",
    "delivery_type": "배송 유형",
    "seller_name": "판매자",
    "source_file": "원본 파일",
    "product_count": "상품 수",
    "overseas_ratio": "해외배송 비율",
    "average_reviews": "평균 리뷰 수",
    "median_price": "중앙 가격",
    "status": "상태",
    "is_overseas": "해외배송 여부",
}
PRICE_COLUMNS: Final[frozenset[str]] = frozenset({"price", "median_price"})
INTEGER_COLUMNS: Final[frozenset[str]] = frozenset({"rank", "review_count", "product_count"})


@final
class ProductTableModel(QAbstractTableModel):
    def __init__(self, frame: pl.DataFrame, mark_candidates: bool = False) -> None:
        super().__init__()
        self._frame: pl.DataFrame = frame
        self._mark_candidates: bool = mark_candidates

    @override
    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = ROOT_INDEX) -> int:
        return 0 if parent.isValid() else self._frame.height

    @override
    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = ROOT_INDEX) -> int:
        return 0 if parent.isValid() else self._frame.width

    @override
    def data(
        self, index: QModelIndex | QPersistentModelIndex, role: int = DISPLAY_ROLE
    ) -> str | int | QBrush | None:
        if not index.isValid():
            return None
        if role == DISPLAY_ROLE:
            return self._display_value(index.row(), index.column())
        if role == int(Qt.ItemDataRole.BackgroundRole):
            return self._row_brush(index.row())
        if role == int(Qt.ItemDataRole.TextAlignmentRole):
            return self._cell_alignment(index.column())
        return None

    @override
    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = DISPLAY_ROLE
    ) -> str | None:
        if role != DISPLAY_ROLE:
            return None
        if orientation == Qt.Orientation.Horizontal:
            column = self._frame.columns[section]
            return DISPLAY_LABELS.get(column, column)
        return str(section + 1)

    def _display_value(self, row: int, column: int) -> str:
        column_name = self._frame.columns[column]
        value = self._frame.item(row, column)
        if column_name in PRICE_COLUMNS:
            return f"{float(value):,.0f}원"
        if column_name in INTEGER_COLUMNS:
            return f"{int(value):,}"
        if column_name == "overseas_ratio":
            return f"{float(value):.0%}"
        if column_name == "average_reviews":
            return f"{float(value):,.1f}"
        if column_name == "is_overseas":
            return "해외배송" if bool(value) else "국내배송"
        return str(value)

    def column_name(self, column: int) -> str:
        return self._frame.columns[column]

    def _cell_alignment(self, column: int) -> int:
        column_name = self._frame.columns[column]
        numeric_columns = PRICE_COLUMNS | INTEGER_COLUMNS | {"overseas_ratio", "average_reviews"}
        horizontal = (
            Qt.AlignmentFlag.AlignRight
            if column_name in numeric_columns
            else Qt.AlignmentFlag.AlignLeft
        )
        return int(Qt.AlignmentFlag.AlignVCenter | horizontal)

    def _row_brush(self, row: int) -> QBrush | None:
        columns = self._frame.columns
        if self._mark_candidates:
            return CANDIDATE_BRUSH
        if "is_overseas" in columns and bool(self._frame.item(row, "is_overseas")):
            return OVERSEAS_BRUSH
        if "delivery_type" in columns and self._frame.item(row, "delivery_type") == "해외배송":
            return OVERSEAS_BRUSH
        return None
