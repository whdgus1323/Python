import polars as pl
from PySide6.QtCore import Qt

from keyword_sourcing.analysis import AnalysisSettings, analyze_products
from keyword_sourcing.table_model import ProductTableModel


def test_analyze_products_recommends_low_competition_overseas_product() -> None:
    # Given: 해외배송 비율이 기준 이상이고 리뷰 경쟁이 과하지 않은 상품 목록
    products = pl.DataFrame(
        {
            "keyword": ["접이식 어닝"] * 5,
            "rank": [1, 2, 3, 4, 5],
            "product_name": ["A", "B", "C", "D", "E"],
            "product_url": ["https://example.com/a"] * 5,
            "price": [100000, 98000, 110000, 105000, 99000],
            "review_count": [600, 80, 30, 400, 20],
            "delivery_type": ["해외배송", "해외배송", "국내배송", "해외배송", "국내배송"],
            "seller_name": ["판매자A", "판매자B", "판매자C", "판매자D", "판매자E"],
        }
    )

    # When: 영상의 40% 해외배송 기준으로 분석한다.
    result = analyze_products(products, AnalysisSettings())

    # Then: 해외배송 비율은 60%이고, 과도한 리뷰의 1위 상품 대신 B가 추천된다.
    assert result.keywords.row(0, named=True)["overseas_ratio"] == 0.6
    assert result.recommendations.row(0, named=True)["product_name"] == "B"


def test_product_table_model_displays_user_facing_column_labels() -> None:
    # Given: a product frame with internal analysis field names.
    frame = pl.DataFrame({"product_name": ["A"], "review_count": [20]})
    model = ProductTableModel(frame)

    # When: the native table asks for horizontal labels.
    product_header = model.headerData(0, Qt.Orientation.Horizontal)
    review_header = model.headerData(1, Qt.Orientation.Horizontal)

    # Then: operators see readable Korean labels instead of internal field names.
    assert product_header == "상품명"
    assert review_header == "리뷰 수"
