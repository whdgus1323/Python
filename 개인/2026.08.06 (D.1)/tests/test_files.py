from pathlib import Path

import polars as pl

from keyword_sourcing.files import load_products


def test_load_products_reads_csv_when_given_a_market_export(tmp_path: Path) -> None:
    # Given: a market search export with the required product columns.
    source = tmp_path / "outdoor.csv"
    pl.DataFrame(
        {
            "keyword": ["folding awning"],
            "rank": [1],
            "product_name": ["awning"],
            "product_url": ["https://example.com/product"],
            "price": [10000],
            "review_count": [20],
            "delivery_type": ["overseas"],
            "seller_name": ["seller"],
        }
    ).write_csv(source)

    # When: the desktop application loads the export.
    products = load_products(source)

    # Then: the source category is retained for later folder-level analysis.
    assert products.row(0, named=True)["source_file"] == "outdoor.csv"
