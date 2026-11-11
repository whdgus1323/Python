# 키워드 서칭 프로

구매대행 상품 목록을 분석해 키워드별 해외배송 비율과 벤치마킹 후보를 선별하는 Windows 데스크톱 앱입니다. 웹 서버나 브라우저를 실행하지 않습니다.

## 실행

```powershell
uv sync
uv run python desktop_app.py
```

`sample_products.csv`를 열어 바로 확인할 수 있습니다. 필수 열은 `keyword`, `rank`, `product_name`, `product_url`, `price`, `review_count`, `delivery_type`, `seller_name`입니다. 여러 파일을 같은 폴더에 넣으면 카테고리 폴더 단위로 합쳐서 분석합니다.

이 도구의 추천은 시장 조사 보조 정보이며 판매 성과를 보장하지 않습니다.
