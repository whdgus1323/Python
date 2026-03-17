import requests
import pandas as pd
import time

BASE_URL = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
COUNTRY_CODES = [
    "KOR","USA","GBR","SGP","IND",
    "JPN","DEU","FRA","CAN","AUS",
    "CHN","NLD","SWE","CHE","FIN"
]
YEAR_START = 2010
YEAR_END = 2024
MAX_RETRIES = 5
TIMEOUT = 60
RETRY_SLEEP = 3.0


def fetch_indicator_multi(codes, indicator, label):
    countries_str = ";".join(codes)
    url = BASE_URL.format(countries=countries_str, indicator=indicator)
    params = {
        "format": "json",
        "per_page": 2000,
        "date": f"{YEAR_START}:{YEAR_END}"
    }
    attempt = 0
    last_error = None
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            if resp.status_code != 200:
                last_error = resp.status_code
                time.sleep(RETRY_SLEEP)
                continue
            data = resp.json()
            if not isinstance(data, list) or len(data) < 2 or not isinstance(data[1], list):
                last_error = "json"
                time.sleep(RETRY_SLEEP)
                continue
            rows = []
            for item in data[1]:
                year = item.get("date")
                value = item.get("value")
                country_info = item.get("country") or {}
                country_code = country_info.get("id")
                country_name = country_info.get("value")
                try:
                    year_int = int(year)
                except:
                    continue
                if year_int < YEAR_START or year_int > YEAR_END:
                    continue
                rows.append({
                    "country_code": country_code,
                    "country_name": country_name,
                    "year": year_int,
                    label: value
                })
            df = pd.DataFrame(rows)
            return df, None
        except Exception as e:
            last_error = str(e)
            time.sleep(RETRY_SLEEP)
    return pd.DataFrame(columns=["country_code","country_name","year",label]), last_error


def build_panel_wdi():
    indicators = [
        ("IT.NET.USER.ZS", "T_internet_users"),
        ("IT.CEL.SETS.P2", "T_mobile_subs"),
        ("IT.NET.BBND.P2", "T_fixed_broadband"),
        ("FS.AST.PRVT.GD.ZS", "M_credit_private_gdp"),
        ("NY.GDP.PCAP.CD", "M_gdp_per_capita")
    ]
    base_df = None
    for ind, label in indicators:
        df, _ = fetch_indicator_multi(COUNTRY_CODES, ind, label)
        if base_df is None:
            base_df = df
        else:
            base_df = pd.merge(
                base_df,
                df,
                on=["country_code","country_name","year"],
                how="outer"
            )
    base_df = base_df.sort_values(["year","country_code"]).reset_index(drop=True)
    return base_df


def build_panel_wgi():
    indicators = [
        ("RQ.EST", "R_regulatory_quality"),
        ("RL.EST", "R_rule_of_law")
    ]
    base_df = None
    for ind, label in indicators:
        df, _ = fetch_indicator_multi(COUNTRY_CODES, ind, label)
        if base_df is None:
            base_df = df
        else:
            base_df = pd.merge(
                base_df,
                df,
                on=["country_code","country_name","year"],
                how="outer"
            )
    base_df = base_df.sort_values(["year","country_code"]).reset_index(drop=True)
    return base_df


def build_full_panel():
    wdi = build_panel_wdi()
    wgi = build_panel_wgi()
    panel = pd.merge(
        wdi,
        wgi,
        on=["country_code","country_name","year"],
        how="outer"
    )
    panel = panel.sort_values(["year","country_code"]).reset_index(drop=True)
    return panel


if __name__ == "__main__":
    panel_df = build_full_panel()
    panel_df.to_csv("topsis_fintech_panel_2010_2024.csv", index=False, encoding="utf-8-sig")
    print(panel_df.shape)
    print("saved topsis_fintech_panel_2010_2024.csv")
