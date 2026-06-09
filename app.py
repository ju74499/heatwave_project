# app.py
# ============================================================
# 서울시 독거노인 폭염 취약지역 및 무더위쉼터 공급 불균형 분석
# 의사결정형 Streamlit 대시보드
# ============================================================
# 실행 방법
# 1) app.py, final.db, requirements.txt를 같은 프로젝트 폴더에 둡니다.
# 2) GeoJSON 파일 2개는 data 폴더 안에 둡니다.
#    - data/seoul_district_boundary_simplified.geojson
#    - data/seoul_adm_dong_simplified.geojson
# 3) 터미널에서 streamlit run app.py 를 실행합니다.
# ============================================================

import os
import json
import copy
import sqlite3
from typing import List, Dict, Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium


# ============================================================
# 1. 수정 가능한 설정값
# ============================================================

DB_PATH = "final.db"

# Streamlit Cloud 배포를 위해 상대경로를 사용합니다.
# data 폴더에 파일이 없으면 app.py와 같은 위치에서도 한 번 더 찾도록 코드에서 처리합니다.
DISTRICT_GEOJSON_PATH = "data/seoul_district_boundary_simplified.geojson"
DONG_GEOJSON_PATH = "data/seoul_adm_dong_simplified.geojson"

# GeoJSON 컬럼명
# 현재 단순화 파일 기준:
# 자치구 경계: gu_code, district, geometry
# 행정동 경계: ADM_CD, ADM_NM, gu_code, district, geometry
GEOJSON_DISTRICT_COL = "district"
GEOJSON_DONG_DISTRICT_COL = "district"
GEOJSON_DONG_COL = "ADM_NM"

REQUIRED_TABLES = [
    "district_summary",
    "district_priority",
    "shelters",
]

SEOUL_CENTER = [37.5665, 126.9780]


# ============================================================
# 2. Streamlit 기본 설정
# ============================================================

st.set_page_config(
    page_title="서울시 독거노인 폭염 취약지역 분석",
    page_icon="🌡️",
    layout="wide",
)


# ============================================================
# 3. 공통 유틸 함수
# ============================================================

def fmt_int(value):
    if pd.isna(value):
        return "-"
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return str(value)


def fmt_float(value, digits=2):
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def fmt_pct(value):
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):,.2f}%"
    except Exception:
        return str(value)


def safe_num(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def file_path_with_fallback(path: str) -> Optional[str]:
    """data 폴더에서 먼저 찾고, 없으면 현재 폴더에서도 찾습니다."""
    if os.path.exists(path):
        return path

    base = os.path.basename(path)
    if os.path.exists(base):
        return base

    return None


@st.cache_data(show_spinner=False)
def get_table_names(db_path: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
        conn,
    )["name"].tolist()
    conn.close()
    return tables


@st.cache_data(show_spinner=False)
def get_table_columns(db_path: str, table_name: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
    conn.close()
    return [row[1] for row in rows]


@st.cache_data(show_spinner=False)
def run_sql(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


@st.cache_data(show_spinner=False)
def load_geojson(path: str) -> Optional[Dict[str, Any]]:
    real_path = file_path_with_fallback(path)
    if real_path is None:
        return None

    try:
        with open(real_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(real_path, "r", encoding="cp949") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"GeoJSON 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None


def has_columns(df: pd.DataFrame, cols: List[str]) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.warning(f"필요한 컬럼이 없습니다: {', '.join(missing)}")
        return False
    return True


def show_sql(sql: str):
    with st.expander("사용한 SQL 보기"):
        st.code(sql.strip(), language="sql")


def insight_box(title: str, lines: List[str]):
    st.markdown(f"**{title}**")
    for line in lines:
        st.caption(f"• {line}")


def enrich_geojson_by_key(
    geojson_data: Dict[str, Any],
    df: pd.DataFrame,
    geo_key: str,
    df_key: str,
    value_cols: List[str],
) -> Dict[str, Any]:
    """GeoJSON properties에 DB 값을 추가합니다."""
    enriched = copy.deepcopy(geojson_data)

    if df.empty or df_key not in df.columns:
        return enriched

    available_cols = [c for c in value_cols if c in df.columns]
    lookup = df.set_index(df_key)[available_cols].to_dict("index")

    for feature in enriched.get("features", []):
        props = feature.get("properties", {})
        key = props.get(geo_key)
        values = lookup.get(key, {})

        for col in available_cols:
            props[col] = values.get(col, None)

        feature["properties"] = props

    return enriched


def make_tooltip_fields(base_fields: List[str], aliases: List[str], geojson_data: Dict[str, Any]):
    """GeoJSON에 실제 존재하는 필드만 tooltip에 사용합니다."""
    if not geojson_data or not geojson_data.get("features"):
        return [], []

    props = geojson_data["features"][0].get("properties", {})
    valid_fields = []
    valid_aliases = []

    for f, a in zip(base_fields, aliases):
        if f in props:
            valid_fields.append(f)
            valid_aliases.append(a)

    return valid_fields, valid_aliases


def make_choropleth_map(
    geojson_data: Optional[Dict[str, Any]],
    df: pd.DataFrame,
    geo_key: str,
    df_key: str,
    value_col: str,
    legend_name: str,
    tooltip_fields: List[str],
    tooltip_aliases: List[str],
    fill_color: str = "YlOrRd",
    zoom_start: int = 11,
):
    """자치구 단위 Choropleth 지도 생성 함수입니다."""
    if geojson_data is None:
        st.warning("경계 GeoJSON 파일이 없어 지도 시각화를 건너뜁니다.")
        return

    if df.empty:
        st.warning("지도에 표시할 데이터가 없습니다.")
        return

    required = [df_key, value_col]
    if not has_columns(df, required):
        return

    map_df = df.copy()
    map_df[value_col] = pd.to_numeric(map_df[value_col], errors="coerce").fillna(0)

    value_cols = [c for c in tooltip_fields if c != geo_key]
    enriched = enrich_geojson_by_key(
        geojson_data,
        map_df,
        geo_key=geo_key,
        df_key=df_key,
        value_cols=value_cols,
    )

    m = folium.Map(location=SEOUL_CENTER, zoom_start=zoom_start, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=enriched,
        data=map_df,
        columns=[df_key, value_col],
        key_on=f"feature.properties.{geo_key}",
        fill_color=fill_color,
        fill_opacity=0.78,
        line_opacity=0.5,
        nan_fill_color="lightgray",
        legend_name=legend_name,
    ).add_to(m)

    fields, aliases = make_tooltip_fields(tooltip_fields, tooltip_aliases, enriched)

    if fields:
        folium.GeoJson(
            enriched,
            name="상세 정보",
            style_function=lambda feature: {
                "fillOpacity": 0,
                "color": "#333333",
                "weight": 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=fields,
                aliases=aliases,
                localize=True,
                sticky=True,
            ),
        ).add_to(m)

    st_folium(m, width=None, height=560)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    rename_map = {
        "district": "자치구",
        "priority_type": "우선유형",
        "priority_score": "우선순위 점수",
        "elderly_total": "독거노인 수",
        "elderly_80_plus": "80세 이상 독거노인 수",
        "elderly_80_plus_rate": "80세 이상 비율",
        "vulnerable_elderly_rate": "취약 독거노인 비율",
        "shelter_count": "쉼터 수",
        "total_capacity": "총 수용가능인원",
        "shelters_per_1000": "1,000명당 쉼터 수",
        "capacity_rate": "수용률",
        "elderly_per_shelter": "쉼터 1개당 독거노인 수",
    }

    int_cols = ["elderly_total", "elderly_80_plus", "shelter_count", "total_capacity", "priority_score"]
    pct_cols = ["elderly_80_plus_rate", "vulnerable_elderly_rate", "capacity_rate"]
    float_cols = ["shelters_per_1000", "elderly_per_shelter"]

    for col in result.columns:
        if col in int_cols:
            result[col] = result[col].apply(fmt_int)
        elif col in pct_cols:
            result[col] = result[col].apply(fmt_pct)
        elif col in float_cols:
            result[col] = result[col].apply(lambda x: fmt_float(x, 2))

    return result.rename(columns=rename_map)


def auto_priority_insights(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["우선지역 데이터가 없어 자동 해석을 생성할 수 없습니다."]

    top = df.iloc[0]
    lines = [
        f"가장 우선순위가 높은 지역은 {top['district']}이며, 우선유형은 '{top['priority_type']}'입니다.",
        f"이 지역은 독거노인 {fmt_int(top['elderly_total'])}명, 수용률 {fmt_pct(top['capacity_rate'])}, 1,000명당 쉼터 수 {fmt_float(top['shelters_per_1000'])}개로 나타납니다.",
        "우선순위 점수가 높은 지역은 단순히 쉼터 수가 적은 곳이 아니라 수요 규모와 공급 부족이 동시에 나타나는 지역으로 해석해야 합니다.",
    ]
    return lines


def district_reason(row: pd.Series, avg_capacity_rate: float, avg_per_1000: float, avg_80_rate: float) -> str:
    reasons = []

    if safe_num(row.get("capacity_rate")) < avg_capacity_rate:
        reasons.append("수용률이 평균보다 낮아 실제 수용능력 보강이 필요합니다")

    if safe_num(row.get("shelters_per_1000")) < avg_per_1000:
        reasons.append("독거노인 1,000명당 쉼터 수가 평균보다 낮아 공급 밀도가 부족합니다")

    if safe_num(row.get("elderly_80_plus_rate")) > avg_80_rate:
        reasons.append("80세 이상 독거노인 비율이 평균보다 높아 고령 취약성이 큽니다")

    if not reasons:
        reasons.append("복합 지표 기준에서 상대적으로 우선 검토가 필요한 지역입니다")

    return " / ".join(reasons)


# ============================================================
# 4. DB 점검
# ============================================================

if not os.path.exists(DB_PATH):
    st.error("final.db 파일을 찾을 수 없습니다. app.py와 같은 폴더에 final.db를 넣어 주세요.")
    st.stop()

existing_tables = get_table_names(DB_PATH)
missing_tables = [t for t in REQUIRED_TABLES if t not in existing_tables]

if missing_tables:
    st.error("필요한 테이블이 DB에 없습니다.")
    st.write("없는 테이블:", ", ".join(missing_tables))
    st.write("현재 DB 테이블:", ", ".join(existing_tables))
    st.stop()


# ============================================================
# 5. 데이터 불러오기
# ============================================================

priority_sql = """
SELECT
    district,
    elderly_total,
    elderly_65_79,
    elderly_80_plus,
    elderly_80_plus_rate,
    vulnerable_elderly_rate,
    shelter_count,
    total_capacity,
    shelters_per_1000,
    elderly_per_shelter,
    capacity_rate,
    priority_type,
    priority_score
FROM district_priority;
"""

summary_sql = """
SELECT
    district,
    elderly_total,
    elderly_80_plus,
    elderly_80_plus_rate,
    vulnerable_elderly_rate,
    shelter_count,
    total_capacity,
    shelters_per_1000,
    elderly_per_shelter,
    capacity_rate
FROM district_summary;
"""

shelter_sql = """
SELECT
    shelter_name,
    district,
    dong_guess,
    facility_type1,
    facility_type2,
    road_address,
    capacity,
    area,
    latitude,
    longitude
FROM shelters
WHERE latitude IS NOT NULL
  AND longitude IS NOT NULL;
"""

priority_df = run_sql(priority_sql)
summary_df = run_sql(summary_sql)
shelter_df = run_sql(shelter_sql)

district_geojson = load_geojson(DISTRICT_GEOJSON_PATH)


# ============================================================
# 6. 사이드바 필터
# ============================================================

st.sidebar.header("필터")

all_districts = sorted(priority_df["district"].dropna().unique().tolist())

top5_order_sql = """
SELECT district
FROM district_priority
ORDER BY priority_score DESC, shelters_per_1000 ASC, capacity_rate ASC, elderly_total DESC
LIMIT 5;
"""
top5_districts = run_sql(top5_order_sql)["district"].tolist()

shelter_view = st.sidebar.radio(
    "무더위쉼터 위치 지도 보기 범위",
    ["전체 보기", "개선 우선지역 TOP 5만 보기", "특정 자치구 보기"],
)

selected_district = st.sidebar.selectbox(
    "특정 자치구 선택",
    all_districts,
    index=0,
)

st.sidebar.divider()
st.sidebar.caption("지도 파일 경로")
st.sidebar.code(
    f"{DISTRICT_GEOJSON_PATH}\n{DONG_GEOJSON_PATH}",
    language="text",
)


# ============================================================
# 1. 프로젝트 소개
# ============================================================

st.title("서울시 독거노인 폭염 취약지역 및 무더위쉼터 공급 불균형 분석")
st.caption("서울시 자치구별 독거노인 수요와 무더위쉼터 공급 수준을 비교하여 개선 우선지역을 찾는 의사결정형 대시보드입니다.")

st.markdown(
    """
### 프로젝트 목적
서울시 자치구별 독거노인 현황과 무더위쉼터 공급 현황을 비교하여 폭염 취약지역을 찾고, 
무더위쉼터 배치 개선이 필요한 우선지역을 제안합니다.

### 핵심 질문
1. 독거노인이 많이 거주하는 자치구는 어디인가?  
2. 무더위쉼터 공급이 독거노인 수요에 비해 부족한 지역은 어디인가?  
3. 쉼터 추가 배치나 수용인원 확대가 필요한 우선지역은 어디인가?
"""
)

with st.expander("분석에 사용한 핵심 SQL 보기"):
    st.code(priority_sql.strip(), language="sql")
    st.code(summary_sql.strip(), language="sql")
    st.code(shelter_sql.strip(), language="sql")

st.divider()


# ============================================================
# 2. 핵심 지표 카드
# ============================================================

st.header("1. 서울시 전체 핵심 지표")

kpi_sql = """
SELECT
    SUM(elderly_total) AS total_elderly,
    SUM(elderly_80_plus) AS total_elderly_80_plus,
    SUM(shelter_count) AS total_shelters,
    SUM(total_capacity) AS total_capacity,
    AVG(capacity_rate) AS avg_capacity_rate
FROM district_summary;
"""

priority_count_sql = """
SELECT COUNT(*) AS priority_area_count
FROM district_priority
WHERE priority_type = '개선 우선지역';
"""

kpi = run_sql(kpi_sql).iloc[0]
priority_area_count = run_sql(priority_count_sql).iloc[0]["priority_area_count"]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("전체 독거노인 수", fmt_int(kpi["total_elderly"]))
    st.caption("폭염 상황에서 보호와 안내가 필요한 핵심 수요 규모입니다.")
with c2:
    st.metric("80세 이상 독거노인 수", fmt_int(kpi["total_elderly_80_plus"]))
    st.caption("연령 취약성이 높은 고령 독거노인 규모입니다.")
with c3:
    st.metric("전체 무더위쉼터 수", fmt_int(kpi["total_shelters"]))
    st.caption("서울시 전역에 등록된 무더위쉼터 공급량입니다.")

c4, c5, c6 = st.columns(3)
with c4:
    st.metric("총 수용가능인원", fmt_int(kpi["total_capacity"]))
    st.caption("쉼터가 실제로 수용할 수 있는 총 인원입니다.")
with c5:
    st.metric("평균 수용률", fmt_pct(kpi["avg_capacity_rate"]))
    st.caption("자치구별 독거노인 수 대비 쉼터 수용가능인원의 평균입니다.")
with c6:
    st.metric("개선 우선지역 수", fmt_int(priority_area_count))
    st.caption("우선유형이 '개선 우선지역'으로 분류된 자치구 수입니다.")

show_sql(kpi_sql + "\n" + priority_count_sql)

st.divider()


# ============================================================
# 3. 서울시 자치구별 개선 우선순위 지도
# ============================================================

st.header("2. 서울시 자치구별 개선 우선순위 지도")
st.caption("priority_score가 높은 자치구일수록 무더위쉼터 공급 개선을 우선적으로 검토해야 하는 지역입니다.")

priority_map_df = priority_df.copy()

priority_tooltip_fields = [
    GEOJSON_DISTRICT_COL,
    "priority_type",
    "priority_score",
    "elderly_total",
    "shelter_count",
    "capacity_rate",
    "shelters_per_1000",
]
priority_tooltip_aliases = [
    "자치구",
    "우선유형",
    "우선순위 점수",
    "독거노인 수",
    "쉼터 수",
    "수용률",
    "1,000명당 쉼터 수",
]

make_choropleth_map(
    geojson_data=district_geojson,
    df=priority_map_df,
    geo_key=GEOJSON_DISTRICT_COL,
    df_key="district",
    value_col="priority_score",
    legend_name="개선 우선순위 점수",
    tooltip_fields=priority_tooltip_fields,
    tooltip_aliases=priority_tooltip_aliases,
    fill_color="YlOrRd",
)

show_sql(priority_sql)
insight_box("왜 이 지도를 보는가", [
    "이 지도는 서울시 전체에서 개선 우선순위가 높은 자치구를 가장 먼저 보여주는 핵심 시각화입니다.",
    "priority_score는 독거노인 수요와 쉼터 공급 부족을 함께 반영하므로, 단순한 시설 수 비교보다 정책 판단에 적합합니다.",
])
insight_box("자동 해석", auto_priority_insights(priority_df.sort_values("priority_score", ascending=False)))

st.divider()


# ============================================================
# 4. 자치구별 무더위쉼터 수 지도
# ============================================================

st.header("3. 자치구별 무더위쉼터 수 지도")
st.caption("자치구별 쉼터 개수 자체를 공간적으로 비교합니다.")

shelter_count_map_sql = """
SELECT district, shelter_count
FROM district_summary
ORDER BY shelter_count DESC;
"""
shelter_count_map_df = run_sql(shelter_count_map_sql)

make_choropleth_map(
    geojson_data=district_geojson,
    df=shelter_count_map_df,
    geo_key=GEOJSON_DISTRICT_COL,
    df_key="district",
    value_col="shelter_count",
    legend_name="무더위쉼터 수",
    tooltip_fields=[GEOJSON_DISTRICT_COL, "shelter_count"],
    tooltip_aliases=["자치구", "쉼터 수"],
    fill_color="YlGnBu",
)

show_sql(shelter_count_map_sql)
if not shelter_count_map_df.empty:
    max_row = shelter_count_map_df.iloc[0]
    min_row = shelter_count_map_df.iloc[-1]
    insight_box("해석", [
        f"쉼터 수가 가장 많은 지역은 {max_row['district']}이며 {fmt_int(max_row['shelter_count'])}개입니다.",
        f"쉼터 수가 가장 적은 지역은 {min_row['district']}이며 {fmt_int(min_row['shelter_count'])}개입니다.",
        "다만 쉼터 수가 많다고 반드시 충분한 것은 아니므로 수용률과 1,000명당 쉼터 수를 함께 봐야 합니다.",
    ])

st.divider()


# ============================================================
# 5. 자치구별 쉼터 수용률 지도
# ============================================================

st.header("4. 자치구별 쉼터 수용률 지도")
st.caption("독거노인 수 대비 무더위쉼터가 몇 %까지 수용 가능한지 보여줍니다.")

capacity_rate_map_sql = """
SELECT district, elderly_total, total_capacity, capacity_rate
FROM district_summary
ORDER BY capacity_rate ASC;
"""
capacity_rate_map_df = run_sql(capacity_rate_map_sql)

make_choropleth_map(
    geojson_data=district_geojson,
    df=capacity_rate_map_df,
    geo_key=GEOJSON_DISTRICT_COL,
    df_key="district",
    value_col="capacity_rate",
    legend_name="독거노인 대비 쉼터 수용률(%)",
    tooltip_fields=[GEOJSON_DISTRICT_COL, "elderly_total", "total_capacity", "capacity_rate"],
    tooltip_aliases=["자치구", "독거노인 수", "총 수용가능인원", "수용률"],
    fill_color="PuBuGn",
)

show_sql(capacity_rate_map_sql)
if not capacity_rate_map_df.empty:
    low_row = capacity_rate_map_df.iloc[0]
    insight_box("해석", [
        f"수용률이 가장 낮은 지역은 {low_row['district']}이며 {fmt_pct(low_row['capacity_rate'])}입니다.",
        "수용률이 낮은 지역은 쉼터가 존재하더라도 실제 수용가능인원이 독거노인 수요를 충분히 감당하기 어려울 수 있습니다.",
    ])

st.divider()


# ============================================================
# 6. 자치구별 독거노인 1,000명당 쉼터 수 지도
# ============================================================

st.header("5. 자치구별 독거노인 1,000명당 쉼터 수 지도")
st.caption("독거노인 수요 대비 쉼터 공급 밀도를 확인합니다.")

per_1000_map_sql = """
SELECT district, elderly_total, shelter_count, shelters_per_1000
FROM district_summary
ORDER BY shelters_per_1000 ASC;
"""
per_1000_map_df = run_sql(per_1000_map_sql)

make_choropleth_map(
    geojson_data=district_geojson,
    df=per_1000_map_df,
    geo_key=GEOJSON_DISTRICT_COL,
    df_key="district",
    value_col="shelters_per_1000",
    legend_name="독거노인 1,000명당 쉼터 수",
    tooltip_fields=[GEOJSON_DISTRICT_COL, "elderly_total", "shelter_count", "shelters_per_1000"],
    tooltip_aliases=["자치구", "독거노인 수", "쉼터 수", "1,000명당 쉼터 수"],
    fill_color="YlGn",
)

show_sql(per_1000_map_sql)
if not per_1000_map_df.empty:
    low_row = per_1000_map_df.iloc[0]
    insight_box("해석", [
        f"독거노인 1,000명당 쉼터 수가 가장 낮은 지역은 {low_row['district']}입니다.",
        "값이 낮을수록 독거노인 수요에 비해 쉼터 개수가 부족할 가능성이 높습니다.",
    ])

st.divider()


# ============================================================
# 7. 지도 3개 비교 설명 영역
# ============================================================

st.header("6. 세 가지 공급 지도는 함께 해석해야 합니다")

compare_cols = st.columns(3)
with compare_cols[0]:
    st.subheader("쉼터 수")
    st.write("자치구별 쉼터 개수 자체를 보여줍니다.")
    st.caption("공급량의 절대 규모 확인")
with compare_cols[1]:
    st.subheader("수용률")
    st.write("독거노인 수 대비 실제 수용가능인원을 보여줍니다.")
    st.caption("실제 수용능력 확인")
with compare_cols[2]:
    st.subheader("1,000명당 쉼터 수")
    st.write("독거노인 수요 대비 쉼터 공급 밀도를 보여줍니다.")
    st.caption("접근성 또는 공급 밀도 확인")

st.info(
    "세 지표를 함께 보면 쉼터가 단순히 적은 지역인지, 쉼터는 있지만 수용능력이 부족한 지역인지, "
    "독거노인 수요 대비 공급 밀도가 낮은 지역인지 구분할 수 있습니다."
)

st.divider()


# ============================================================
# 8. 무더위쉼터 위치 지도
# ============================================================

st.header("7. 무더위쉼터 위치 지도")
st.caption("개별 쉼터가 실제 공간상에서 어디에 분포하는지 확인합니다.")

if shelter_df.empty:
    st.warning("위도와 경도가 있는 무더위쉼터 데이터가 없습니다.")
else:
    if shelter_view == "개선 우선지역 TOP 5만 보기":
        shelter_show = shelter_df[shelter_df["district"].isin(top5_districts)].copy()
    elif shelter_view == "특정 자치구 보기":
        shelter_show = shelter_df[shelter_df["district"] == selected_district].copy()
    else:
        shelter_show = shelter_df.copy()

    if shelter_show.empty:
        st.warning("선택한 조건에 해당하는 쉼터 위치 데이터가 없습니다.")
    else:
        center = [shelter_show["latitude"].mean(), shelter_show["longitude"].mean()]
        shelter_map = folium.Map(location=center, zoom_start=12 if shelter_view == "특정 자치구 보기" else 11, tiles="cartodbpositron")
        cluster = MarkerCluster().add_to(shelter_map)

        for _, row in shelter_show.iterrows():
            capacity_value = safe_num(row.get("capacity"), 0)
            capacity_text = "수용인원 미입력 또는 확인 필요" if capacity_value == 0 else f"{fmt_int(capacity_value)}명"

            popup_html = f"""
            <b>{row.get('shelter_name', '')}</b><br>
            자치구: {row.get('district', '')}<br>
            추정 행정동: {row.get('dong_guess', '')}<br>
            시설유형: {row.get('facility_type1', '')} / {row.get('facility_type2', '')}<br>
            주소: {row.get('road_address', '')}<br>
            수용가능인원: {capacity_text}<br>
            면적: {fmt_float(row.get('area'))}㎡
            """

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=360),
                tooltip=f"{row.get('shelter_name', '')} | {row.get('district', '')}",
            ).add_to(cluster)

        st_folium(shelter_map, width=None, height=560)

show_sql(shelter_sql)
insight_box("왜 이 지도를 보는가", [
    "자치구 단위 지도는 공급 수준을 요약하지만, 위치 지도는 실제 쉼터가 어디에 있는지 보여줍니다.",
    "개선 우선지역 안에서도 쉼터가 특정 행정동에 몰려 있다면 취약 행정동 중심의 재배치 검토가 필요합니다.",
])

st.divider()


# ============================================================
# 9. 수요-공급 불균형 산점도
# ============================================================

st.header("8. 수요-공급 불균형 산점도")
st.caption("독거노인 수요와 쉼터 수용능력 사이의 불균형을 사분면으로 해석합니다.")

scatter_sql = """
SELECT
    district,
    elderly_total,
    capacity_rate,
    shelter_count,
    priority_type,
    shelters_per_1000,
    elderly_per_shelter,
    priority_score
FROM district_priority;
"""
scatter_df = run_sql(scatter_sql)

if not scatter_df.empty and has_columns(scatter_df, ["elderly_total", "capacity_rate", "shelter_count", "priority_type"]):
    avg_elderly = scatter_df["elderly_total"].mean()
    avg_capacity = scatter_df["capacity_rate"].mean()

    fig = px.scatter(
        scatter_df,
        x="elderly_total",
        y="capacity_rate",
        size="shelter_count",
        color="priority_type",
        hover_name="district",
        hover_data={
            "shelters_per_1000": ":.2f",
            "elderly_per_shelter": ":.2f",
            "priority_score": True,
            "shelter_count": True,
            "elderly_total": ":,",
            "capacity_rate": ":.2f",
        },
        title="독거노인 수요와 쉼터 수용률의 관계",
        labels={
            "elderly_total": "독거노인 수",
            "capacity_rate": "쉼터 수용률(%)",
            "shelter_count": "쉼터 수",
            "priority_type": "우선유형",
        },
        size_max=45,
    )

    fig.add_vline(x=avg_elderly, line_dash="dash", line_width=1)
    fig.add_hline(y=avg_capacity, line_dash="dash", line_width=1)

    fig.add_annotation(
        x=scatter_df["elderly_total"].max(),
        y=scatter_df["capacity_rate"].min(),
        text="독거노인 많음 + 수용률 낮음<br>가장 취약한 영역",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
    )

    fig.update_layout(height=620, margin=dict(l=20, r=20, t=70, b=40))
    st.plotly_chart(fig, use_container_width=True)

show_sql(scatter_sql)
insight_box("해석 기준", [
    "오른쪽 아래는 독거노인은 많지만 수용률이 낮은 지역으로 가장 취약합니다.",
    "오른쪽 위는 독거노인은 많지만 수용능력도 비교적 있는 지역입니다.",
    "왼쪽 아래는 독거노인 규모는 작지만 수용률이 부족한 지역입니다.",
    "왼쪽 위는 상대적으로 안정적인 지역입니다.",
])

st.divider()


# ============================================================
# 10. 개선 우선지역 TOP 5
# ============================================================

st.header("9. 개선 우선지역 TOP 5")
st.caption("우선순위 점수와 수요-공급 지표를 함께 고려하여 개선이 필요한 자치구를 도출합니다.")

top5_sql = """
SELECT
    district,
    priority_type,
    priority_score,
    elderly_total,
    elderly_80_plus_rate,
    vulnerable_elderly_rate,
    shelter_count,
    total_capacity,
    shelters_per_1000,
    capacity_rate,
    elderly_per_shelter
FROM district_priority
ORDER BY priority_score DESC, shelters_per_1000 ASC, capacity_rate ASC, elderly_total DESC
LIMIT 5;
"""

top5_df = run_sql(top5_sql)

if top5_df.empty:
    st.warning("개선 우선지역 TOP 5 데이터가 없습니다.")
else:
    st.dataframe(format_table(top5_df), use_container_width=True, hide_index=True)

    bar_fig = px.bar(
        top5_df,
        x="district",
        y="priority_score",
        text="priority_score",
        title="개선 우선지역 TOP 5 우선순위 점수",
        labels={"district": "자치구", "priority_score": "우선순위 점수"},
    )
    bar_fig.update_traces(textposition="outside")
    bar_fig.update_layout(height=420)
    st.plotly_chart(bar_fig, use_container_width=True)

    avg_capacity_rate = priority_df["capacity_rate"].mean()
    avg_per_1000 = priority_df["shelters_per_1000"].mean()
    avg_80_rate = priority_df["elderly_80_plus_rate"].mean()

    st.subheader("지역별 취약 원인 해석")
    for _, row in top5_df.iterrows():
        st.markdown(f"**{row['district']} | {row['priority_type']} | 점수 {fmt_int(row['priority_score'])}점**")
        st.caption(district_reason(row, avg_capacity_rate, avg_per_1000, avg_80_rate))

show_sql(top5_sql)
insight_box("TOP 5를 이렇게 해석해야 하는 이유", [
    "순위는 단순 쉼터 개수만으로 정한 것이 아니라 독거노인 수요, 공급 밀도, 수용률을 함께 고려한 결과입니다.",
    "따라서 상위 지역은 쉼터 추가 배치, 기존 시설 수용인원 확대, 고령 독거노인 안내 강화 등을 우선 검토할 필요가 있습니다.",
])

st.divider()


# ============================================================
# 11. 결과 해석 및 정책 제안
# ============================================================

st.header("10. 결과 해석 및 정책 제안")

policy_df = pd.DataFrame({
    "유형": [
        "수용능력 부족형",
        "쉼터 수 부족형",
        "고령 취약성 주의형",
        "공간 분포 불균형형",
        "안내 강화 필요형",
    ],
    "판단 기준": [
        "capacity_rate 낮음",
        "shelters_per_1000 낮음",
        "elderly_80_plus_rate 높음",
        "쉼터 위치가 특정 지역에 집중",
        "쉼터는 있으나 이용 접근성이 낮을 가능성",
    ],
    "개선 방향": [
        "대형 공공시설 추가 지정, 기존 쉼터 수용인원 확대",
        "쉼터 추가 배치, 접근성 취약지역 중심 보강",
        "80세 이상 독거노인 대상 폭염 안내와 방문 점검 강화",
        "취약 행정동 중심 재배치 검토",
        "쉼터 위치 안내, 문자 알림, 주민센터 연계 홍보 강화",
    ],
})

st.dataframe(policy_df, use_container_width=True, hide_index=True)

st.success(
    "서울시 무더위쉼터 정책은 단순히 쉼터 수를 늘리는 방식보다, 독거노인 밀집도, 수용가능인원, "
    "쉼터 접근성을 함께 고려해 자치구별로 다르게 설계될 필요가 있습니다."
)

with st.expander("전체 데이터 확인"):
    st.subheader("district_priority")
    st.dataframe(format_table(priority_df.sort_values("priority_score", ascending=False)), use_container_width=True)

    st.subheader("district_summary")
    st.dataframe(format_table(summary_df), use_container_width=True)
