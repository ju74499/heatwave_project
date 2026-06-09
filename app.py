# app.py
# ------------------------------------------------------------
# 서울시 독거노인 폭염 취약지역 및 무더위쉼터 공급 분석 대시보드
# ------------------------------------------------------------
# 실행 방법:
# 1) final.db 파일을 app.py와 같은 폴더에 둡니다.
# 2) 터미널에서 streamlit run app.py 를 실행합니다.
# 3) 지도 파일이 있다면 data 폴더 안에 GeoJSON 파일을 넣습니다.
# ------------------------------------------------------------

import os
import json
import copy
import sqlite3
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ============================================================
# 1. 사용자가 쉽게 수정할 수 있는 설정값
# ============================================================

DB_PATH = "final.db"
DISTRICT_GEOJSON_PATH = "data/seoul_district_boundary_simplified.geojson"
DONG_GEOJSON_PATH = "data/seoul_adm_dong_simplified.geojson"

# GeoJSON 안에서 자치구명을 담고 있는 컬럼명
# 예: "district", "SIGUNGU_NM", "시군구명칭", "SIG_KOR_NM" 등
GEOJSON_DISTRICT_COL = "district"

# 행정동 GeoJSON 안에서 자치구명과 행정동명을 담고 있는 컬럼명
# 예: "district", "dong", "ADM_DR_NM", "행정동명칭" 등
GEOJSON_DONG_DISTRICT_COL = "district"
GEOJSON_DONG_COL = "ADM_NM"

REQUIRED_TABLES = [
    "weather_summary",
    "weather_hourly",
    "district_summary",
    "shelter_district",
    "district_priority",
    "shelters",
    "elderly_district",
    "elderly_dong",
]

# ============================================================
# 2. 기본 설정
# ============================================================

st.set_page_config(
    page_title="서울시 독거노인 폭염 취약지역 분석",
    page_icon="🌡️",
    layout="wide",
)

# ============================================================
# 3. 공통 함수
# ============================================================

def fmt_int(value: Any) -> str:
    """정수를 천 단위 콤마로 표시합니다."""
    if pd.isna(value):
        return "-"
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return str(value)


def fmt_float(value: Any, digits: int = 2) -> str:
    """소수를 보기 좋게 표시합니다."""
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def fmt_percent(value: Any) -> str:
    """비율 값을 %로 표시합니다. DB에는 이미 0~100 범위 값이 들어 있다고 가정합니다."""
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return str(value)


def db_exists() -> bool:
    return os.path.exists(DB_PATH)


@st.cache_data(show_spinner=False)
def get_table_names(db_path: str) -> List[str]:
    """SQLite DB 안의 테이블 목록을 가져옵니다."""
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
        conn,
    )["name"].tolist()
    conn.close()
    return tables


@st.cache_data(show_spinner=False)
def run_query(sql: str, params: Optional[Tuple[Any, ...]] = None) -> pd.DataFrame:
    """SQL을 실행하고 결과를 DataFrame으로 반환합니다."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def show_sql(sql: str) -> None:
    """화면에 SQL을 접을 수 있는 형태로 보여줍니다."""
    with st.expander("사용한 SQL 보기"):
        st.code(sql.strip(), language="sql")


def show_insight(lines: List[str]) -> None:
    """인사이트 문장을 박스로 보여줍니다."""
    st.markdown("**인사이트**")
    for line in lines:
        st.info(line)


def safe_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    x_label: str,
    y_label: str,
    text_col: Optional[str] = None,
) -> None:
    """Plotly 막대그래프를 안전하게 그립니다."""
    if df.empty:
        st.warning("표시할 데이터가 없습니다.")
        return

    fig = px.bar(
        df,
        x=x,
        y=y,
        text=text_col or y,
        title=title,
        labels={x: x_label, y: y_label},
    )
    fig.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        margin=dict(l=20, r=20, t=70, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)


def table_with_format(df: pd.DataFrame) -> pd.DataFrame:
    """표시용으로 숫자 포맷을 적용한 복사본을 만듭니다."""
    result = df.copy()
    int_cols = [
        "elderly_total",
        "elderly_65_79",
        "elderly_80_plus",
        "shelter_count",
        "total_capacity",
        "capacity_missing_count",
        "area_missing_count",
        "public_facility_count",
        "senior_facility_count",
        "priority_score",
    ]
    percent_cols = [
        "elderly_80_plus_rate",
        "vulnerable_elderly_rate",
        "capacity_rate",
        "elderly_80_plus_ratio",
    ]
    float_cols = ["shelters_per_1000", "elderly_per_shelter", "avg_capacity", "avg_area", "total_area"]

    for col in result.columns:
        if col in int_cols:
            result[col] = result[col].apply(fmt_int)
        elif col in percent_cols:
            result[col] = result[col].apply(fmt_percent)
        elif col in float_cols:
            result[col] = result[col].apply(lambda v: fmt_float(v, 2))
    return result


def get_filter_condition(selected_district: str, top5_only: bool) -> str:
    """자치구 선택과 TOP5 옵션에 따라 SQL WHERE 조건을 만듭니다."""
    conditions = []
    if selected_district != "전체":
        conditions.append(f"district = '{selected_district}'")
    if top5_only:
        conditions.append(
            "district IN ("
            "SELECT district FROM district_priority "
            "ORDER BY priority_score DESC, shelters_per_1000 ASC, capacity_rate ASC, elderly_total DESC "
            "LIMIT 5)"
        )
    if not conditions:
        return ""
    return "WHERE " + " AND ".join(conditions)


def load_geojson(path: str) -> Optional[Dict[str, Any]]:
    """GeoJSON 파일을 읽습니다. 파일이 없으면 None을 반환합니다."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp949") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"GeoJSON 파일을 읽는 중 오류가 발생했습니다: {e}")
        return None


def enrich_geojson_single_key(
    geojson_data: Dict[str, Any],
    df: pd.DataFrame,
    geo_key: str,
    df_key: str,
    value_cols: List[str],
) -> Dict[str, Any]:
    """GeoJSON feature properties에 DB 값을 붙입니다."""
    enriched = copy.deepcopy(geojson_data)
    lookup = df.set_index(df_key)[value_cols].to_dict("index") if not df.empty else {}

    for feature in enriched.get("features", []):
        props = feature.get("properties", {})
        key = props.get(geo_key)
        values = lookup.get(key, {})
        for col in value_cols:
            props[col] = values.get(col, None)
        feature["properties"] = props
    return enriched


def add_join_key_to_dong_geojson(
    geojson_data: Dict[str, Any],
    district_col: str,
    dong_col: str,
) -> Dict[str, Any]:
    """행정동 지도 결합용 join_key를 GeoJSON에 추가합니다."""
    enriched = copy.deepcopy(geojson_data)
    for feature in enriched.get("features", []):
        props = feature.get("properties", {})
        district = props.get(district_col, "")
        dong = props.get(dong_col, "")
        props["join_key"] = f"{district}|{dong}"
        feature["properties"] = props
    return enriched

# ============================================================
# 4. DB 점검
# ============================================================

st.title("🌡️ 서울시 독거노인 폭염 취약지역 및 무더위쉼터 공급 분석")
st.caption("서울시 자치구별 독거노인 현황과 무더위쉼터 공급 현황을 비교하여 폭염 대응 인프라가 부족한 지역을 찾는 대시보드입니다.")

if not db_exists():
    st.error("final.db 파일을 찾을 수 없습니다. app.py와 같은 폴더에 final.db를 넣은 뒤 다시 실행해 주세요.")
    st.stop()

existing_tables = get_table_names(DB_PATH)
missing_tables = [t for t in REQUIRED_TABLES if t not in existing_tables]

if missing_tables:
    st.error("필요한 테이블이 DB에 없습니다.")
    st.write("없는 테이블:", ", ".join(missing_tables))
    st.write("현재 DB 테이블:", ", ".join(existing_tables))
    st.stop()

# ============================================================
# 5. 사이드바 필터
# ============================================================

districts_df = run_query("SELECT district FROM district_summary ORDER BY district;")
district_list = districts_df["district"].dropna().tolist()

top5_sql = """
SELECT district
FROM district_priority
ORDER BY priority_score DESC, shelters_per_1000 ASC, capacity_rate ASC, elderly_total DESC
LIMIT 5;
"""
top5_districts = run_query(top5_sql)["district"].tolist()
default_dong_district = top5_districts[0] if top5_districts else (district_list[0] if district_list else "")

st.sidebar.header("🔎 필터")
selected_district = st.sidebar.selectbox("자치구 선택", ["전체"] + district_list)
top5_only = st.sidebar.checkbox("개선 우선지역 TOP 5만 보기", value=False)
dong_map_district = st.sidebar.selectbox(
    "행정동 지도용 자치구 선택",
    district_list,
    index=district_list.index(default_dong_district) if default_dong_district in district_list else 0,
)

filter_condition = get_filter_condition(selected_district, top5_only)

# ============================================================
# 6. 핵심 지표 카드
# ============================================================

st.header("1. 핵심 요약 지표")

summary_sql = """
SELECT
    SUM(elderly_total) AS total_elderly,
    SUM(elderly_80_plus) AS total_elderly_80_plus,
    SUM(shelter_count) AS total_shelters,
    SUM(total_capacity) AS total_capacity,
    AVG(shelters_per_1000) AS avg_shelters_per_1000,
    AVG(capacity_rate) AS avg_capacity_rate
FROM district_summary;
"""
priority_count_sql = """
SELECT COUNT(*) AS priority_area_count
FROM district_priority
WHERE priority_type = '개선 우선지역';
"""
summary = run_query(summary_sql).iloc[0]
priority_count = run_query(priority_count_sql).iloc[0]["priority_area_count"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("서울시 전체 독거노인", fmt_int(summary["total_elderly"]))
c2.metric("80세 이상 독거노인", fmt_int(summary["total_elderly_80_plus"]))
c3.metric("무더위쉼터 수", fmt_int(summary["total_shelters"]))
c4.metric("총 수용가능인원", fmt_int(summary["total_capacity"]))

c5, c6, c7 = st.columns(3)
c5.metric("평균 1,000명당 쉼터 수", fmt_float(summary["avg_shelters_per_1000"]))
c6.metric("평균 수용률", fmt_percent(summary["avg_capacity_rate"]))
c7.metric("개선 우선지역 수", fmt_int(priority_count))

show_sql(summary_sql + "\n" + priority_count_sql)
show_insight([
    "독거노인 규모와 쉼터 공급량을 함께 보면 단순 시설 수가 아니라 수요 대비 공급 수준을 판단할 수 있습니다.",
    "평균 수용률과 1,000명당 쉼터 수는 이후 자치구별 취약지역을 비교하는 기준 지표로 활용됩니다.",
])

# ============================================================
# 7. 서울 폭염 배경
# ============================================================

st.header("2. 서울 폭염 배경")

weather_summary_sql = """
SELECT
    station_name,
    period_start,
    period_end,
    hourly_count,
    avg_temperature_c,
    max_temperature_c,
    min_temperature_c,
    avg_ground_temperature_c,
    max_ground_temperature_c
FROM weather_summary;
"""
weather_hourly_sql = """
SELECT
    datetime,
    temperature_c,
    ground_temperature_c
FROM weather_hourly
ORDER BY datetime;
"""
weather_summary = run_query(weather_summary_sql)
weather_hourly = run_query(weather_hourly_sql)

if not weather_summary.empty:
    w = weather_summary.iloc[0]
    wc1, wc2, wc3, wc4, wc5 = st.columns(5)
    wc1.metric("평균기온", f"{fmt_float(w['avg_temperature_c'])}℃")
    wc2.metric("최고기온", f"{fmt_float(w['max_temperature_c'])}℃")
    wc3.metric("최저기온", f"{fmt_float(w['min_temperature_c'])}℃")
    wc4.metric("평균 지면온도", f"{fmt_float(w['avg_ground_temperature_c'])}℃")
    wc5.metric("최고 지면온도", f"{fmt_float(w['max_ground_temperature_c'])}℃")
    st.caption(f"관측지점: {w['station_name']} / 관측기간: {w['period_start']} ~ {w['period_end']} / 관측 수: {fmt_int(w['hourly_count'])}건")

if not weather_hourly.empty:
    weather_hourly["datetime"] = pd.to_datetime(weather_hourly["datetime"])
    weather_long = weather_hourly.melt(
        id_vars="datetime",
        value_vars=["temperature_c", "ground_temperature_c"],
        var_name="구분",
        value_name="온도",
    )
    weather_long["구분"] = weather_long["구분"].replace({
        "temperature_c": "기온",
        "ground_temperature_c": "지면온도",
    })
    fig = px.line(
        weather_long,
        x="datetime",
        y="온도",
        color="구분",
        markers=True,
        title="시간대별 기온 및 지면온도 변화",
        labels={"datetime": "시간", "온도": "온도(℃)", "구분": "구분"},
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("weather_hourly 테이블에 표시할 데이터가 없습니다.")

show_sql(weather_summary_sql + "\n" + weather_hourly_sql)
show_insight([
    "기상 데이터는 폭염 문제가 왜 중요한지 보여주는 배경 자료입니다.",
    "실제 취약지역 도출은 독거노인 규모와 무더위쉼터 공급 지표를 중심으로 진행됩니다.",
])

# ============================================================
# 8. 자치구별 주요 그래프
# ============================================================

st.header("3. 자치구별 독거노인 및 무더위쉼터 공급 분석")

# 8-1 독거노인 수
sql_elderly_total = f"""
SELECT district, elderly_total
FROM district_summary
{filter_condition}
ORDER BY elderly_total DESC;
"""
df_elderly_total = run_query(sql_elderly_total)
safe_bar_chart(df_elderly_total, "district", "elderly_total", "자치구별 독거노인 수", "자치구", "독거노인 수", "elderly_total")
show_sql(sql_elderly_total)
if not df_elderly_total.empty:
    top = df_elderly_total.iloc[0]
    show_insight([
        f"독거노인 수가 가장 많은 지역은 {top['district']}이며, {fmt_int(top['elderly_total'])}명입니다.",
        "독거노인 수가 많은 지역은 폭염 발생 시 돌봄과 쉼터 안내가 우선적으로 필요한 지역으로 볼 수 있습니다.",
    ])

# 8-2 80세 이상 비율
sql_80_rate = f"""
SELECT district, elderly_80_plus_rate
FROM district_summary
{filter_condition}
ORDER BY elderly_80_plus_rate DESC;
"""
df_80_rate = run_query(sql_80_rate)
safe_bar_chart(df_80_rate, "district", "elderly_80_plus_rate", "자치구별 80세 이상 독거노인 비율", "자치구", "80세 이상 비율(%)", "elderly_80_plus_rate")
show_sql(sql_80_rate)
if not df_80_rate.empty:
    top = df_80_rate.iloc[0]
    show_insight([
        f"80세 이상 독거노인 비율이 가장 높은 지역은 {top['district']}이며, {fmt_percent(top['elderly_80_plus_rate'])}입니다.",
        "고령 비율이 높은 지역은 같은 독거노인 규모라도 폭염 대응 위험도가 더 높게 해석될 수 있습니다.",
    ])

# 8-3 취약 독거노인 비율
sql_vulnerable_rate = f"""
SELECT district, vulnerable_elderly_rate
FROM district_summary
{filter_condition}
ORDER BY vulnerable_elderly_rate DESC;
"""
df_vulnerable_rate = run_query(sql_vulnerable_rate)
safe_bar_chart(df_vulnerable_rate, "district", "vulnerable_elderly_rate", "자치구별 취약 독거노인 비율", "자치구", "취약 독거노인 비율(%)", "vulnerable_elderly_rate")
show_sql(sql_vulnerable_rate)
if not df_vulnerable_rate.empty:
    top = df_vulnerable_rate.iloc[0]
    show_insight([
        f"취약 독거노인 비율이 가장 높은 지역은 {top['district']}이며, {fmt_percent(top['vulnerable_elderly_rate'])}입니다.",
        "기초생활보장 수급자와 저소득 노인 비율은 냉방시설 이용이나 위기 대응 여건을 해석하는 보조 지표로 활용할 수 있습니다.",
    ])

# 8-4 쉼터 수
sql_shelter_count = f"""
SELECT district, shelter_count
FROM district_summary
{filter_condition}
ORDER BY shelter_count DESC;
"""
df_shelter_count = run_query(sql_shelter_count)
safe_bar_chart(df_shelter_count, "district", "shelter_count", "자치구별 무더위쉼터 수", "자치구", "쉼터 수", "shelter_count")
show_sql(sql_shelter_count)
if not df_shelter_count.empty:
    top = df_shelter_count.iloc[0]
    show_insight([
        f"무더위쉼터 수가 가장 많은 지역은 {top['district']}이며, {fmt_int(top['shelter_count'])}개입니다.",
        "다만 쉼터 수만으로는 충분성을 판단하기 어렵기 때문에 독거노인 수와 수용가능인원을 함께 비교해야 합니다.",
    ])

# 8-5 총 수용가능인원
sql_capacity = f"""
SELECT district, total_capacity
FROM district_summary
{filter_condition}
ORDER BY total_capacity DESC;
"""
df_capacity = run_query(sql_capacity)
safe_bar_chart(df_capacity, "district", "total_capacity", "자치구별 총 수용가능인원", "자치구", "총 수용가능인원", "total_capacity")
show_sql(sql_capacity)
if not df_capacity.empty:
    top = df_capacity.iloc[0]
    show_insight([
        f"총 수용가능인원이 가장 큰 지역은 {top['district']}이며, {fmt_int(top['total_capacity'])}명입니다.",
        "쉼터 수가 많더라도 개별 시설 규모가 작으면 실제 수용능력은 낮을 수 있습니다.",
    ])

# 8-6 1,000명당 쉼터 수
sql_per_1000 = f"""
SELECT district, elderly_total, shelter_count, shelters_per_1000
FROM district_summary
{filter_condition}
ORDER BY shelters_per_1000 ASC;
"""
df_per_1000 = run_query(sql_per_1000)
safe_bar_chart(df_per_1000, "district", "shelters_per_1000", "독거노인 1,000명당 무더위쉼터 수", "자치구", "1,000명당 쉼터 수", "shelters_per_1000")
show_sql(sql_per_1000)
if not df_per_1000.empty:
    low = df_per_1000.iloc[0]
    show_insight([
        f"독거노인 1,000명당 쉼터 수가 가장 낮은 지역은 {low['district']}입니다.",
        "이 값이 낮을수록 독거노인 수에 비해 쉼터 접근성이 부족할 가능성이 있습니다.",
    ])

# 8-7 수용률
sql_capacity_rate = f"""
SELECT district, elderly_total, total_capacity, capacity_rate
FROM district_summary
{filter_condition}
ORDER BY capacity_rate ASC;
"""
df_capacity_rate = run_query(sql_capacity_rate)
safe_bar_chart(df_capacity_rate, "district", "capacity_rate", "독거노인 대비 쉼터 수용률", "자치구", "수용률(%)", "capacity_rate")
show_sql(sql_capacity_rate)
if not df_capacity_rate.empty:
    low = df_capacity_rate.iloc[0]
    show_insight([
        f"수용률이 가장 낮은 지역은 {low['district']}이며, {fmt_percent(low['capacity_rate'])}입니다.",
        "수용률이 낮은 지역은 쉼터가 있더라도 실제 수요를 충분히 감당하기 어려울 수 있습니다.",
    ])

# 8-8 쉼터 1개당 독거노인 수
sql_per_shelter = f"""
SELECT district, elderly_total, shelter_count, elderly_per_shelter
FROM district_summary
{filter_condition}
ORDER BY elderly_per_shelter DESC;
"""
df_per_shelter = run_query(sql_per_shelter)
safe_bar_chart(df_per_shelter, "district", "elderly_per_shelter", "쉼터 1개당 담당 독거노인 수", "자치구", "쉼터 1개당 독거노인 수", "elderly_per_shelter")
show_sql(sql_per_shelter)
if not df_per_shelter.empty:
    high = df_per_shelter.iloc[0]
    show_insight([
        f"쉼터 1개당 담당 독거노인 수가 가장 많은 지역은 {high['district']}입니다.",
        "이 값이 높을수록 쉼터 1곳이 감당해야 하는 잠재 수요가 많다는 의미입니다.",
    ])

# ============================================================
# 9. 개선 우선지역 TOP 5
# ============================================================

st.header("4. 개선 우선지역 TOP 5")

top5_detail_sql = """
SELECT
    district,
    elderly_total,
    elderly_80_plus_rate,
    vulnerable_elderly_rate,
    shelter_count,
    total_capacity,
    shelters_per_1000,
    elderly_per_shelter,
    capacity_rate,
    priority_type,
    priority_score
FROM district_priority
ORDER BY priority_score DESC, shelters_per_1000 ASC, capacity_rate ASC, elderly_total DESC
LIMIT 5;
"""
top5_df = run_query(top5_detail_sql)

if top5_df.empty:
    st.warning("개선 우선지역 TOP 5 데이터가 없습니다.")
else:
    st.dataframe(table_with_format(top5_df), use_container_width=True)

    safe_bar_chart(top5_df, "district", "priority_score", "개선 우선지역 TOP 5 취약도 점수", "자치구", "우선순위 점수", "priority_score")

    compare_df = top5_df[["district", "capacity_rate", "shelters_per_1000"]].copy()
    compare_long = compare_df.melt(id_vars="district", var_name="지표", value_name="값")
    compare_long["지표"] = compare_long["지표"].replace({
        "capacity_rate": "수용률(%)",
        "shelters_per_1000": "1,000명당 쉼터 수",
    })
    fig = px.bar(
        compare_long,
        x="district",
        y="값",
        color="지표",
        barmode="group",
        title="TOP 5 지역의 수용률과 1,000명당 쉼터 수 비교",
        labels={"district": "자치구", "값": "지표값", "지표": "지표"},
    )
    st.plotly_chart(fig, use_container_width=True)

show_sql(top5_detail_sql)
show_insight([
    "개선 우선지역은 독거노인 규모, 고령 취약성, 쉼터 접근성, 수용률을 함께 고려해 도출했습니다.",
    "TOP 5 지역은 단순히 쉼터 수가 적은 곳이 아니라 수요 대비 공급 수준이 낮은 지역으로 해석하는 것이 중요합니다.",
    "각 지역은 쉼터 추가 지정, 수용가능인원 확대, 독거노인 대상 안내 강화 등 맞춤형 개선이 필요할 수 있습니다.",
])

# ============================================================
# 10. 자치구별 취약도 지도
# ============================================================

st.header("5. 자치구별 취약도 지도")

district_geojson = load_geojson(DISTRICT_GEOJSON_PATH)
if district_geojson is None:
    st.warning("자치구 경계 GeoJSON 파일이 없어 지도 시각화를 건너뜁니다. data/seoul_district_boundary_simplified.geojson 파일을 추가하면 지도가 표시됩니다.")
else:
    map_df_sql = """
    SELECT
        district,
        priority_type,
        priority_score,
        elderly_total,
        shelter_count,
        capacity_rate,
        shelters_per_1000
    FROM district_priority;
    """
    map_df = run_query(map_df_sql)
    value_cols = ["priority_type", "priority_score", "elderly_total", "shelter_count", "capacity_rate", "shelters_per_1000"]
    enriched_geojson = enrich_geojson_single_key(district_geojson, map_df, GEOJSON_DISTRICT_COL, "district", value_cols)

    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=enriched_geojson,
        data=map_df,
        columns=["district", "priority_score"],
        key_on=f"feature.properties.{GEOJSON_DISTRICT_COL}",
        fill_opacity=0.75,
        line_opacity=0.5,
        legend_name="취약도 점수",
    ).add_to(m)
    folium.GeoJson(
        enriched_geojson,
        name="자치구 정보",
        tooltip=folium.GeoJsonTooltip(
            fields=[GEOJSON_DISTRICT_COL, "priority_type", "priority_score", "elderly_total", "shelter_count", "capacity_rate", "shelters_per_1000"],
            aliases=["자치구", "유형", "점수", "독거노인 수", "쉼터 수", "수용률", "1,000명당 쉼터 수"],
            localize=True,
        ),
    ).add_to(m)
    st_folium(m, width=None, height=550)
    show_sql(map_df_sql)
    show_insight([
        "지도는 자치구별 priority_score를 기준으로 폭염 취약도를 시각화한 것입니다.",
        "색이 진한 지역은 독거노인 수요 대비 쉼터 공급과 수용능력을 우선적으로 점검할 필요가 있습니다.",
    ])

# ============================================================
# 11. 무더위쉼터 위치 지도
# ============================================================

st.header("6. 무더위쉼터 위치 지도")

shelter_map_sql = """
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
shelter_points = run_query(shelter_map_sql)

if shelter_points.empty:
    st.warning("지도에 표시할 무더위쉼터 위치 데이터가 없습니다.")
else:
    if selected_district != "전체":
        shelter_points_show = shelter_points[shelter_points["district"] == selected_district].copy()
    elif top5_only:
        shelter_points_show = shelter_points[shelter_points["district"].isin(top5_districts)].copy()
    else:
        shelter_points_show = shelter_points.copy()

    center_lat = shelter_points_show["latitude"].mean() if not shelter_points_show.empty else 37.5665
    center_lon = shelter_points_show["longitude"].mean() if not shelter_points_show.empty else 126.9780
    m2 = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")
    cluster = MarkerCluster().add_to(m2)

    for _, row in shelter_points_show.iterrows():
        capacity_text = "수용인원 미입력 또는 확인 필요" if row.get("capacity", 0) == 0 else f"{fmt_int(row.get('capacity'))}명"
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
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=row.get("shelter_name", "무더위쉼터"),
        ).add_to(cluster)

    st_folium(m2, width=None, height=550)

show_sql(shelter_map_sql)
show_insight([
    "무더위쉼터 위치 지도는 쉼터가 서울시 공간상에 어떻게 분포하는지 보여줍니다.",
    "쉼터가 특정 지역에 몰려 있거나 취약지역과 떨어져 있다면 추가 배치나 안내 강화가 필요할 수 있습니다.",
])

# ============================================================
# 12. 행정동별 독거노인 분포 지도
# ============================================================

st.header("7. 행정동별 독거노인 분포 지도")
st.caption("개선 우선지역 내부에서 어느 행정동에 독거노인이 많이 분포하는지 확인하는 심층 분석입니다.")

dong_geojson = load_geojson(DONG_GEOJSON_PATH)
dong_sql = """
SELECT
    district,
    dong,
    elderly_total,
    elderly_80_plus,
    elderly_80_plus_ratio
FROM elderly_dong
WHERE district = ?
ORDER BY elderly_total DESC;
"""
dong_df = run_query(dong_sql, params=(dong_map_district,))

if dong_df.empty:
    st.warning(f"{dong_map_district}의 행정동별 독거노인 데이터가 없습니다.")
else:
    st.subheader(f"{dong_map_district} 행정동별 독거노인 TOP 10")
    st.dataframe(table_with_format(dong_df.head(10)), use_container_width=True)

if dong_geojson is None:
    st.warning("행정동 경계 GeoJSON 파일이 없어 행정동 지도 시각화를 건너뜁니다. data/seoul_adm_dong_simplified.geojson 파일을 추가하면 지도가 표시됩니다.")
else:
    dong_geojson_with_key = add_join_key_to_dong_geojson(
        dong_geojson,
        GEOJSON_DONG_DISTRICT_COL,
        GEOJSON_DONG_COL,
    )
    dong_df_map = dong_df.copy()
    dong_df_map["join_key"] = dong_df_map["district"] + "|" + dong_df_map["dong"]
    value_cols = ["district", "dong", "elderly_total", "elderly_80_plus", "elderly_80_plus_ratio"]
    enriched_dong_geojson = enrich_geojson_single_key(dong_geojson_with_key, dong_df_map, "join_key", "join_key", value_cols)

    m3 = folium.Map(location=[37.5665, 126.9780], zoom_start=12, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=enriched_dong_geojson,
        data=dong_df_map,
        columns=["join_key", "elderly_total"],
        key_on="feature.properties.join_key",
        fill_opacity=0.75,
        line_opacity=0.4,
        legend_name="행정동별 독거노인 수",
    ).add_to(m3)
    folium.GeoJson(
        enriched_dong_geojson,
        tooltip=folium.GeoJsonTooltip(
            fields=[GEOJSON_DONG_DISTRICT_COL, GEOJSON_DONG_COL, "elderly_total", "elderly_80_plus", "elderly_80_plus_ratio"],
            aliases=["자치구", "행정동", "독거노인 수", "80세 이상", "80세 이상 비율"],
            localize=True,
        ),
    ).add_to(m3)
    st_folium(m3, width=None, height=550)

show_sql(dong_sql.replace("?", f"'{dong_map_district}'"))
show_insight([
    f"{dong_map_district} 내부에서도 행정동별 독거노인 규모는 다르게 나타날 수 있습니다.",
    "자치구 단위 분석으로 우선지역을 찾은 뒤, 행정동 단위 분석을 통해 실제 보강이 필요한 세부 지역을 확인할 수 있습니다.",
])

# ============================================================
# 13. 원본 데이터 확인용
# ============================================================

st.header("8. 데이터 확인")
with st.expander("DB 테이블 목록 보기"):
    st.write(existing_tables)

with st.expander("district_priority 원본 데이터 보기"):
    raw_priority = run_query("SELECT * FROM district_priority ORDER BY priority_score DESC, shelters_per_1000 ASC;")
    st.dataframe(table_with_format(raw_priority), use_container_width=True)
