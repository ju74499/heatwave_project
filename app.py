# -*- coding: utf-8 -*-
"""
서울시 독거노인 폭염 취약지역 분석 대시보드
- 핵심 목표: 자치구별 독거노인 수요와 무더위쉼터 공급의 불균형을 찾아 개선 우선지역을 제안
- 주요 데이터: SQLite DB(final.db 또는 heatwave_sql_final.db), 온열질환 CSV/XLSX, 자치구 GeoJSON
"""

import os
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 0. 기본 설정
# =========================================================
st.set_page_config(
    page_title="서울시 독거노인 폭염 취약지역 분석",
    page_icon="🌡️",
    layout="wide",
)

DB_CANDIDATES = ["final.db", "heatwave_sql_final.db", "heatwave_shelter_analysis.db"]
HEAT_CSV_CANDIDATES = ["heat_illness.csv", "온열질환.csv", "온열질환 발생현황.csv"]
HEAT_XLSX_CANDIDATES = ["heat_illness.xlsx", "온열질환.xlsx", "온열질환 발생현황.xlsx"]
DISTRICT_GEOJSON_PATH = "data/seoul_district_boundary_simplified.geojson"
ADM_DONG_GEOJSON_PATH = "data/seoul_adm_dong_simplified.geojson"  # 행정동 지도는 기본적으로 사용하지 않음

# Plotly 기본 템플릿
px.defaults.template = "plotly_white"

# =========================================================
# 1. 공통 함수
# =========================================================
def find_existing_file(candidates):
    """후보 파일명 중 현재 폴더에 존재하는 첫 번째 파일을 반환한다."""
    for file_name in candidates:
        if Path(file_name).exists():
            return file_name
    return None


@st.cache_resource
def get_connection(db_path: str):
    """SQLite DB 연결 객체를 만든다."""
    return sqlite3.connect(db_path, check_same_thread=False)


@st.cache_data(show_spinner=False)
def run_sql(db_path: str, query: str) -> pd.DataFrame:
    """SQL 실행 결과를 DataFrame으로 반환한다."""
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_geojson(path: str):
    """GeoJSON 파일을 읽는다. 파일이 없으면 None을 반환한다."""
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_heat_illness_data() -> pd.DataFrame:
    """온열질환 CSV 또는 XLSX 파일을 읽고 컬럼명을 정리한다."""
    csv_path = find_existing_file(HEAT_CSV_CANDIDATES)
    xlsx_path = find_existing_file(HEAT_XLSX_CANDIDATES)

    df = None
    if csv_path:
        # 공공데이터 CSV는 인코딩이 제각각이라 여러 방식으로 시도한다.
        for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except Exception:
                continue
    elif xlsx_path:
        df = pd.read_excel(xlsx_path)

    if df is None or df.empty:
        return pd.DataFrame()

    # 앞뒤 공백 제거
    df.columns = [str(c).strip() for c in df.columns]

    # 한글 컬럼명을 영어 컬럼명으로 통일
    rename_map = {
        "발생일자": "occur_date",
        "성별": "gender",
        "나이": "age",
        "연령": "age",
        "발생시도": "sido",
        "발생 시도": "sido",
        "발생시군구": "district",
        "발생 시군구": "district",
        "실내외구분": "indoor_outdoor",
        "실내외 구분": "indoor_outdoor",
        "발생장소": "place",
        "발생 장소": "place",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 날짜, 나이 정리
    if "occur_date" in df.columns:
        df["occur_date"] = pd.to_datetime(df["occur_date"], errors="coerce")
        df["year"] = df["occur_date"].dt.year

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        bins = [-1, 19, 39, 64, 79, 200]
        labels = ["0~19세", "20~39세", "40~64세", "65~79세", "80세 이상"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
        df["age_type_65"] = np.where(df["age"] >= 65, "65세 이상", "65세 미만")

    return df


def show_sql(title: str, query: str):
    """SQL을 접힌 영역으로 보여준다."""
    with st.expander(f"사용한 SQL 보기: {title}"):
        st.code(query.strip(), language="sql")


def insight_box(text: str):
    """인사이트 박스"""
    st.info(text)


def format_int(value):
    try:
        return f"{int(value):,}"
    except Exception:
        return "-"


def classify_objective_priority(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """평균이 아니라 분위 기준으로 우선지역 유형과 점수를 재계산한다.

    기준:
    - 독거노인 수 상위 40%: 수요 높음
    - 쉼터 수용률 하위 40%: 수용능력 부족
    - 독거노인 1,000명당 쉼터 수 하위 40%: 접근성 부족
    - 80세 이상 비율 상위 40%: 고령 취약성 높음
    """
    out = df.copy()

    thresholds = {
        "elderly_total_high_60pct": out["elderly_total"].quantile(0.60),
        "capacity_rate_low_40pct": out["capacity_rate"].quantile(0.40),
        "shelters_per_1000_low_40pct": out["shelters_per_1000"].quantile(0.40),
        "elderly_80_plus_rate_high_60pct": out["elderly_80_plus_rate"].quantile(0.60),
    }

    out["demand_high"] = out["elderly_total"] >= thresholds["elderly_total_high_60pct"]
    out["capacity_low"] = out["capacity_rate"] <= thresholds["capacity_rate_low_40pct"]
    out["access_low"] = out["shelters_per_1000"] <= thresholds["shelters_per_1000_low_40pct"]
    out["senior_risk_high"] = out["elderly_80_plus_rate"] >= thresholds["elderly_80_plus_rate_high_60pct"]

    def make_type(row):
        if row["demand_high"] and row["capacity_low"] and row["access_low"]:
            return "개선 우선지역"
        if row["demand_high"] and row["capacity_low"]:
            return "수용능력 부족지역"
        if row["demand_high"] and row["access_low"]:
            return "쉼터 접근성 부족지역"
        if row["senior_risk_high"] and (row["capacity_low"] or row["access_low"]):
            return "고령 취약성 주의지역"
        return "상대적 안정지역"

    out["objective_priority_type"] = out.apply(make_type, axis=1)

    # 점수는 위험 방향이 클수록 높게 산정한다.
    out["objective_priority_score"] = (
        out["demand_high"].astype(int) * 2
        + out["capacity_low"].astype(int) * 3
        + out["access_low"].astype(int) * 3
        + out["senior_risk_high"].astype(int) * 1
        + (out["elderly_per_shelter"] >= out["elderly_per_shelter"].quantile(0.60)).astype(int) * 1
    )

    return out, thresholds


# =========================================================
# 2. 데이터 확인
# =========================================================
db_path = find_existing_file(DB_CANDIDATES)

st.title("🌡️ 서울시 독거노인 폭염 취약지역과 무더위쉼터 공급 분석")
st.caption("자치구별 독거노인 수요와 무더위쉼터 공급의 불균형을 찾아 개선 우선지역을 제안하는 대시보드")

if db_path is None:
    st.error(
        "SQLite DB 파일을 찾을 수 없습니다. 같은 폴더에 `final.db` 또는 `heatwave_sql_final.db`를 넣어주세요."
    )
    st.stop()

# 주요 SQL
SQL_DISTRICT_SUMMARY = """
SELECT
    district,
    elderly_total,
    elderly_65_79,
    elderly_80_plus,
    elderly_80_plus_rate,
    vulnerable_elderly_rate,
    shelter_count,
    total_capacity,
    avg_capacity,
    shelters_per_1000,
    elderly_per_shelter,
    capacity_rate
FROM district_summary
ORDER BY district;
"""

SQL_DISTRICT_PRIORITY = """
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
ORDER BY priority_score DESC;
"""

SQL_SHELTERS = """
SELECT
    district,
    shelter_name,
    facility_type1,
    facility_type2,
    road_address,
    jibun_address,
    capacity,
    area,
    latitude,
    longitude
FROM shelters
WHERE latitude IS NOT NULL
  AND longitude IS NOT NULL;
"""

SQL_ELDERLY_DONG = """
SELECT
    district,
    dong,
    elderly_total,
    elderly_65_79,
    elderly_80_plus,
    elderly_80_plus_ratio
FROM elderly_dong
WHERE dong IS NOT NULL
  AND dong != '소계';
"""

try:
    district_df = run_sql(db_path, SQL_DISTRICT_SUMMARY)
    priority_db_df = run_sql(db_path, SQL_DISTRICT_PRIORITY)
    shelters_df = run_sql(db_path, SQL_SHELTERS)
    elderly_dong_df = run_sql(db_path, SQL_ELDERLY_DONG)
except Exception as e:
    st.error("DB에서 필요한 테이블을 불러오지 못했습니다. `district_summary`, `district_priority`, `shelters`, `elderly_dong` 테이블이 있는지 확인해주세요.")
    st.exception(e)
    st.stop()

# 분위 기준으로 우선순위 재계산
priority_df, thresholds = classify_objective_priority(priority_db_df)
# district_df에도 객관 기준 컬럼 병합
merge_cols = [
    "district",
    "objective_priority_type",
    "objective_priority_score",
    "demand_high",
    "capacity_low",
    "access_low",
    "senior_risk_high",
]
district_df = district_df.merge(priority_df[merge_cols], on="district", how="left")

district_geojson = load_geojson(DISTRICT_GEOJSON_PATH)
heat_df = load_heat_illness_data()

# =========================================================
# 3. 사이드바
# =========================================================
st.sidebar.header("대시보드 설정")
st.sidebar.write(f"사용 DB: `{db_path}`")

all_districts = ["전체"] + sorted(district_df["district"].dropna().unique().tolist())
selected_district = st.sidebar.selectbox("무더위쉼터 위치 지도에서 볼 자치구", all_districts)

metric_choice = st.sidebar.radio(
    "중간 지도에서 볼 지표",
    ["쉼터 수용률", "독거노인 1,000명당 쉼터 수"],
    index=0,
)

# =========================================================
# 4. 프로젝트 소개
# =========================================================
st.header("1. 프로젝트 목표")
st.markdown(
    """
이 대시보드의 목표는 **서울시 자치구별 독거노인 수요와 무더위쉼터 공급의 불균형**을 확인하고,  
폭염 대응 인프라 개선이 필요한 자치구를 찾는 것입니다.

핵심 질문은 다음과 같습니다.

> **독거노인이 많이 거주하는 지역에 무더위쉼터가 충분히 배치되어 있는가?**

단순히 쉼터 수만 비교하지 않고, **수용가능인원, 독거노인 1,000명당 쉼터 수, 쉼터 수용률, 80세 이상 비율**을 함께 고려합니다.
"""
)

# =========================================================
# 5. 온열질환 데이터 도입부
# =========================================================
st.header("2. 온열질환 발생 데이터로 보는 폭염 피해")

if heat_df.empty:
    st.warning(
        "온열질환 파일을 찾지 못했습니다. 같은 폴더에 `heat_illness.csv` 또는 `heat_illness.xlsx`를 넣으면 이 섹션이 자동으로 표시됩니다."
    )
else:
    st.markdown(
        "온열질환 데이터는 핵심 JOIN 데이터가 아니라, **폭염이 실제 건강 피해로 이어지고 고령층이 위험하다는 문제의식**을 보여주는 도입부 자료입니다."
    )

    col1, col2, col3 = st.columns(3)
    total_cases = len(heat_df)
    older_cases = int((heat_df["age"] >= 65).sum()) if "age" in heat_df.columns else 0
    older_rate = older_cases / total_cases * 100 if total_cases else 0
    max_year = int(heat_df["year"].max()) if "year" in heat_df.columns and heat_df["year"].notna().any() else None

    col1.metric("전체 온열질환 발생 건수", f"{total_cases:,}건")
    col2.metric("65세 이상 발생 건수", f"{older_cases:,}건")
    col3.metric("65세 이상 비중", f"{older_rate:.1f}%")

    hcol1, hcol2 = st.columns([1.2, 1])

    with hcol1:
        if "year" in heat_df.columns:
            yearly = heat_df.dropna(subset=["year"]).groupby("year", as_index=False).size()
            yearly = yearly.rename(columns={"size": "illness_count"})
            fig = px.line(
                yearly,
                x="year",
                y="illness_count",
                markers=True,
                title="연도별 온열질환 발생 추이",
                labels={"year": "연도", "illness_count": "발생 건수"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("처리 방식: 발생일자에서 연도를 추출한 뒤 연도별 발생 건수를 집계")

    with hcol2:
        if "age_group" in heat_df.columns:
            age_order = ["0~19세", "20~39세", "40~64세", "65~79세", "80세 이상"]
            age_group = heat_df.dropna(subset=["age_group"]).groupby("age_group", as_index=False).size()
            age_group = age_group.rename(columns={"size": "illness_count"})
            age_group["age_group"] = age_group["age_group"].astype(str)
            age_group["age_group"] = pd.Categorical(age_group["age_group"], age_order, ordered=True)
            age_group = age_group.sort_values("age_group")
            fig = px.bar(
                age_group,
                x="age_group",
                y="illness_count",
                title="연령대별 온열질환 발생 건수",
                labels={"age_group": "연령대", "illness_count": "발생 건수"},
            )
            st.plotly_chart(fig, use_container_width=True)

    if "age_type_65" in heat_df.columns:
        age_type = heat_df.groupby("age_type_65", as_index=False).size().rename(columns={"size": "illness_count"})
        fig = px.pie(
            age_type,
            names="age_type_65",
            values="illness_count",
            hole=0.55,
            title="65세 이상 온열질환 발생 비중",
        )
        st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "온열질환 데이터는 폭염이 단순한 날씨 문제가 아니라 실제 건강 피해로 이어진다는 점을 보여준다. 특히 65세 이상 발생 비중을 통해 고령층을 폭염 취약계층으로 설정할 근거를 마련할 수 있다. 이후 분석은 고령층 중에서도 돌봄 공백 가능성이 큰 독거노인을 중심으로 진행한다."
    )

# =========================================================
# 6. 서울시 전체 핵심 지표
# =========================================================
st.header("3. 서울시 전체 핵심 지표")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("독거노인 수", f"{district_df['elderly_total'].sum():,.0f}명")
m2.metric("무더위쉼터 수", f"{district_df['shelter_count'].sum():,.0f}개")
m3.metric("총 수용가능인원", f"{district_df['total_capacity'].sum():,.0f}명")
m4.metric("평균 1,000명당 쉼터 수", f"{district_df['shelters_per_1000'].mean():.2f}개")
m5.metric("평균 쉼터 수용률", f"{district_df['capacity_rate'].mean():.1f}%")
show_sql("서울시 전체 핵심 지표", SQL_DISTRICT_SUMMARY)

# =========================================================
# 7. 객관 기준 제시와 기준선 산점도
# =========================================================
st.header("4. 개선 유형을 나누는 객관 기준")
st.markdown(
    "평균만으로 유형을 나누면 기준이 다소 주관적으로 보일 수 있어, 이 대시보드에서는 **서울시 25개 자치구의 분위 기준**을 함께 사용합니다."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("수요 높음 기준", f"독거노인 {thresholds['elderly_total_high_60pct']:,.0f}명 이상")
c2.metric("수용능력 부족 기준", f"수용률 {thresholds['capacity_rate_low_40pct']:.1f}% 이하")
c3.metric("접근성 부족 기준", f"1,000명당 쉼터 {thresholds['shelters_per_1000_low_40pct']:.2f}개 이하")
c4.metric("고령 취약성 기준", f"80세 이상 {thresholds['elderly_80_plus_rate_high_60pct']:.1f}% 이상")

st.markdown(
    """
- **개선 우선지역**: 수요 높음 + 수용능력 부족 + 접근성 부족  
- **수용능력 부족지역**: 수요 높음 + 수용능력 부족  
- **쉼터 접근성 부족지역**: 수요 높음 + 접근성 부족  
- **고령 취약성 주의지역**: 고령 취약성 높음 + 공급 지표 부족  
"""
)

fig = px.scatter(
    district_df,
    x="elderly_total",
    y="capacity_rate",
    size="shelter_count",
    color="objective_priority_type",
    hover_name="district",
    hover_data={
        "elderly_total": ":,.0f",
        "capacity_rate": ":.2f",
        "shelter_count": ":,.0f",
        "shelters_per_1000": ":.2f",
        "elderly_80_plus_rate": ":.2f",
    },
    title="유형 분류 기준선: 독거노인 수 × 쉼터 수용률",
    labels={
        "elderly_total": "독거노인 수",
        "capacity_rate": "쉼터 수용률(%)",
        "objective_priority_type": "객관 기준 유형",
        "shelter_count": "쉼터 수",
    },
    size_max=35,
)
fig.add_vline(
    x=thresholds["elderly_total_high_60pct"],
    line_dash="dash",
    annotation_text="수요 높음 기준",
    annotation_position="top right",
)
fig.add_hline(
    y=thresholds["capacity_rate_low_40pct"],
    line_dash="dash",
    annotation_text="수용률 부족 기준",
    annotation_position="bottom right",
)
st.plotly_chart(fig, use_container_width=True)
show_sql("자치구별 독거노인 수와 쉼터 수용률 조회", SQL_DISTRICT_SUMMARY)
insight_box(
    "오른쪽 아래 영역은 독거노인 수가 많지만 쉼터 수용률이 낮은 자치구를 의미한다. 이 영역에 가까운 지역일수록 독거노인 수요 대비 무더위쉼터 수용능력이 부족할 가능성이 크다."
)

# =========================================================
# 8. 중간 진단 지도
# =========================================================
st.header("5. 자치구별 무더위쉼터 공급 진단 지도")

if district_geojson is None:
    st.warning("`data/seoul_district_boundary_simplified.geojson` 파일이 없어 지도 시각화를 건너뜁니다.")
else:
    if metric_choice == "쉼터 수용률":
        map_col = "capacity_rate"
        map_title = "자치구별 쉼터 수용률 지도"
        map_label = "쉼터 수용률(%)"
        map_insight = "쉼터 수용률은 독거노인 수 대비 무더위쉼터 총 수용가능인원을 보여준다. 수용률이 낮은 자치구는 쉼터가 있어도 실제 수용능력이 부족할 수 있다."
    else:
        map_col = "shelters_per_1000"
        map_title = "자치구별 독거노인 1,000명당 쉼터 수 지도"
        map_label = "1,000명당 쉼터 수"
        map_insight = "독거노인 1,000명당 쉼터 수는 수요 규모를 반영한 공급 접근성 지표이다. 값이 낮은 자치구는 독거노인 규모에 비해 쉼터 수가 부족할 가능성이 있다."

    fig = px.choropleth_mapbox(
        district_df,
        geojson=district_geojson,
        locations="district",
        featureidkey="properties.district",
        color=map_col,
        hover_name="district",
        hover_data={
            "elderly_total": ":,.0f",
            "shelter_count": ":,.0f",
            "total_capacity": ":,.0f",
            "capacity_rate": ":.2f",
            "shelters_per_1000": ":.2f",
            "objective_priority_type": True,
        },
        mapbox_style="carto-positron",
        center={"lat": 37.5665, "lon": 126.9780},
        zoom=9.3,
        opacity=0.65,
        title=map_title,
        labels={map_col: map_label},
    )
    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0}, height=620)
    st.plotly_chart(fig, use_container_width=True)
    show_sql("자치구별 지도 지표 조회", SQL_DISTRICT_SUMMARY)
    insight_box(map_insight)

# =========================================================
# 9. 부족 지표 비교 그래프
# =========================================================
st.header("6. 수요 대비 공급 부족 지표 비교")

low_df = district_df.sort_values(["shelters_per_1000", "capacity_rate"], ascending=[True, True]).head(10)
fig = go.Figure()
fig.add_trace(
    go.Bar(
        y=low_df["district"],
        x=low_df["shelters_per_1000"],
        name="독거노인 1,000명당 쉼터 수",
        orientation="h",
    )
)
fig.add_trace(
    go.Scatter(
        y=low_df["district"],
        x=low_df["capacity_rate"],
        name="쉼터 수용률(%)",
        mode="lines+markers",
        xaxis="x2",
    )
)
fig.update_layout(
    title="접근성 부족 하위 10개 자치구와 쉼터 수용률 비교",
    yaxis={"autorange": "reversed"},
    xaxis={"title": "독거노인 1,000명당 쉼터 수"},
    xaxis2={"title": "쉼터 수용률(%)", "overlaying": "x", "side": "top"},
    legend={"orientation": "h", "y": -0.2},
    height=520,
)
st.plotly_chart(fig, use_container_width=True)
show_sql("접근성 부족 하위 자치구 조회", SQL_DISTRICT_SUMMARY)
insight_box(
    "독거노인 1,000명당 쉼터 수와 쉼터 수용률을 함께 보면, 쉼터 개수 부족과 수용능력 부족이 동시에 나타나는 지역을 찾을 수 있다. 두 지표가 모두 낮은 자치구는 개선 우선 검토 대상이 된다."
)

# =========================================================
# 10. 무더위쉼터 위치 지도
# =========================================================
st.header("7. 무더위쉼터 위치 지도")

map_shelters = shelters_df.copy()
if selected_district != "전체":
    map_shelters = map_shelters[map_shelters["district"] == selected_district]

if map_shelters.empty:
    st.warning("선택한 조건에 해당하는 무더위쉼터 위치 데이터가 없습니다.")
else:
    fig = px.scatter_mapbox(
        map_shelters,
        lat="latitude",
        lon="longitude",
        size="capacity",
        size_max=18,
        hover_name="shelter_name",
        hover_data={
            "district": True,
            "capacity": ":,.0f",
            "facility_type1": True,
            "facility_type2": True,
            "road_address": True,
            "latitude": False,
            "longitude": False,
        },
        color="district" if selected_district == "전체" else "facility_type1",
        mapbox_style="carto-positron",
        center={"lat": 37.5665, "lon": 126.9780},
        zoom=10 if selected_district == "전체" else 11.5,
        title="무더위쉼터 위치와 수용가능인원",
    )
    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0}, height=620)
    st.plotly_chart(fig, use_container_width=True)
show_sql("무더위쉼터 위치 데이터 조회", SQL_SHELTERS)
insight_box(
    "무더위쉼터 위치 지도는 개선 우선지역 내부에서 쉼터가 어디에 분포하는지 확인하기 위한 시각화이다. 자치구를 선택하면 해당 지역의 쉼터 위치와 수용가능인원을 함께 볼 수 있다."
)

# =========================================================
# 11. 수요-공급 불균형 산점도
# =========================================================
st.header("8. 수요-공급 불균형 산점도")

fig = px.scatter(
    district_df,
    x="elderly_total",
    y="total_capacity",
    size="shelter_count",
    color="objective_priority_type",
    hover_name="district",
    hover_data={
        "elderly_total": ":,.0f",
        "total_capacity": ":,.0f",
        "shelter_count": ":,.0f",
        "capacity_rate": ":.2f",
        "shelters_per_1000": ":.2f",
        "elderly_80_plus_rate": ":.2f",
    },
    title="독거노인 수요 × 쉼터 수용능력 버블 차트",
    labels={
        "elderly_total": "독거노인 수",
        "total_capacity": "총 수용가능인원",
        "shelter_count": "쉼터 수",
        "objective_priority_type": "객관 기준 유형",
    },
    size_max=42,
)
st.plotly_chart(fig, use_container_width=True)
show_sql("수요-공급 산점도 데이터 조회", SQL_DISTRICT_SUMMARY)
insight_box(
    "독거노인 수는 많은데 총 수용가능인원이 낮은 자치구는 수요 대비 공급이 부족한 지역으로 볼 수 있다. 버블 크기는 쉼터 수를 의미하므로, 쉼터 개수와 실제 수용능력이 함께 충분한지도 확인할 수 있다."
)

# =========================================================
# 12. 개선 우선지역 TOP 5 심층 진단
# =========================================================
st.header("9. 개선 우선지역 TOP 5 심층 진단")

top5 = priority_df.sort_values(
    ["objective_priority_score", "shelters_per_1000", "capacity_rate", "elderly_total"],
    ascending=[False, True, True, False],
).head(5)

st.subheader("TOP 5 표")
st.dataframe(
    top5[
        [
            "district",
            "elderly_total",
            "elderly_80_plus_rate",
            "vulnerable_elderly_rate",
            "shelter_count",
            "total_capacity",
            "shelters_per_1000",
            "capacity_rate",
            "elderly_per_shelter",
            "objective_priority_type",
            "objective_priority_score",
        ]
    ].rename(
        columns={
            "district": "자치구",
            "elderly_total": "독거노인 수",
            "elderly_80_plus_rate": "80세 이상 비율",
            "vulnerable_elderly_rate": "취약 독거노인 비율",
            "shelter_count": "쉼터 수",
            "total_capacity": "총 수용가능인원",
            "shelters_per_1000": "1,000명당 쉼터 수",
            "capacity_rate": "쉼터 수용률",
            "elderly_per_shelter": "쉼터 1개당 독거노인 수",
            "objective_priority_type": "개선 유형",
            "objective_priority_score": "우선순위 점수",
        }
    ),
    use_container_width=True,
)

SQL_TOP5 = """
SELECT
    district,
    elderly_total,
    elderly_80_plus_rate,
    vulnerable_elderly_rate,
    shelter_count,
    total_capacity,
    shelters_per_1000,
    capacity_rate,
    elderly_per_shelter,
    priority_type,
    priority_score
FROM district_priority
ORDER BY
    priority_score DESC,
    shelters_per_1000 ASC,
    capacity_rate ASC,
    elderly_total DESC
LIMIT 5;
"""
show_sql("기존 DB 기준 TOP 5 조회", SQL_TOP5)

# 레이더 차트는 지표 방향을 '위험도'로 맞추기 위해 0~100으로 변환한다.
radar_base = priority_df.copy()
radar_cols = {
    "elderly_total": "독거노인 수",
    "elderly_80_plus_rate": "80세 이상 비율",
    "vulnerable_elderly_rate": "취약 독거노인 비율",
    "elderly_per_shelter": "쉼터 1개당 독거노인",
}
# 공급 지표는 낮을수록 위험하므로 역방향 점수로 변환
radar_base["lack_shelters_per_1000"] = radar_base["shelters_per_1000"].max() - radar_base["shelters_per_1000"]
radar_base["lack_capacity_rate"] = radar_base["capacity_rate"].max() - radar_base["capacity_rate"]
radar_cols["lack_shelters_per_1000"] = "쉼터 접근성 부족"
radar_cols["lack_capacity_rate"] = "수용능력 부족"

def minmax(series):
    if series.max() == series.min():
        return pd.Series([50] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min()) * 100

for col in radar_cols:
    radar_base[col + "_score"] = minmax(radar_base[col])

radar_top5 = radar_base[radar_base["district"].isin(top5["district"])]
radar_labels = list(radar_cols.values())
radar_score_cols = [col + "_score" for col in radar_cols]

fig = go.Figure()
for _, row in radar_top5.iterrows():
    values = [row[c] for c in radar_score_cols]
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=row["district"],
        )
    )
fig.update_layout(
    title="TOP 5 자치구 복합 위험 지표 비교",
    polar={"radialaxis": {"visible": True, "range": [0, 100]}},
    height=620,
)
st.plotly_chart(fig, use_container_width=True)

insight_box(
    "TOP 5는 단순히 쉼터 수가 적은 지역이 아니라, 독거노인 수요·고령 취약성·수용률·접근성 부족을 함께 고려해 도출한 지역이다. 레이더 차트는 지역별로 어떤 지표가 특히 취약한지 비교하는 데 사용된다."
)

# =========================================================
# 13. 개선 우선지역 내부 행정동 심층 분석
# =========================================================
st.header("10. 개선 우선지역 내부 행정동 심층 분석")
st.markdown(
    "자치구 단위로 개선 우선지역을 찾은 뒤, 해당 자치구 내부에서 독거노인이 많은 행정동을 확인합니다. 지도는 자치구 단위 결론이 중심이고, 이 섹션은 보조 심층 분석입니다."
)

selected_top_district = st.selectbox("심층 분석할 TOP 5 자치구 선택", top5["district"].tolist())
dong_focus = elderly_dong_df[elderly_dong_df["district"] == selected_top_district].sort_values("elderly_total", ascending=False)

if dong_focus.empty:
    st.warning("선택한 자치구의 행정동 데이터가 없습니다.")
else:
    dcol1, dcol2 = st.columns([1, 1.2])
    with dcol1:
        st.dataframe(
            dong_focus.head(10).rename(
                columns={
                    "district": "자치구",
                    "dong": "행정동",
                    "elderly_total": "독거노인 수",
                    "elderly_65_79": "65~79세",
                    "elderly_80_plus": "80세 이상",
                    "elderly_80_plus_ratio": "80세 이상 비율",
                }
            ),
            use_container_width=True,
        )
    with dcol2:
        fig = px.bar(
            dong_focus.head(10).sort_values("elderly_total"),
            x="elderly_total",
            y="dong",
            orientation="h",
            title=f"{selected_top_district} 행정동별 독거노인 TOP 10",
            labels={"elderly_total": "독거노인 수", "dong": "행정동"},
        )
        st.plotly_chart(fig, use_container_width=True)

    show_sql("행정동별 독거노인 데이터 조회", SQL_ELDERLY_DONG)
    insight_box(
        f"{selected_top_district}가 개선 우선지역으로 도출되었다면, 그 안에서도 독거노인이 집중된 행정동을 우선적으로 확인할 필요가 있다. 이 결과는 쉼터 안내 강화, 이동 지원, 추가 후보지 검토의 세부 근거로 활용할 수 있다."
    )

# =========================================================
# 14. 유형별 정책 제안
# =========================================================
st.header("11. 유형별 정책 제안")

policy_df = pd.DataFrame(
    {
        "유형": [
            "개선 우선지역",
            "수용능력 부족지역",
            "쉼터 접근성 부족지역",
            "고령 취약성 주의지역",
            "상대적 안정지역",
        ],
        "판단 기준": [
            "독거노인 수요가 높고, 수용률과 접근성 지표가 모두 낮음",
            "독거노인 수요가 높지만 총 수용가능인원이 부족함",
            "독거노인 수요가 높지만 1,000명당 쉼터 수가 낮음",
            "80세 이상 비율이 높고 공급 지표 중 일부가 부족함",
            "상대적으로 수요 대비 공급이 안정적임",
        ],
        "정책 제안": [
            "쉼터 추가 배치, 수용인원 확대, 취약가구 안내 강화 우선 추진",
            "기존 쉼터의 수용가능인원 확대, 임시 쉼터 추가 확보",
            "쉼터 접근성이 낮은 생활권에 신규 쉼터 후보지 검토",
            "80세 이상 독거노인 대상 방문 안내, 폭염 알림, 이동 지원 강화",
            "현 수준 유지와 모니터링 중심 관리",
        ],
    }
)
st.dataframe(policy_df, use_container_width=True)

# =========================================================
# 15. 최종 결론 지도
# =========================================================
st.header("12. 최종 결론: 서울시 자치구별 개선 우선순위 지도")

if district_geojson is None:
    st.warning("자치구 GeoJSON 파일이 없어 최종 결론 지도를 표시하지 못했습니다.")
else:
    final_map_df = district_df.copy()
    fig = px.choropleth_mapbox(
        final_map_df,
        geojson=district_geojson,
        locations="district",
        featureidkey="properties.district",
        color="objective_priority_score",
        hover_name="district",
        hover_data={
            "objective_priority_type": True,
            "elderly_total": ":,.0f",
            "shelter_count": ":,.0f",
            "total_capacity": ":,.0f",
            "shelters_per_1000": ":.2f",
            "capacity_rate": ":.2f",
            "elderly_80_plus_rate": ":.2f",
        },
        mapbox_style="carto-positron",
        center={"lat": 37.5665, "lon": 126.9780},
        zoom=9.3,
        opacity=0.72,
        title="객관 기준 기반 개선 우선순위 지도",
        labels={"objective_priority_score": "우선순위 점수"},
    )
    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0}, height=640)
    st.plotly_chart(fig, use_container_width=True)

    show_sql("최종 결론 지도 데이터 조회", SQL_DISTRICT_SUMMARY)
    insight_box(
        "최종 지도는 독거노인 수요, 쉼터 수용률, 독거노인 1,000명당 쉼터 수, 80세 이상 비율을 종합한 결과이다. 색이 진한 자치구일수록 폭염 대응 인프라 개선을 우선 검토할 필요가 있다."
    )

st.success("분석 완료: 이 대시보드는 온열질환 피해 근거 → 독거노인 수요와 쉼터 공급 분석 → 개선 우선지역 도출 → 정책 제안 흐름으로 구성되어 있습니다.")
