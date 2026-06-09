서울시 행정동/자치구 경계 단순화 GeoJSON

원본: BND_ADM_DONG_PG SHP 파일 세트
처리:
1. ADM_CD가 11로 시작하는 서울시 행정동만 필터링
2. 좌표계를 EPSG:5186에서 EPSG:4326으로 변환
3. geometry를 simplify(0.0001, preserve_topology=True)로 단순화
4. 자치구명(district)과 자치구코드(gu_code) 컬럼 추가
5. 자치구 경계 파일은 행정동 경계를 district 기준으로 dissolve하여 생성

파일:
- seoul_adm_dong_simplified.geojson
  컬럼: ADM_CD, ADM_NM, gu_code, district, geometry
  용도: 행정동별 독거노인 분포 지도

- seoul_district_boundary_simplified.geojson
  컬럼: gu_code, district, geometry
  용도: 자치구별 취약도/우선지역 지도

Streamlit에서는 geopandas.read_file() 또는 folium.GeoJson()으로 사용할 수 있습니다.
