import streamlit as st
import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from geopy.geocoders import Nominatim

st.set_page_config(page_title="감계 배리어프리 내비", layout="wide")
st.title("🗺️ 감계지구 스마트 우회 내비게이션")

# [1] 데이터 로드
sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ9_vnph9VqvmqqmA-_njbzjKR9dKTIOhFESErGsSSGaiQ9617tOmurA4Y8C9c-wu1t2LKQXtSPtEVk/pub?output=csv"

@st.cache_data(ttl=60) # 데이터 업데이트 확인을 위해 캐시 시간 단축
def get_obstacle_data(url):
    try: return pd.read_csv(url)
    except: return pd.DataFrame()

@st.cache_resource
def get_graph_data():
    center_point = (35.300, 128.595)
    # 걷기 가능한 모든 도로망 로드
    return ox.graph_from_point(center_point, dist=2000, network_type='walk')

df = get_obstacle_data(sheet_url)
graph = get_graph_data()
geolocator = Nominatim(user_agent="my_bfree_nav_v5")

# [2] 사이드바 설정
st.sidebar.header("📍 경로 설정")
input_method = st.sidebar.radio("방식 선택", ["장소 이름 검색", "위도/경도 직접 입력"])

start_coords, end_coords = None, None

if input_method == "장소 이름 검색":
    start_input = st.sidebar.text_input("출발지", value="감계중학교")
    end_input = st.sidebar.text_input("목적지", value="북면사무소")
    if st.sidebar.button("🚀 경로 탐색"):
        try:
            s_loc = geolocator.geocode(f"{start_input.strip()}, 창원시")
            e_loc = geolocator.geocode(f"{end_input.strip()}, 창원시")
            if s_loc and e_loc:
                start_coords, end_coords = (s_loc.latitude, s_loc.longitude), (e_loc.latitude, e_loc.longitude)
        except: st.sidebar.error("검색 중 오류 발생")
else:
    s_lat = st.sidebar.number_input("출발 위도", value=35.299396, format="%.6f")
    s_lon = st.sidebar.number_input("출발 경도", value=128.595954, format="%.6f")
    e_lat = st.sidebar.number_input("목적 위도", value=35.302278, format="%.6f")
    e_lon = st.sidebar.number_input("목적 경도", value=128.593880, format="%.6f")
    if st.sidebar.button("🚀 좌표 탐색"):
        start_coords, end_coords = (s_lat, s_lon), (e_lat, e_lon)

# [3] 경로 탐색 및 시각화 로직 개선
if start_coords and end_coords:
    G = graph.copy()
    
    # --- 파라미터 미세 조정 ---
    DETECTION_RADIUS = 0.00010  # 약 15~20m (너무 넓으면 옆길까지 막힘)
    PENALTY_FACTOR = 5         # 5배 정도의 페널티 (적절한 우회 유도)
    # -----------------------

    for u, v, k, data in G.edges(keys=True, data=True):
        # 기본 거리에 '보행 편의성' 가중치 부여
        length = data.get('length', 1)
        data['my_weight'] = length
        
        if 'geometry' in data:
            edge_geom = data['geometry']
        else:
            edge_geom = LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
        
        if not df.empty:
            for _, row in df.iterrows():
                obs_point = Point(row['경도'], row['위도'])
                # 장애물이 도로와 아주 가까울 때만 페널티 부여
                if edge_geom.distance(obs_point) < DETECTION_RADIUS:
                    # 거리에 비례한 페널티 부여 (무조건 막는 게 아니라 비용을 높임)
                    data['my_weight'] = length * PENALTY_FACTOR
                    break

    orig_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
    dest_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])

    try:
        # 가중치 기반 최단 경로 계산
        route = nx.shortest_path(G, orig_node, dest_node, weight='my_weight')
        
        # 실제 이동 거리 계산 (페널티 제외 순수 미터법 거리)
        actual_distance = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length'))
        
        # 시각화 및 결과 출력
        route_nodes = [G.nodes[node] for node in route]
        lats, lons = [n['y'] for n in route_nodes], [n['x'] for n in route_nodes]
        bbox = (max(lats)+0.001, min(lats)-0.001, max(lons)+0.001, min(lons)-0.001)

        fig, ax = ox.plot_graph_route(G, route, route_color='#3b82f6', route_linewidth=5,
                                    node_size=0, bgcolor='white', show=False, close=False)
        ax.set_ylim(bbox[1], bbox[0]); ax.set_xlim(bbox[3], bbox[2])

        if not df.empty:
            ax.scatter(df['경도'], df['위도'], c='#ef4444', s=60, label='Obstacle', zorder=5, edgecolors='white')
        ax.scatter(start_coords[1], start_coords[0], c='#10b981', s=150, marker='s', label='Start', zorder=6)
        ax.scatter(end_coords[1], end_coords[0], c='#3b82f6', s=150, marker='X', label='Goal', zorder=6)
        
        st.pyplot(fig)
        st.success(f"✅ 우회 경로 탐색 완료! (예상 보행 거리: {actual_distance:.0f}m)")
    except Exception as e:
        st.error(f"경로를 계산할 수 없습니다: {e}")
