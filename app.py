import streamlit as st
import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from geopy.geocoders import Nominatim

# 1. 페이지 설정
st.set_page_config(page_title="감계 배리어프리 내비", layout="wide")
st.title("🗺️ 감계지구 스마트 우회 내비게이션")

# 2. 데이터 및 지도 로드 (캐싱)
sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ9_vnph9VqvmqqmA-_njbzjKR9dKTIOhFESErGsSSGaiQ9617tOmurA4Y8C9c-wu1t2LKQXtSPtEVk/pub?output=csv"

@st.cache_data(ttl=300)
def get_obstacle_data(url):
    try: return pd.read_csv(url)
    except: return pd.DataFrame()

@st.cache_resource
def get_graph_data():
    center_point = (35.300, 128.595)
    # 넓은 지역의 데이터를 미리 확보 (2.5km)
    return ox.graph_from_point(center_point, dist=2500, network_type='walk')

df = get_obstacle_data(sheet_url)
graph = get_graph_data()
geolocator = Nominatim(user_agent="my_bfree_nav_v4")

# 3. 사이드바 설정
st.sidebar.header("📍 경로 설정")
input_method = st.sidebar.radio("방식 선택", ["장소 이름 검색", "위도/경도 직접 입력"])

start_coords, end_coords = None, None

if input_method == "장소 이름 검색":
    start_input = st.sidebar.text_input("출발지", value="감계중학교")
    end_input = st.sidebar.text_input("목적지", value="창원북면고등학교")
    
    if st.sidebar.button("🚀 경로 탐색"):
        try:
            s_loc = geolocator.geocode(f"{start_input.strip()}, 창원시")
            e_loc = geolocator.geocode(f"{end_input.strip()}, 창원시")
            if s_loc and e_loc:
                start_coords, end_coords = (s_loc.latitude, s_loc.longitude), (e_loc.latitude, e_loc.longitude)
            else: st.sidebar.error("장소를 찾을 수 없습니다.")
        except: st.sidebar.error("검색 중 오류가 발생했습니다.")

else:
    s_lat = st.sidebar.number_input("출발 위도", value=35.299396, format="%.6f")
    s_lon = st.sidebar.number_input("출발 경도", value=128.595954, format="%.6f")
    e_lat = st.sidebar.number_input("목적 위도", value=35.302278, format="%.6f")
    e_lon = st.sidebar.number_input("목적 경도", value=128.593880, format="%.6f")
    if st.sidebar.button("🚀 좌표로 탐색"):
        start_coords, end_coords = (s_lat, s_lon), (e_lat, e_lon)

# 4. 경로 계산 및 "확대" 시각화
if start_coords and end_coords:
    # 가중치 계산 (기존 동일)
    for u, v, k, data in graph.edges(keys=True, data=True):
        data['barrier_free_weight'] = data['length']
        if not df.empty:
            edge_geom = data.get('geometry', LineString([(graph.nodes[u]['x'], graph.nodes[u]['y']), (graph.nodes[v]['x'], graph.nodes[v]['y'])]))
            for _, row in df.iterrows():
                if edge_geom.distance(Point(row['경도'], row['위도'])) < 0.00025:
                    data['barrier_free_weight'] *= 15
                    break

    orig_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
    dest_node = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])

    try:
        route = nx.shortest_path(graph, orig_node, dest_node, weight='barrier_free_weight')
        
        # --- [중요] 자동 줌 로직 추가 ---
        # 경로 상의 모든 좌표를 모아 최소/최대 위도 경도를 찾습니다.
        route_nodes = [graph.nodes[node] for node in route]
        lats = [node['y'] for node in route_nodes]
        lons = [node['x'] for node in route_nodes]
        
        # 여백(padding) 설정
        padding = 0.002 
        bbox = (max(lats) + padding, min(lats) - padding, max(lons) + padding, min(lons) - padding)

        fig, ax = ox.plot_graph_route(
            graph, route, route_color='#3b82f6', node_size=0, 
            edge_color='#e2e8f0', bgcolor='white', show=False, close=False
        )
        
        # 지도의 범위를 경로 주변으로 고정 (확대 효과)
        ax.set_ylim(bbox[1], bbox[0])
        ax.set_xlim(bbox[3], bbox[2])

        if not df.empty:
            ax.scatter(df['경도'], df['위도'], c='#ef4444', s=50, label='Obstacle', zorder=5)
        ax.scatter(start_coords[1], start_coords[0], c='#10b981', s=100, marker='s', label='Start', zorder=6)
        ax.scatter(end_coords[1], end_coords[0], c='#3b82f6', s=100, marker='X', label='Goal', zorder=6)
        
        st.pyplot(fig)
        st.success("🏁 경로 탐색 결과 (경로에 맞춰 지도를 확대했습니다)")
    except Exception as e:
        st.error(f"경로 탐색 실패: {e}")
