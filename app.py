import streamlit as st
import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title="감계 배리어프리 내비", layout="wide")
st.title("🗺️ 감계지구 배리어프리 내비게이션")

# [1] 데이터 로드
sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ9_vnph9VqvmqqmA-_njbzjKR9dKTIOhFESErGsSSGaiQ9617tOmurA4Y8C9c-wu1t2LKQXtSPtEVk/pub?output=csv"

@st.cache_data(ttl=10)
def get_obstacle_data(url):
    try: return pd.read_csv(url)
    except: return pd.DataFrame()

@st.cache_resource
def get_graph_data():
    center_point = (35.300, 128.595)
    return ox.graph_from_point(center_point, dist=2000, network_type='walk')

df = get_obstacle_data(sheet_url)
graph = get_graph_data()
geolocator = Nominatim(user_agent="my_bfree_nav_v6")

# --- 세션 상태 초기화 ---
if 'start_coords' not in st.session_state:
    st.session_state.start_coords = (35.299396, 128.595954)
if 'end_coords' not in st.session_state:
    st.session_state.end_coords = (35.302278, 128.593880)
if 'run_nav' not in st.session_state:
    st.session_state.run_nav = False

# [2] 사이드바 설정
st.sidebar.header("📍 경로 설정")
input_method = st.sidebar.radio("방식 선택", ["장소 이름 검색", "위도/경도 직접 입력"])

if input_method == "장소 이름 검색":
    start_input = st.sidebar.text_input("출발지", value="감계중학교")
    end_input = st.sidebar.text_input("목적지", value="북면사무소")
    if st.sidebar.button("🚀 장소 검색"):
        try:
            s_loc = geolocator.geocode(f"{start_input.strip()}, 창원시")
            e_loc = geolocator.geocode(f"{end_input.strip()}, 창원시")
            if s_loc and e_loc:
                st.session_state.start_coords = (s_loc.latitude, s_loc.longitude)
                st.session_state.end_coords = (e_loc.latitude, e_loc.longitude)
                st.session_state.run_nav = False
                st.rerun() 
        except: st.sidebar.error("검색 중 오류 발생")
else:
    s_lat = st.sidebar.number_input("출발 위도", value=st.session_state.start_coords[0], format="%.6f")
    s_lon = st.sidebar.number_input("출발 경도", value=st.session_state.start_coords[1], format="%.6f")
    e_lat = st.sidebar.number_input("목적 위도", value=st.session_state.end_coords[0], format="%.6f")
    e_lon = st.sidebar.number_input("목적 경도", value=st.session_state.end_coords[1], format="%.6f")
    if st.sidebar.button("🚀 좌표 반영"):
        st.session_state.start_coords = (s_lat, s_lon)
        st.session_state.end_coords = (e_lat, e_lon)
        st.session_state.run_nav = False
        st.rerun()

# --- 지도 클릭 섹션 ---
st.markdown("### 🖱️ 지도를 더블클릭하여 출발지와 목적지를 설정하세요")
m = folium.Map(location=[st.session_state.start_coords[0], st.session_state.start_coords[1]], zoom_start=15)
folium.Marker(st.session_state.start_coords, tooltip="출발지", icon=folium.Icon(color='green')).add_to(m)
folium.Marker(st.session_state.end_coords, tooltip="목적지", icon=folium.Icon(color='blue')).add_to(m)

map_data = st_folium(m, key="main_map", width=900, height=450, returned_objects=["last_clicked"])

if map_data and map_data.get('last_clicked'):
    clicked_lat = map_data['last_clicked']['lat']
    clicked_lng = map_data['last_clicked']['lng']
    st.info(f"선택된 좌표: {clicked_lat:.6f}, {clicked_lng:.6f}")
    c1, c2 = st.columns(2)
    if c1.button("📌 여기를 [출발지]로"):
        st.session_state.start_coords = (clicked_lat, clicked_lng)
        st.session_state.run_nav = False
        st.rerun()
    if c2.button("📌 여기를 [목적지]로"):
        st.session_state.end_coords = (clicked_lat, clicked_lng)
        st.session_state.run_nav = False
        st.rerun()

start_coords = st.session_state.start_coords
end_coords = st.session_state.end_coords

st.markdown("---")
if st.button("🏁 AI 우회 경로 탐색 시작", use_container_width=True, type="primary"):
    st.session_state.run_nav = True

# [3] 경로 탐색 및 시각화
if st.session_state.run_nav and start_coords and end_coords:
    G = graph.copy()
    
    try:
        # [수정 1] nearest_edges 대신 nearest_nodes 사용 (직접 노드 스냅)
        # 이렇게 하면 '가장 가까운 선'을 찾아서 노드를 고르는 과정에서 발생하는 역주행이 사라집니다.
        orig_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
        dest_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])

        # 장애물 우회 가중치
        DETECTION_RADIUS = 0.0001  
        PENALTY = 50               
        for u, v, k, data in G.edges(keys=True, data=True):
            data['my_weight'] = data['length']
            if 'geometry' in data: edge_geom = data['geometry']
            else: edge_geom = LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
            if not df.empty:
                for _, row in df.iterrows():
                    obs_point = Point(row['경도'], row['위도'])
                    if edge_geom.distance(obs_point) < DETECTION_RADIUS:
                        data['my_weight'] = data['length'] * PENALTY
                        break

        route = nx.shortest_path(G, orig_node, dest_node, weight='my_weight')
        
        # 거리 계산 (단순화: 경로상의 거리 합산)
        total_meters = 0
        for u, v in zip(route[:-1], route[1:]):
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                min_len = min(d.get('length', 0) for d in edge_data.values())
                total_meters += min_len
        
        # [수정 2] 노드까지의 직선 거리 합산 제거 (보조선을 안 그리므로 거리 계산에서도 제외)
        total_meters = int(total_meters)

        # 시각화
        fig, ax = plt.subplots(figsize=(10, 10))
        ox.plot_graph(G, ax=ax, node_size=0, edge_color='#94a3b8', edge_linewidth=1.2, bgcolor='white', show=False, close=False)
        ox.plot_graph_route(G, route, ax=ax, route_color='#1d4ed8', route_linewidth=6, node_size=0, show=False, close=False)

        # [수정 3] 기존에 있던 ax.plot(보조선 그리기) 코드 2줄 삭제함

        if not df.empty:
            ax.scatter(df['경도'], df['위도'], c='#ef4444', s=80, zorder=10, edgecolors='white')

        ax.scatter(start_coords[1], start_coords[0], c='#10b981', s=150, marker='s', zorder=11, edgecolors='white')
        ax.scatter(end_coords[1], end_coords[0], c='#3b82f6', s=150, marker='X', zorder=11, edgecolors='white')
        
        lats = [G.nodes[node]['y'] for node in route] + [start_coords[0], end_coords[0]]
        lons = [G.nodes[node]['x'] for node in route] + [start_coords[1], end_coords[1]]
        pad = 0.0003
        ax.set_ylim(min(lats)-pad, max(lats)+pad)
        ax.set_xlim(min(lons)-pad, max(lons)+pad)
        ax.axis('off')
        st.pyplot(fig)
        
        st.metric(label="🏁 예상 총 보행 거리", value=f"{total_meters} m")
        st.success(f"최적 경로를 찾았습니다. (도보 약 {max(1, round(total_meters/67))}분 소요)")
        
    except Exception as e:
        st.error(f"경로를 찾을 수 없습니다: {e}")
