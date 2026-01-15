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
st.title("🗺️ 감계지구 스마트 우회 내비게이션")

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
    if st.sidebar.button("🔍 장소 검색"):
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
    if st.sidebar.button("📍 좌표 반영"):
        st.session_state.start_coords = (s_lat, s_lon)
        st.session_state.end_coords = (e_lat, e_lon)
        st.session_state.run_nav = False
        st.rerun()

# --- 1단계: 지도 클릭 위치 설정 ---
st.markdown("### 1️⃣ 지도를 클릭하여 위치를 설정하세요")
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

# --- 2단계: 실행 버튼 ---
st.markdown("---")
st.markdown("### 2️⃣ 경로 탐색을 시작합니다")
if st.button("🚀 AI 우회 경로 찾기", use_container_width=True, type="primary"):
    st.session_state.run_nav = True

# [3] 경로 탐색 및 시각화
if st.session_state.run_nav:
    G = graph.copy()
    
    # 교차로 탐색 보정 (유턴 방지)
    def get_truest_node(graph, coords):
        edge = ox.distance.nearest_edges(graph, coords[1], coords[0])
        u, v, _ = edge
        dist_u = (coords[0]-graph.nodes[u]['y'])**2 + (coords[1]-graph.nodes[u]['x'])**2
        dist_v = (coords[0]-graph.nodes[v]['y'])**2 + (coords[1]-graph.nodes[v]['x'])**2
        return u if dist_u < dist_v else v

    orig_node = get_truest_node(G, st.session_state.start_coords)
    dest_node = get_truest_node(G, st.session_state.end_coords)

    # 장애물 가중치 적용
    DETECTION_RADIUS = 0.0001
    PENALTY = 50
    for u, v, k, data in G.edges(keys=True, data=True):
        data['my_weight'] = data['length']
        if 'geometry' in data: edge_geom = data['geometry']
        else: edge_geom = LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
        if not df.empty:
            for _, row in df.iterrows():
                if edge_geom.distance(Point(row['경도'], row['위도'])) < DETECTION_RADIUS:
                    data['my_weight'] *= PENALTY
                    break

    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight='my_weight')
        
        # --- [에러 해결 부분] 직접 루프를 돌아 거리 합산 ---
        total_meters = 0
        for u, v in zip(route[:-1], route[1:]):
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                # MultiGraph 대응: 가장 짧은 length 선택
                min_len = min(d.get('length', 0) for d in edge_data.values())
                total_meters += min_len
        total_meters = int(total_meters)
        # ----------------------------------------------

        fig, ax = plt.subplots(figsize=(10, 8))
        ox.plot_graph_route(G, route, route_color='#3b82f6', route_linewidth=6, node_size=0, bgcolor='white', show=False, close=False, ax=ax)
        
        # 실제 좌표 연결선
        ax.plot([st.session_state.start_coords[1], G.nodes[route[0]]['x']], [st.session_state.start_coords[0], G.nodes[route[0]]['y']], color='#3b82f6', linewidth=6, alpha=0.7)
        ax.plot([st.session_state.end_coords[1], G.nodes[route[-1]]['x']], [st.session_state.end_coords[0], G.nodes[route[-1]]['y']], color='#3b82f6', linewidth=6, alpha=0.7)

        # 줌 최적화
        lats = [G.nodes[n]['y'] for n in route] + [st.session_state.start_coords[0], st.session_state.end_coords[0]]
        lons = [G.nodes[n]['x'] for n in route] + [st.session_state.start_coords[1], st.session_state.end_coords[1]]
        pad = 0.0003
        ax.set_ylim(min(lats)-pad, max(lats)+pad); ax.set_xlim(min(lons)-pad, max(lons)+pad)
        
        if not df.empty: ax.scatter(df['경도'], df['위도'], c='#ef4444', s=80, edgecolors='white', zorder=4)
        ax.scatter(st.session_state.start_coords[1], st.session_state.start_coords[0], c='#10b981', s=200, marker='s', edgecolors='white', zorder=5)
        ax.scatter(st.session_state.end_coords[1], st.session_state.end_coords[0], c='#3b82f6', s=250, marker='X', edgecolors='white', zorder=5)

        ax.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig)
        
        st.metric("🏁 예상 보행 거리", f"{total_meters} m")
        st.success(f"최적 우회 경로를 탐색했습니다. (도보 약 {round(total_meters/67)}분)")

    except Exception as e:
        st.error(f"경로 탐색 실패: {e}")
