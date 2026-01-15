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
        st.rerun()

# --- 지도 클릭 섹션 ---
st.markdown("### 🖱️ 지도를 클릭하여 위치를 미세 조정하세요")

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
        st.session_state.start_coords = (clicked_lat, clicked_lng); st.rerun()
    if c2.button("📌 여기를 [목적지]로"):
        st.session_state.end_coords = (clicked_lat, clicked_lng); st.rerun()

start_coords = st.session_state.start_coords
end_coords = st.session_state.end_coords

# [3] 경로 탐색 및 시각화 (유턴 현상 방지 로직 적용)
if start_coords and end_coords:
    G = graph.copy()
    
    # 1. 가장 가까운 도로(Edge)들을 찾습니다.
    # 결과값은 (u, v, key) 튜플 형태입니다.
    ne_s = ox.distance.nearest_edges(G, start_coords[1], start_coords[0])
    ne_e = ox.distance.nearest_edges(G, end_coords[1], end_coords[0])
    
    # 2. [핵심 수정] 도로의 양 끝점(u, v) 중 사용자와 더 가까운 진짜 교차로를 찾습니다.
    def get_truest_nearest_node(graph, coords, edge):
        u, v, _ = edge
        # 두 노드의 좌표 가져오기
        node_u_coords = (graph.nodes[u]['y'], graph.nodes[u]['x'])
        node_v_coords = (graph.nodes[v]['y'], graph.nodes[v]['x'])
        
        # 사용자와 각 노드 사이의 직선 거리 계산 (간단한 피타고라스 방식)
        dist_u = (coords[0]-node_u_coords[0])**2 + (coords[1]-node_u_coords[1])**2
        dist_v = (coords[0]-node_v_coords[0])**2 + (coords[1]-node_v_coords[1])**2
        
        return u if dist_u < dist_v else v

    # 실제 출발/도착 노드 결정
    orig_node = get_truest_nearest_node(G, start_coords, ne_s)
    dest_node = get_truest_nearest_node(G, end_coords, ne_e)

    # --- 이하 장애물 우회 및 경로 탐색 로직 (기존과 동일) ---
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

    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight='my_weight')
        
        # 거리 계산
        total_meters = 0
        for u, v in zip(route[:-1], route[1:]):
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                min_len = min(d.get('length', 0) for d in edge_data.values())
                total_meters += min_len
        total_meters = int(total_meters)

        # 시각화 및 결과 출력 (기존 유지)
        fig, ax = ox.plot_graph_route(G, route, route_color='#3b82f6', route_linewidth=5, 
                                    node_size=0, bgcolor='white', show=False, close=False)
        
        # 실제 위치 연결선
        start_node_pt = (G.nodes[route[0]]['x'], G.nodes[route[0]]['y'])
        ax.plot([start_coords[1], start_node_pt[0]], [start_coords[0], start_node_pt[1]], 
                color='#3b82f6', linewidth=5, alpha=0.7, zorder=4)
        end_node_pt = (G.nodes[route[-1]]['x'], G.nodes[route[-1]]['y'])
        ax.plot([end_coords[1], end_node_pt[0]], [end_coords[0], end_node_pt[1]], 
                color='#3b82f6', linewidth=5, alpha=0.7, zorder=4)

        # 마커 및 줌 설정 생략...
        st.pyplot(fig)
        st.metric(label="🏁 예상 총 보행 거리", value=f"{total_meters} m")
        
    except Exception as e:
        st.error(f"경로를 찾을 수 없습니다: {e}")

