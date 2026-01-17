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
    try:
        data = pd.read_csv(url)
        return data.dropna(subset=['위도', '경도']) # 결측치 제거로 에러 방지
    except:
        return pd.DataFrame()

@st.cache_resource
def get_graph_data():
    center_point = (35.300, 128.595)
    # dist를 1500 정도로 조절하면 연산 속도가 빨라집니다.
    return ox.graph_from_point(center_point, dist=1500, network_type='walk')

df = get_obstacle_data(sheet_url)
graph = get_graph_data()
geolocator = Nominatim(user_agent="my_bfree_nav_v6")

# 세션 상태 유지
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
            else:
                st.sidebar.warning("장소를 찾을 수 없습니다.")
        except:
            st.sidebar.error("검색 중 오류 발생")
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

# [지도 클릭 처리]
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

st.markdown("---")
if st.button("🏁 AI 우회 경로 탐색 시작", use_container_width=True, type="primary"):
    st.session_state.run_nav = True

# [3] 경로 탐색 로직
if st.session_state.run_nav:
    G = graph.copy()
    try:
        start_coords = st.session_state.start_coords
        end_coords = st.session_state.end_coords

        # 1. 근접 도로 찾기
        start_edge = ox.distance.nearest_edges(G, start_coords[1], start_coords[0])
        end_edge = ox.distance.nearest_edges(G, end_coords[1], end_coords[0])

        def get_dist(n_id, target_coords):
            node_data = G.nodes[n_id]
            # 위도, 경도 순서 주의
            return ox.distance.great_circle(node_data['y'], node_data['x'], target_coords[0], target_coords[1])

        orig_node = start_edge[0] if get_dist(start_edge[0], end_coords) < get_dist(start_edge[1], end_coords) else start_edge[1]
        dest_node = end_edge[0] if get_dist(end_edge[0], start_coords) < get_dist(end_edge[1], start_coords) else end_edge[1]

        # 2. 가중치 페널티 적용
        DETECTION_RADIUS = 0.00025 # 약 25m로 확장하여 인식률 상향
        PENALTY = 1000             # 확실한 우회를 위해 가중치 대폭 상향
        
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

        # 3. 경로 탐색 및 거리 계산
        route = nx.shortest_path(G, orig_node, dest_node, weight='my_weight')
        
        # 엣지 속성에서 실제 길이 합산 (get_edge_data가 멀티그래프이므로 인덱스 0 사용)
        total_meters = sum(G.get_edge_data(u, v)[0]['length'] for u, v in zip(route[:-1], route[1:]))
        total_meters = int(total_meters + get_dist(orig_node, start_coords) + get_dist(dest_node, end_coords))

        # 4. 시각화
        fig, ax = plt.subplots(figsize=(10, 10))
        ox.plot_graph(G, ax=ax, node_size=0, edge_color='#cbd5e1', edge_linewidth=1, bgcolor='white', show=False, close=False)
        ox.plot_graph_route(G, route, ax=ax, route_color='#1d4ed8', route_linewidth=5, node_size=0, show=False, close=False)

        # 실제 좌표 연결 보조선
        ax.plot([start_coords[1], G.nodes[orig_node]['x']], [start_coords[0], G.nodes[orig_node]['y']], color='#1d4ed8', linewidth=5, solid_capstyle='round')
        ax.plot([end_coords[1], G.nodes[dest_node]['x']], [end_coords[0], G.nodes[dest_node]['y']], color='#1d4ed8', linewidth=5, solid_capstyle='round')

        if not df.empty:
            ax.scatter(df['경도'], df['위도'], c='#ef4444', s=100, zorder=10, edgecolors='white', marker='o', label='Obstacle')

        ax.scatter(start_coords[1], start_coords[0], c='#10b981', s=200, marker='s', zorder=11, edgecolors='white')
        ax.scatter(end_coords[1], end_coords[0], c='#3b82f6', s=200, marker='X', zorder=11, edgecolors='white')
        
        ax.axis('off')
        st.pyplot(fig)
        
        st.metric(label="🏁 예상 총 보행 거리", value=f"{total_meters} m")
        st.success(f"최적 경로 탐색 완료 (약 {max(1, round(total_meters/67))}분 소요)")
        
    except Exception as e:
        st.error(f"경로 탐색 오류: {e}")
