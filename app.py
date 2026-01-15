import streamlit as st
import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from geopy.geocoders import Nominatim

# 1. 페이지 기본 설정
st.set_page_config(page_title="감계 배리어프리 내비", layout="wide")
st.title("🗺️ 감계지구 스마트 우회 내비게이션")

# 2. 데이터 및 지도 로드 (캐싱)
sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ9_vnph9VqvmqqmA-_njbzjKR9dKTIOhFESErGsSSGai (이전과 동일한 URL)"
# 실제 사용시에는 위 sheet_url에 질문자님의 구글 시트 주소를 넣으세요.
sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ9_vnph9VqvmqqmA-_njbzjKR9dKTIOhFESErGsSSGaiQ9617tOmurA4Y8C9c-wu1t2LKQXtSPtEVk/pub?output=csv"

@st.cache_data(ttl=300)
def get_obstacle_data(url):
    try: return pd.read_csv(url)
    except: return pd.DataFrame()

@st.cache_resource
def get_graph_data():
    center_point = (35.300, 128.595)
    # 범위를 2.5km로 확장하여 북면고 등 외곽 지역 포함
    return ox.graph_from_point(center_point, dist=2500, network_type='walk')

df = get_obstacle_data(sheet_url)
graph = get_graph_data()
geolocator = Nominatim(user_agent="my_bfree_nav_v3")

# 3. 사이드바 - 입력 방식 선택
st.sidebar.header("📍 경로 설정 방식")
input_method = st.sidebar.radio("입력 방식을 선택하세요", ["장소 이름 검색", "위도/경도 직접 입력"])

start_coords, end_coords = None, None

if input_method == "장소 이름 검색":
    start_input = st.sidebar.text_input("출발지 (예: 감계중학교)", value="감계중학교")
    end_input = st.sidebar.text_input("목적지 (예: 창원북면고등학교)", value="창원북면고등학교")
    
    def get_coords(address):
        try:
            location = geolocator.geocode(f"{address.strip()}, 창원시")
            if not location:
                location = geolocator.geocode(f"{address.strip()}, 경상남도")
            return (location.latitude, location.longitude) if location else None
        except: return None

    if st.sidebar.button("🚀 경로 탐색"):
        start_coords = get_coords(start_input)
        end_coords = get_coords(end_input)
        if not start_coords or not end_coords:
            st.sidebar.error("장소를 찾을 수 없습니다. 좌표 입력 방식을 이용해 보세요.")

else:  # 위도/경도 직접 입력
    st.sidebar.write("구글 지도 등에서 좌표를 복사해 넣으세요.")
    s_lat = st.sidebar.number_input("출발지 위도", value=35.299396, format="%.6f")
    s_lon = st.sidebar.number_input("출발지 경도", value=128.595954, format="%.6f")
    e_lat = st.sidebar.number_input("목적지 위도", value=35.302278, format="%.6f")
    e_lon = st.sidebar.number_input("목적지 경도", value=128.593880, format="%.6f")
    
    if st.sidebar.button("🚀 경로 탐색"):
        start_coords = (s_lat, s_lon)
        end_coords = (e_lat, e_lon)

# 4. 경로 계산 및 시각화 (좌표가 결정된 경우에만 실행)
if start_coords and end_coords:
    with st.spinner('장애물을 우회하는 경로를 계산 중...'):
        # 가중치 설정 로직 (동일)
        for u, v, k, data in graph.edges(keys=True, data=True):
            data['barrier_free_weight'] = data['length']
            edge_shape = data.get('geometry', LineString([(graph.nodes[u]['x'], graph.nodes[u]['y']), (graph.nodes[v]['x'], graph.nodes[v]['y'])]))
            if not df.empty:
                for _, row in df.iterrows():
                    if edge_shape.distance(Point(row['경도'], row['위도'])) < 0.00025:
                        data['barrier_free_weight'] *= 15
                        break

        orig_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
        dest_node = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])
        
        try:
            route = nx.shortest_path(graph, orig_node, dest_node, weight='barrier_free_weight')
            fig, ax = ox.plot_graph_route(graph, route, route_color='#3b82f6', node_size=0, bgcolor='white', edge_color='#e2e8f0', show=False, close=False)
            
            if not df.empty:
                ax.scatter(df['경도'], df['위도'], c='#ef4444', s=60, label='Obstacles', zorder=5)
            ax.scatter(start_coords[1], start_coords[0], c='#10b981', s=150, marker='s', label='Start', zorder=6)
            ax.scatter(end_coords[1], end_coords[0], c='#3b82f6', s=150, marker='X', label='Goal', zorder=6)
            ax.legend()
            st.pyplot(fig)
            st.success("✅ 경로 탐색 완료!")
        except Exception as e:
            st.error(f"경로를 찾을 수 없습니다: {e}")
