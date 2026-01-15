import streamlit as st
import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from geopy.geocoders import Nominatim  # 주소 -> 좌표 변환 도구

# 페이지 설정
st.set_page_config(page_title="감계 배리어프리 내비", layout="wide")
st.title("🗺️ 감계지구 스마트 우회 내비게이션")

# [1] 데이터 및 지도 로드 (캐싱 적용)
sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ9_vnph9VqvmqqmA-_njbzjKR9dKTIOhFESErGsSSGaiQ9617tOmurA4Y8C9c-wu1t2LKQXtSPtEVk/pub?output=csv"

@st.cache_data(ttl=300)
def get_data(url):
    return pd.read_csv(url)

@st.cache_resource
def get_graph():
    center_point = (35.300, 128.595)
    return ox.graph_from_point(center_point, dist=1500, network_type='walk') # 범위를 조금 넓힘

df = get_data(sheet_url)
graph = get_graph()
geolocator = Nominatim(user_agent="my_navigation_app")

# [2] 사이드바 장소 검색창
st.sidebar.header("🔍 장소 검색")
start_input = st.sidebar.text_input("출발지 입력", value="창원 감계 푸르지오")
end_input = st.sidebar.text_input("목적지 입력", value="감계중학교")

def get_coords(address):
    try:
        # 검색 범위를 창원으로 한정하여 정확도 높임
        location = geolocator.geocode(address + ", 창원시")
        if location:
            return (location.latitude, location.longitude)
        return None
    except:
        return None

# [3] 경로 탐색 실행
if st.sidebar.button("경로 탐색 시작"):
    start_coords = get_coords(start_input)
    end_coords = get_coords(end_input)

    if start_coords and end_coords:
        # 가중치 계산 (기존 로직 유지)
        DETECTION_RADIUS = 0.00025
        OBSTACLE_MULTIPLIER = 15
        
        for u, v, k, data in graph.edges(keys=True, data=True):
            data['barrier_free_weight'] = data['length']
            edge_shape = data.get('geometry', LineString([(graph.nodes[u]['x'], graph.nodes[u]['y']), (graph.nodes[v]['x'], graph.nodes[v]['y'])]))
            for _, row in df.iterrows():
                if edge_shape.distance(Point(row['경도'], row['위도'])) < DETECTION_RADIUS:
                    data['barrier_free_weight'] *= OBSTACLE_MULTIPLIER
                    break

        # 노드 찾기 및 경로 계산
        orig_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
        dest_node = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])
        
        try:
            route = nx.shortest_path(graph, orig_node, dest_node, weight='barrier_free_weight')
            
            # 시각화
            fig, ax = ox.plot_graph_route(graph, route, route_color='#3b82f6', route_linewidth=5, node_size=0, bgcolor='white', edge_color='#e2e8f0', show=False, close=False)
            if not df.empty:
                ax.scatter(df['경도'], df['위도'], c='#ef4444', s=60, label='Obstacles', zorder=5, edgecolors='white')
            ax.scatter(start_coords[1], start_coords[0], c='#10b981', s=150, marker='s', label='Start', zorder=6)
            ax.scatter(end_coords[1], end_coords[0], c='#3b82f6', s=150, marker='X', label='Goal', zorder=6)
            ax.legend()
            st.pyplot(fig)
            st.success(f"📍 '{start_input}'에서 '{end_input}'까지의 우회 경로입니다.")
            
        except Exception as e:
            st.error(f"경로를 찾을 수 없습니다. (범위 초과 등): {e}")
    else:
        st.error("입력하신 장소의 좌표를 찾을 수 없습니다. 더 정확하게 입력해 주세요.")
