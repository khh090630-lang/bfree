import streamlit as st
import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from geopy.geocoders import Nominatim
import time

# 1. 페이지 기본 설정
st.set_page_config(page_title="감계 배리어프리 내비", layout="wide")
st.title("🗺️ 감계지구 스마트 우회 내비게이션")
st.markdown("사용자가 장소를 입력하면 **구글 시트의 실시간 장애물 데이터**를 반영하여 최적의 우회 경로를 계산합니다.")

# 2. 데이터 및 지도 로드 (성능 최적화용 캐싱)
sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ9_vnph9VqvmqqmA-_njbzjKR9dKTIOhFESErGsSSGaiQ9617tOmurA4Y8C9c-wu1t2LKQXtSPtEVk/pub?output=csv"

@st.cache_data(ttl=300)
def get_obstacle_data(url):
    try:
        return pd.read_csv(url)
    except:
        return pd.DataFrame()

@st.cache_resource
def get_graph_data():
    # 감계지구 중심점 및 탐색 범위 설정 (1.5km)
    center_point = (35.300, 128.595)
    return ox.graph_from_point(center_point, dist=1500, network_type='walk')

df = get_obstacle_data(sheet_url)
graph = get_graph_data()
geolocator = Nominatim(user_agent="my_bfree_nav_v2")

# 3. 사이드바 - 장소 검색 및 설정
st.sidebar.header("🔍 장소 검색")
st.sidebar.write("창원 감계 지역 내 장소명을 입력하세요.")

start_input = st.sidebar.text_input("출발지 (예: 감계중학교)", value="감계중학교")
end_input = st.sidebar.text_input("목적지 (예: 북면사무소)", value="북면사무소")

# 주소를 좌표로 변환하는 함수 (검색 보정 로직 포함)
def get_coords(address):
    if not address: return None
    try:
        # 검색 확률을 높이기 위해 "창원시"를 자동 부착
        query = f"{address.strip()}, 창원시"
        location = geolocator.geocode(query)
        if not location:
            # 실패 시 "경남 창원시"로 재시도
            query = f"{address.strip()}, 경상남도 창원시"
            location = geolocator.geocode(query)
        
        if location:
            return (location.latitude, location.longitude)
        return None
    except:
        return None

# 4. 경로 탐색 실행 버튼
if st.sidebar.button("🚀 경로 탐색 시작"):
    with st.spinner('최적의 우회 경로를 찾고 있습니다...'):
        start_coords = get_coords(start_input)
        end_coords = get_coords(end_input)

        if start_coords and end_coords:
            # 가중치 계산 로직 (장애물 우회 페널티 부여)
            DETECTION_RADIUS = 0.00025  # 약 25m
            OBSTACLE_MULTIPLIER = 15    # 장애물 발견 시 15배 우회

            # 가중치 초기화 및 장애물 검사
            for u, v, k, data in graph.edges(keys=True, data=True):
                data['barrier_free_weight'] = data['length']
                edge_shape = data.get('geometry', LineString([(graph.nodes[u]['x'], graph.nodes[u]['y']), (graph.nodes[v]['x'], graph.nodes[v]['y'])]))
                
                if not df.empty:
                    for _, row in df.iterrows():
                        obstacle_p = Point(row['경도'], row['위도'])
                        if edge_shape.distance(obstacle_p) < DETECTION_RADIUS:
                            data['barrier_free_weight'] *= OBSTACLE_MULTIPLIER
                            break

            # 지도 상의 가장 가까운 노드 찾기
            orig_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
            dest_node = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])
            
            try:
                # 우회 가중치를 반영한 최단 경로 탐색
                route = nx.shortest_path(graph, orig_node, dest_node, weight='barrier_free_weight')
                
                # 결과 시각화
                fig, ax = ox.plot_graph_route(
                    graph, route, route_color='#3b82f6', route_linewidth=5, 
                    node_size=0, bgcolor='white', edge_color='#e2e8f0', 
                    show=False, close=False
                )

                # 장애물 포인트 표시
                if not df.empty:
                    ax.scatter(df['경도'], df['위도'], c='#ef4444', s=60, label='Obstacles', zorder=5, edgecolors='white')

                # 시작/종료 마커
                ax.scatter(start_coords[1], start_coords[0], c='#10b981', s=150, marker='s', label='Start', zorder=6)
                ax.scatter(end_coords[1], end_coords[0], c='#3b82f6', s=150, marker='X', label='Goal', zorder=6)
                
                ax.legend()
                st.pyplot(fig)
                st.success(f"✅ '{start_input}'에서 '{end_input}'까지의 무장애 경로를 찾았습니다!")
                
            except Exception as e:
                st.error(f"⚠️ 경로를 계산할 수 없습니다: {e}")
        else:
            st.error("📍 입력하신 장소를 찾을 수 없습니다. (지역명 포함하여 더 정확하게 입력해 보세요)")

else:
    st.info("왼쪽 사이드바에서 장소를 입력하고 버튼을 눌러주세요.")

# 데이터 현황 안내
with st.expander("📊 현재 시스템 데이터 정보"):
    st.write(f"최신 장애물 데이터 개수: {len(df)}개")
    if not df.empty:
        st.dataframe(df.head())
