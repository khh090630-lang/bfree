# [1] 필수 라이브러리 설치 (로컬/배포 환경용)
# !pip install osmnx pandas networkx matplotlib shapely streamlit

import streamlit as st  # 웹 인터페이스 도구 추가
import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString

# ==========================================
# [웹 설정] 페이지 제목 및 레이아웃
# ==========================================
st.set_page_config(page_title="감계지구 배리어프리 내비게이션", layout="centered")
st.title("🚶‍♂️ 장애물 우회 경로 시뮬레이션")
st.write("구글 스프레드시트의 장애물 데이터를 실시간으로 반영하여 최적 우회 경로를 계산합니다.")

# ==========================================
# [1] 데이터 및 도로망 로드
# ==========================================
sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ9_vnph9VqvmqqmA-_njbzjKR9dKTIOhFESErGsSSGaiQ9617tOmurA4Y8C9c-wu1t2LKQXtSPtEVk/pub?output=csv"

@st.cache_data(ttl=300) # 5분마다 데이터 새로고침 (웹 성능 최적화)
def get_data(url):
    try:
        return pd.read_csv(url)
    except:
        return pd.DataFrame()

df = get_data(sheet_url)

if not df.empty:
    st.success(f"✅ 데이터 동기화 완료! 현재 장애물: {len(df)}개 반영 중")
else:
    st.error("❌ 데이터를 가져오지 못했습니다.")

# 감계지구 도로망 불러오기
@st.cache_resource # 지도 데이터는 무거우므로 한 번만 불러오도록 고정
def get_graph():
    center_point = (35.300, 128.595)
    return ox.graph_from_point(center_point, dist=1000, network_type='walk')

graph = get_graph()

# ==========================================
# [2] 스마트 가중치 설정 (최적 우회 로직) - 기존 로직 유지
# ==========================================
DETECTION_RADIUS = 0.00025  
OBSTACLE_MULTIPLIER = 15    

for u, v, k, data in graph.edges(keys=True, data=True):
    base_length = data['length']
    data['barrier_free_weight'] = base_length

    if 'geometry' not in data:
        u_node = graph.nodes[u]
        v_node = graph.nodes[v]
        edge_shape = LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])
    else:
        edge_shape = data['geometry']

    if not df.empty:
        for _, row in df.iterrows():
            obstacle_p = Point(row['경도'], row['위도'])
            if edge_shape.distance(obstacle_p) < DETECTION_RADIUS:
                data['barrier_free_weight'] = base_length * OBSTACLE_MULTIPLIER
                break

# ==========================================
# [3] 좌표 설정 및 경로 탐색 - 기존 좌표 유지
# ==========================================
start_coords = (35.299396, 128.595954)
end_coords = (35.302278, 128.593880)

orig_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
dest_node = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])

try:
    route = nx.shortest_path(graph, orig_node, dest_node, weight='barrier_free_weight')
    st.info("✅ 최적 우회 경로 탐색에 성공했습니다.")
except Exception as e:
    st.warning(f"❌ 경로 탐색 실패: {e}")
    route = []

# ==========================================
# [4] 시각화 및 웹 출력
# ==========================================
if route:
    fig, ax = ox.plot_graph_route(
        graph, route, route_color='#3b82f6', route_linewidth=5,
        node_size=0, bgcolor='white', edge_color='#e2e8f0', show=False, close=False
    )

    if not df.empty:
        ax.scatter(df['경도'], df['위도'], c='#ef4444', s=60, label='Obstacles', zorder=5, edgecolors='white')

    ax.scatter(start_coords[1], start_coords[0], c='#10b981', s=150, marker='s', label='Start', zorder=6)
    ax.scatter(end_coords[1], end_coords[0], c='#3b82f6', s=150, marker='X', label='Goal', zorder=6)
    
    ax.legend()
    
    # plt.show() 대신 streamlit 전용 출력 함수 사용
    st.pyplot(fig)

    # 경로 정보 출력
    route_coords = [[graph.nodes[node]['y'], graph.nodes[node]['x']] for node in route]
    st.write(f"📍 첫 번째 경로 좌표: {route_coords[0]}")