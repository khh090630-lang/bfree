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



# [3] 경로 탐색 및 시각화

if start_coords and end_coords:

    # ⚠️ 중요: 그래프 복사본을 생성하여 매번 깨끗한 상태에서 가중치 부여
    G = graph.copy()

    # 장애물 감지 범위 및 페널티 설정
    DETECTION_RADIUS = 0.0001  # 약 10m로 약간 확장
    PENALTY = 50              # 장애물 통과 시 거리를 50배로 계산 (강력 우회)

    # 모든 간선(Edge)에 대해 가중치 계산
    for u, v, k, data in G.edges(keys=True, data=True):
        # 기본 가중치는 실제 거리(length)
        data['my_weight'] = data['length']
        
        # 도로의 모양(geometry) 추출
        if 'geometry' in data:
            edge_geom = data['geometry']
        else:
            edge_geom = LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
        
        # 장애물 데이터와 대조
        if not df.empty:
            for _, row in df.iterrows():
                obs_point = Point(row['경도'], row['위도'])
                if edge_geom.distance(obs_point) < DETECTION_RADIUS:
                    data['my_weight'] = data['length'] * PENALTY
                    break

    # 최단 경로 노드 찾기
    orig_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
    dest_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])

    try:
        # ⚠️ 중요: weight 파라미터에 우리가 만든 'my_weight'를 지정해야 함
        route = nx.shortest_path(G, orig_node, dest_node, weight='my_weight')
        
        # 확대 로직용 좌표 리스트 (출발/목적지 실제 좌표 포함)
        route_nodes = [G.nodes[node] for node in route]
        lats = [n['y'] for n in route_nodes] + [start_coords[0], end_coords[0]]
        lons = [n['x'] for n in route_nodes] + [start_coords[1], end_coords[1]]
        bbox = (max(lats)+0.001, min(lats)-0.001, max(lons)+0.001, min(lons)-0.001)

        # 기본 그래프 경로 그리기
        fig, ax = ox.plot_graph_route(G, route, route_color='#3b82f6', route_linewidth=5, node_size=0, bgcolor='white', show=False, close=False)
        ax.set_ylim(bbox[1], bbox[0]); ax.set_xlim(bbox[3], bbox[2])

        # --- [수정된 선 연결 로직] 출발/목적지 실제 좌표와 경로 끝점을 선으로 잇기 ---
        # 1. 출발지 실제 좌표 -> 첫 번째 노드 좌표 연결
        start_node_coords = (G.nodes[orig_node]['x'], G.nodes[orig_node]['y'])
        ax.plot([start_coords[1], start_node_coords[0]], [start_coords[0], start_node_coords[1]], 
                color='#3b82f6', linewidth=5, zorder=4)

        # 2. 목적지 실제 좌표 -> 마지막 노드 좌표 연결
        end_node_coords = (G.nodes[dest_node]['x'], G.nodes[dest_node]['y'])
        ax.plot([end_coords[1], end_node_coords[0]], [end_coords[0], end_node_coords[1]], 
                color='#3b82f6', linewidth=5, zorder=4)
        # --------------------------------------------------------------------------

        if not df.empty:
            ax.scatter(df['경도'], df['위도'], c='#ef4444', s=60, label='Obstacle', zorder=5, edgecolors='white')
        
        # 마커 표시 (zorder를 높여 선 위에 오도록 설정)
        ax.scatter(start_coords[1], start_coords[0], c='#10b981', s=150, marker='s', label='Start', zorder=6, edgecolors='white')
        ax.scatter(end_coords[1], end_coords[0], c='#3b82f6', s=150, marker='X', label='Goal', zorder=6, edgecolors='white')
        
        st.pyplot(fig)
        st.success("장애물을 우회하는 최적 경로를 찾았습니다!")
    except Exception as e:
        st.error(f"경로를 찾을 수 없습니다: {e}")


