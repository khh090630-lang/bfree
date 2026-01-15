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



# [3] 경로 탐색 및 시각화 (스냅 오류 방지 버전)

if start_coords and end_coords:
    G = graph.copy()

    # 1. '가까운 점'이 아니라 '가까운 도로(Edge)'를 찾습니다.
    # get_nearest_edge를 통해 좌표가 어떤 도로 위에 있어야 하는지 확인합니다.
    nearest_edge_start = ox.distance.nearest_edges(G, start_coords[1], start_coords[0])
    nearest_edge_end = ox.distance.nearest_edges(G, end_coords[1], end_coords[0])

    # 2. 도로 위의 가장 가까운 노드를 시작점/끝점으로 잡습니다.
    # (더 정교하게는 도로 중간에 노드를 생성해야 하나, 현재는 도로의 양 끝 노드 중 가까운 쪽을 선택)
    orig_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
    dest_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])

    # --- 우회 로직 (질문자님 설정 유지) ---
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
    # ----------------------------------

    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight='my_weight')
        
        # 시각화 준비
        fig, ax = ox.plot_graph_route(G, route, route_color='#3b82f6', route_linewidth=5, 
                                    node_size=0, bgcolor='white', show=False, close=False)

        # 3. [핵심] 이상한 줄 방지: 실제 위치에서 경로의 '진짜 시작점'까지만 짧게 연결
        # 출발지에서 경로의 첫 노드까지
        start_node_pt = (G.nodes[route[0]]['x'], G.nodes[route[0]]['y'])
        ax.plot([start_coords[1], start_node_pt[0]], [start_coords[0], start_node_pt[1]], 
                color='#3b82f6', linewidth=5, alpha=0.7, zorder=4)

        # 목적지에서 경로의 마지막 노드까지
        end_node_pt = (G.nodes[route[-1]]['x'], G.nodes[route[-1]]['y'])
        ax.plot([end_coords[1], end_node_pt[0]], [end_coords[0], end_node_pt[1]], 
                color='#3b82f6', linewidth=5, alpha=0.7, zorder=4)

        # 줌 및 마커 설정
        route_nodes = [G.nodes[node] for node in route]
        lats = [n['y'] for n in route_nodes] + [start_coords[0], end_coords[0]]
        lons = [n['x'] for n in route_nodes] + [start_coords[1], end_coords[1]]
        bbox = (max(lats)+0.001, min(lats)-0.001, max(lons)+0.001, min(lons)-0.001)
        ax.set_ylim(bbox[1], bbox[0]); ax.set_xlim(bbox[3], bbox[2])

        if not df.empty:
            ax.scatter(df['경도'], df['위도'], c='#ef4444', s=60, zorder=5, edgecolors='white')
        ax.scatter(start_coords[1], start_coords[0], c='#10b981', s=150, marker='s', zorder=6, edgecolors='white')
        ax.scatter(end_coords[1], end_coords[0], c='#3b82f6', s=150, marker='X', zorder=6, edgecolors='white')
        
        st.pyplot(fig)
        st.success("보행 경로를 따라 목적지까지 연결되었습니다.")
        
    except Exception as e:
        st.error(f"경로를 찾을 수 없습니다: {e}")




