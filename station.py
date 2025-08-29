# app.py

import streamlit as st
import pandas as pd
import re
import datetime
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

# --- 웹페이지 기본 설정 ---
st.set_page_config(page_title="지하철 혼잡도 예측", layout="wide")
st.title("🚇 지하철 경로 혼잡도 예측 시스템")

# --- 폰트 설정 (한글 깨짐 방지) ---
@st.cache_resource
def set_korean_font():
    font_path = None
    for font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
        if 'Malgun Gothic' in font or 'malgun' in font.lower():
            font_path = font
            break
        if 'AppleGothic' in font:
            font_path = font
            break
            
    if font_path:
        rc('font', family=font_manager.FontProperties(fname=font_path).get_name())
        plt.rcParams['axes.unicode_minus'] = False
    else:
        st.warning("경고: 'Malgun Gothic' 또는 'AppleGothic' 폰트를 찾을 수 없습니다.")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# --- 데이터 로드 ---
@st.cache_data
def load_data():
    try:
        passenger_df = pd.read_csv('서울교통공사_역별 시간대별 승하차인원(24.1~24.12).csv', encoding='cp949')
        station_df = pd.read_csv('실시간도착_역정보(20250801)_data.csv', encoding='cp949')
        def clean_station_name(name): return re.sub(r'\([^)]*\)', '', str(name))
        passenger_df['역명'] = passenger_df['역명'].apply(clean_station_name)
        station_df['STATN_NM'] = station_df['STATN_NM'].apply(clean_station_name)
        passenger_df['날짜'] = pd.to_datetime(passenger_df['날짜'])
        return passenger_df, station_df
    except FileNotFoundError:
        st.error("오류: 데이터 CSV 파일을 찾을 수 없습니다. app.py와 동일한 폴더에 파일이 있는지 확인해주세요.")
        return None, None

passenger_data, station_info = load_data()

# --- 분석 함수들 ---
def get_route(start, end, line, station_df):
    line_name_simple = line.replace('호선', '')
    line_stations_df = station_df[station_df['호선이름'].str.contains(line_name_simple, na=False)]
    line_stations = line_stations_df['STATN_NM'].unique().tolist()
    if start not in line_stations or end not in line_stations: return None
    start_idx, end_idx = line_stations.index(start), line_stations.index(end)
    return line_stations[start_idx : end_idx + 1] if start_idx <= end_idx else line_stations[start_idx : end_idx - 1 : -1]

def predict_ridership(station, line, ride_type, future_dt, time_col, p_data):
    time_col_for_filename = time_col.replace('시-', '_').replace('시', '').replace(' 이전', '_before').replace(' 이후', '_after')
    model_filename = f"models/{line}_{station}_{ride_type}_{time_col_for_filename}_model.joblib"
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
    else:
        df = p_data[(p_data['역명'] == station) & (p_data['호선'] == line) & (p_data['구분'] == ride_type)].copy()
        if df.empty or df[time_col].isnull().all(): return 0
        df['월'] = df['날짜'].dt.month
        df['요일'] = df['날짜'].dt.dayofweek
        df['주말'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)
        df.dropna(subset=[time_col], inplace=True)
        if len(df) < 5: return int(df[time_col].mean()) if not df.empty else 0
        X, y = df[['월', '요일', '주말']], df[time_col]
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)
        if not os.path.exists('models'): os.makedirs('models')
        joblib.dump(model, model_filename)
    day_of_week_val = future_dt.weekday()
    future_features = pd.DataFrame([{'월': future_dt.month, '요일': day_of_week_val, '주말': 1 if day_of_week_val >= 5 else 0}])
    prediction = model.predict(future_features)
    return int(prediction[0])

# --- 웹페이지 UI 구성 ---
if passenger_data is not None:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        available_lines = sorted(station_info['호선이름'].str.replace('0', '').unique())
        line = st.selectbox("호선 선택", available_lines, index=1)
    with col2:
        start_station = st.text_input("출발역", "강남")
    with col3:
        end_station = st.text_input("도착역", "홍대입구")
    with col4:
        prediction_date_input = st.date_input("예측 날짜", datetime.date.today())
    with col5:
        prediction_time_input = st.time_input("예측 시간", datetime.datetime.now().time().replace(minute=0, second=0, microsecond=0))

    if st.button("혼잡도 예측 실행"):
        full_prediction_datetime = datetime.datetime.combine(prediction_date_input, prediction_time_input)
        target_hour = full_prediction_datetime.hour
        time_col_name = next((col for col in passenger_data.columns if f'{target_hour:02d}시' in col), None)

        if not time_col_name:
            st.error(f"오류: 선택하신 시간({target_hour}시)에 해당하는 데이터 컬럼을 찾을 수 없습니다.")
        else:
            with st.spinner(f'{line} {start_station}역에서 {end_station}역까지의 혼잡도를 예측하는 중입니다...'):
                route = get_route(start_station, end_station, line, station_info)
                if route:
                    predictions = []
                    progress_bar = st.progress(0, text="예측 시작...")
                    for i, station in enumerate(route):
                        boarding = predict_ridership(station, line, '승차', full_prediction_datetime, time_col_name, passenger_data)
                        alighting = predict_ridership(station, line, '하차', full_prediction_datetime, time_col_name, passenger_data)
                        predictions.append({'역명': station, '승차': boarding, '하차': alighting})
                        progress_bar.progress((i + 1) / len(route), text=f"'{station}' 역 예측 완료")
                    
                    predicted_df = pd.DataFrame(predictions).set_index('역명')
                    st.success("✅ 예측이 완료되었습니다!")

                    # --- ✨ 그래프 추가 부분 ✨ ---
                    st.subheader("📊 예측 결과 분석")
                    
                    # 탭 생성
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["데이터", "승하차 막대그래프", "승하차 라인그래프", "역별 순증감", "예상 누적 혼잡도"])

                    with tab1:
                        st.write("각 역별 예측된 승하차 인원입니다.")
                        st.dataframe(predicted_df)

                    with tab2:
                        st.write("각 역의 승차와 하차 인원을 비교합니다.")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        predicted_df.plot(kind='bar', ax=ax, width=0.8, color=['#4285F4', '#FBBC05'])
                        for p in ax.patches:
                            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
                        ax.set_title(f'[{line}] {start_station}역 → {end_station}역: 승하차 인원 예측', fontsize=16)
                        ax.set_xlabel('역명')
                        ax.set_ylabel('예측 인원 수')
                        plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.legend(title='구분')
                        st.pyplot(fig)

                    with tab3:
                        st.write("경로에 따른 승하차 인원의 변화 추이를 보여줍니다.")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        predicted_df.plot(kind='line', ax=ax, style='-o', color=['#4285F4', '#FBBC05'])
                        ax.set_title(f'[{line}] {start_station}역 → {end_station}역: 승하차 추이', fontsize=16)
                        ax.set_xlabel('역명')
                        ax.set_ylabel('예측 인원 수')
                        plt.xticks(rotation=45, ha='right'); plt.grid(True, linestyle='--', alpha=0.7); plt.legend(title='구분')
                        st.pyplot(fig)

                    with tab4:
                        st.write("각 역에서 실제 승객이 얼마나 늘거나 줄었는지 보여줍니다.")
                        predicted_df['순증감'] = predicted_df['승차'] - predicted_df['하차']
                        colors = ['#d9534f' if x < 0 else '#5cb85c' for x in predicted_df['순증감']]
                        fig, ax = plt.subplots(figsize=(12, 6))
                        predicted_df['순증감'].plot(kind='bar', ax=ax, color=colors)
                        for p in ax.patches:
                             ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9 if p.get_height() > 0 else -15), textcoords='offset points')
                        ax.set_title(f'[{line}] {start_station}역 → {end_station}역: 역별 순증감', fontsize=16)
                        ax.set_xlabel('역명')
                        ax.set_ylabel('순증감 인원 수 (승차-하차)')
                        plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', linestyle='--', alpha=0.7)
                        st.pyplot(fig)
                        
                    with tab5:
                        st.write("출발 후 열차에 승객이 누적되는 과정을 보여줍니다. 가장 높은 지점이 최고 혼잡 구간입니다.")
                        predicted_df['누적 혼잡도'] = predicted_df['순증감'].cumsum()
                        fig, ax = plt.subplots(figsize=(12, 6))
                        predicted_df['누적 혼잡도'].plot(kind='line', ax=ax, style='-o', color='#0275d8', marker='o')
                        ax.set_title(f'[{line}] {start_station}역 → {end_station}역: 예상 누적 혼잡도', fontsize=16)
                        ax.set_xlabel('역명')
                        ax.set_ylabel('예상 누적 인원 수')
                        plt.xticks(rotation=45, ha='right'); plt.grid(True, linestyle='--', alpha=0.7)
                        st.pyplot(fig)
                else:
                    st.error(f"오류: '{start_station}' 또는 '{end_station}'이 포함된 {line} 경로를 찾을 수 없습니다.")