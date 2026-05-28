import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import matplotlib
from sklearn.preprocessing import MinMaxScaler
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ─── 페이지 설정 ─────────────────────────────────────────────
st.set_page_config(page_title="관악구 자취 점수 산출기", layout="wide")

# ─── 폰트 설정 ───────────────────────────────────────────────
try:
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
except Exception:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ─── 전역 설정 ───────────────────────────────────────────────
BASE = 'data/'

카테고리 = ['주거비', '대중교통', '서울대', '안전', '식당', '병원/약국', '카페', '세탁소', '마트']
컬럼맵 = {
    '주거비':    '주거비_점수',
    '대중교통':  '교통_점수',
    '서울대':    '서울대_점수',
    '안전':      '안전_점수_정규화',
    '식당':      '식당_점수_정규화',
    '병원/약국': '병원_약국_점수_정규화',
    '카페':      '카페_점수_정규화',
    '세탁소':    '세탁소_점수_정규화',
    '마트':      '마트_슈퍼_점수_정규화',
}

VALID_DONGS = [
    '보라매동', '청림동', '성현동', '행운동', '낙성대동', '청룡동',
    '은천동', '중앙동', '인헌동', '남현동', '서원동', '신원동',
    '서림동', '신사동', '신림동', '난향동', '조원동', '대학동',
    '삼성동', '미성동', '난곡동'
]

VOTE_FILE = 'votes.json'


# ════════════════════════════════════════════════════════════
# 데이터 전처리 함수
# ════════════════════════════════════════════════════════════

def get_rent_data():
    """주거 파트: 전월세 원본 → 행정동별 1㎡당 환산월세"""
    # 1. 환산월세 계산 (연 5% 전월세전환율)
    df = pd.read_csv(BASE + 'rent.csv', encoding='utf-8-sig')
    df = df[df['임대면적(㎡)'] > 0].copy()
    df['환산월세(만원)'] = (df['보증금(만원)'] * 0.05 / 12) + df['임대료(만원)']
    df['1㎡당_환산월세'] = df['환산월세(만원)'] / df['임대면적(㎡)']

    # 2. 법정동별 평균 환산월세 집계
    mean_rent = (df.groupby('법정동명')['1㎡당_환산월세']
                   .mean().round(2).reset_index()
                   .rename(columns={'1㎡당_환산월세': '환산월세'}))

    # 3. KIK 매핑: 법정동 → 행정동
    kik = pd.read_excel(BASE + 'kik.xlsx')
    kik_gwanak = (kik[kik['시군구명'] == '관악구'][['읍면동명', '동리명']]
                    .dropna()
                    .rename(columns={'읍면동명': '행정동', '동리명': '법정동명'})
                    .drop_duplicates())
    kik_gwanak = kik_gwanak[kik_gwanak['행정동'].isin(VALID_DONGS)]

    # 4. 면적 데이터 (비례배분용)
    area_raw = pd.read_csv(BASE + 'area.csv', encoding='utf-8-sig')
    area_df = (
        area_raw[(area_raw['동별(2)'] == '관악구') & (area_raw['동별(3)'] != '소계')]
        [['동별(3)', '2025']].copy()
        .rename(columns={'동별(3)': '행정동', '2025': '면적_km2'})
    )
    area_df['면적_km2'] = pd.to_numeric(area_df['면적_km2'], errors='coerce')

    # 5. 법정동 내 면적 비율 계산 후 행정동에 환산월세 배분
    kik_area = kik_gwanak.merge(area_df, on='행정동', how='left')
    법정동_총면적 = (kik_area.groupby('법정동명')['면적_km2'].sum()
                              .reset_index().rename(columns={'면적_km2': '법정동_총면적'}))
    kik_area = kik_area.merge(법정동_총면적, on='법정동명')
    kik_area['면적비율'] = kik_area['면적_km2'] / kik_area['법정동_총면적']

    result = (kik_area.merge(mean_rent, on='법정동명', how='left')
                      [['행정동', '환산월세']]
                      .rename(columns={'행정동': 'dong'}))
    return result


def get_amenity_data():
    """편의시설 파트: 점포/매출/상비약 원본 → 행정동별 편의시설 점수"""
    # 내부 가중치 (자취생 체감도 반영)
    W_MED_INNER  = {'일반의원': 2.0, '의약품': 1.5, '치과의원': 1.0, '한의원': 0.5}
    W_MART_INNER = {'편의점': 2.0, '반찬가게': 1.5, '슈퍼마켓': 1.0}
    W_REST_INNER = {'분식전문점': 1.5, '패스트푸드점': 1.5, '치킨전문점': 1.2,
                    '한식음식점': 1.0, '중식음식점': 1.0, '일식음식점': 1.0, '양식음식점': 0.8}
    W_CAFE_INNER = {'커피-음료': 1.0, '제과점': 1.2}

    W_RESTAURANT, W_MEDICAL = 0.25, 0.20
    W_CAFE, W_LAUNDRY, W_MART = 0.15, 0.20, 0.20
    AREA_WEIGHT, COUNT_WEIGHT = 0.5, 0.5

    # 데이터 로드
    df_stores = pd.read_csv(BASE + 'stores.csv',           encoding='utf-8-sig')
    df_sales  = pd.read_csv(BASE + 'sales.csv',             encoding='utf-8-sig')
    df_med    = pd.read_csv(BASE + 'medicine.csv', encoding='utf-8-sig')
    area_raw  = pd.read_csv(BASE + 'area.csv',                 encoding='utf-8-sig')

    # 면적 테이블
    area_df = (
        area_raw[(area_raw['동별(2)'] == '관악구') & (area_raw['동별(3)'] != '소계')]
        [['동별(3)', '2025']].copy()
        .rename(columns={'동별(3)': '행정동', '2025': '면적_km2'})
    )
    area_df['면적_km2'] = pd.to_numeric(area_df['면적_km2'], errors='coerce')
    area_series = area_df.set_index('행정동')['면적_km2']

    # 최신 분기 점포 pivot
    latest_q      = df_stores['기준_년분기_코드'].max()
    stores_latest = df_stores[df_stores['기준_년분기_코드'] == latest_q]
    pivot_counts  = stores_latest.pivot_table(
        index='행정동_코드_명', columns='서비스_업종_코드_명',
        values='점포_수', aggfunc='sum').fillna(0)
    pivot_density = pivot_counts.div(area_series, axis=0)

    def minmax(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    def blend(col_name):
        raw   = pivot_counts[col_name]  if col_name in pivot_counts.columns  else pd.Series(0, index=pivot_counts.index)
        dense = pivot_density[col_name] if col_name in pivot_density.columns else pd.Series(0, index=pivot_density.index)
        return minmax(raw) * COUNT_WEIGHT + minmax(dense) * AREA_WEIGHT

    def blend_weighted_sum(weight_dict):
        total = pd.Series(0.0, index=pivot_counts.index)
        for col, weight in weight_dict.items():
            total += blend(col) * weight
        return total

    # 카테고리별 인프라 점수 계산
    master_df = pd.DataFrame(index=pivot_counts.index)
    master_df['식당_인프라']     = blend_weighted_sum(W_REST_INNER)
    master_df['카페_인프라']     = blend_weighted_sum(W_CAFE_INNER)
    master_df['세탁소_인프라']   = blend('세탁소')
    master_df['마트슈퍼_인프라'] = blend_weighted_sum(W_MART_INNER)
    master_df['기본의료_인프라'] = blend_weighted_sum(W_MED_INNER)

    # 식당 2030 매출 비중 보정
    sales_latest = df_sales[df_sales['기준_년분기_코드'] == df_sales['기준_년분기_코드'].max()]
    sales_rest   = sales_latest[sales_latest['서비스_업종_코드_명'].isin(W_REST_INNER.keys())]
    sales_agg    = sales_rest.groupby('행정동_코드_명').agg(
        당월=('당월_매출_건수', 'sum'),
        이십대=('연령대_20_매출_건수', 'sum'),
        삼십대=('연령대_30_매출_건수', 'sum'))
    sales_agg['식당_2030비중'] = (sales_agg['이십대'] + sales_agg['삼십대']) / sales_agg['당월']
    master_df = master_df.join(sales_agg['식당_2030비중']).fillna(0)

    # 상비약 편의점 보정
    pharmacy_counts = (
        df_med[df_med['행정동명'] != '기타']
        .groupby('행정동명').size()
        .rename('상비약_편의점수'))
    master_df = master_df.join(pharmacy_counts).fillna(0)

    # 0~100점 정규화 및 최종 합산
    scaler = MinMaxScaler(feature_range=(0, 100))

    def scale(col):
        return scaler.fit_transform(master_df[[col]]).flatten()

    master_df['식당_점수']      = scale('식당_인프라')     * 0.7 + scale('식당_2030비중')  * 0.3
    master_df['병원_약국_점수'] = scale('기본의료_인프라') * 0.7 + scale('상비약_편의점수') * 0.3
    master_df['카페_점수']      = scale('카페_인프라')
    master_df['세탁소_점수']    = scale('세탁소_인프라')
    master_df['마트_슈퍼_점수'] = scale('마트슈퍼_인프라')

    result = master_df[['식당_점수', '병원_약국_점수', '카페_점수',
                         '세탁소_점수', '마트_슈퍼_점수']].reset_index()
    result = result.rename(columns={'행정동_코드_명': 'dong'})
    return result

def get_safety_data():
    """안전 파트: 완성된 CSV 직접 로드"""
    df = pd.read_csv(BASE + 'safety.csv', encoding='utf-8-sig')
    result = (df[['행정동', '최종_안전_점수']]
              .rename(columns={'행정동': 'dong', '최종_안전_점수': '안전_점수'}))
    result['dong'] = result['dong'].replace('온천동', '은천동')
    return result


    

@st.cache_data
def load_data():
    """모든 파트 통합 및 정규화"""
    # 교통 (CSV 직접 읽기)
    df_transport = pd.read_csv(BASE + 'transport.csv',    encoding='utf-8-sig')
    df_snu       = pd.read_csv(BASE + 'snu.csv', encoding='utf-8-sig')

    transport = (df_transport[['행정동', '교통_종합점수']]
                 .rename(columns={'행정동': 'dong'}))
    snu       = (df_snu[['행정동', 'snu_접근성_종합']]
                 .rename(columns={'행정동': 'dong', 'snu_접근성_종합': '서울대접근성_점수'}))
    transport['dong'] = transport['dong'].replace('온천동', '은천동')
    snu['dong']       = snu['dong'].replace('온천동', '은천동')

    # 나머지 파트 (함수로 전처리)
    rent    = get_rent_data()
    amenity = get_amenity_data()
    safety  = get_safety_data()

    # 전체 병합
    df = (transport
          .merge(snu,     on='dong', how='left')
          .merge(amenity, on='dong', how='left')
          .merge(rent,    on='dong', how='left')
          .merge(safety,  on='dong', how='left'))

    # 정규화
    scaler = MinMaxScaler()
    df['주거비_점수']      = 1 - scaler.fit_transform(df[['환산월세']]).flatten()
    df['교통_점수']        = scaler.fit_transform(df[['교통_종합점수']]).flatten()
    df['서울대_점수']      = scaler.fit_transform(df[['서울대접근성_점수']]).flatten()
    df['안전_점수_정규화'] = scaler.fit_transform(df[['안전_점수']]).flatten()

    for col in ['식당_점수', '병원_약국_점수', '카페_점수', '세탁소_점수', '마트_슈퍼_점수']:
        df[col + '_정규화'] = scaler.fit_transform(df[[col]]).flatten()

    return df


# ════════════════════════════════════════════════════════════
# 투표 함수
# ════════════════════════════════════════════════════════════

def load_votes():
    if os.path.exists(VOTE_FILE):
        with open(VOTE_FILE, 'r') as f:
            return json.load(f)
    return {'방향1': 0, '방향2': 0, '방향3': 0, '방향4': 0}


def save_vote(choice):
    votes = load_votes()
    votes[choice] += 1
    with open(VOTE_FILE, 'w') as f:
        json.dump(votes, f)


# ════════════════════════════════════════════════════════════
# 데이터 로드
# ════════════════════════════════════════════════════════════

df = load_data()


# ════════════════════════════════════════════════════════════
# UI — 사이드바 슬라이더
# ════════════════════════════════════════════════════════════

st.title("🏠 관악구 자취 점수 산출기")
st.markdown("슬라이더로 항목별 중요도를 설정하면 맞춤형 동네 순위를 알려드려요.")

st.sidebar.header("⚙️ 가중치 설정")

st.sidebar.subheader("[ 주거비 ]")
s_주거비 = st.sidebar.slider('주거비', 0, 100, 50, 5)

st.sidebar.subheader("[ 교통 ]")
s_교통     = st.sidebar.slider('교통 (대분류)',  0, 100, 40, 5)
s_대중교통 = st.sidebar.slider('  ▸ 대중교통', 0, 100, 50, 5)
s_서울대   = st.sidebar.slider('  ▸ 서울대',   0, 100, 50, 5)

st.sidebar.subheader("[ 안전 ]")
s_안전 = st.sidebar.slider('안전', 0, 100, 50, 5)

st.sidebar.subheader("[ 편의시설 ]")
s_편의시설 = st.sidebar.slider('편의시설 (대분류)', 0, 100, 60, 5)
s_식당     = st.sidebar.slider('  ▸ 식당',          0, 100, 30, 5)
s_병원     = st.sidebar.slider('  ▸ 병원/약국',      0, 100, 20, 5)
s_카페     = st.sidebar.slider('  ▸ 카페',           0, 100, 15, 5)
s_세탁소   = st.sidebar.slider('  ▸ 세탁소',         0, 100, 15, 5)
s_마트     = st.sidebar.slider('  ▸ 마트',           0, 100, 20, 5)


# ════════════════════════════════════════════════════════════
# 가중치 계산
# ════════════════════════════════════════════════════════════

대분류     = {'주거비': s_주거비, '교통': s_교통, '안전': s_안전, '편의시설': s_편의시설}
대분류_합  = sum(대분류.values()) or 1
W          = {k: v / 대분류_합 for k, v in 대분류.items()}

교통_합    = s_대중교통 + s_서울대 or 1
W_대중교통 = s_대중교통 / 교통_합
W_서울대   = s_서울대   / 교통_합

편의_합  = s_식당 + s_병원 + s_카페 + s_세탁소 + s_마트 or 1
W_식당   = s_식당   / 편의_합
W_병원   = s_병원   / 편의_합
W_카페   = s_카페   / 편의_합
W_세탁소 = s_세탁소 / 편의_합
W_마트   = s_마트   / 편의_합

가중치맵 = {
    '주거비':    W['주거비'],
    '대중교통':  W['교통'] * W_대중교통,
    '서울대':    W['교통'] * W_서울대,
    '안전':      W['안전'],
    '식당':      W['편의시설'] * W_식당,
    '병원/약국': W['편의시설'] * W_병원,
    '카페':      W['편의시설'] * W_카페,
    '세탁소':    W['편의시설'] * W_세탁소,
    '마트':      W['편의시설'] * W_마트,
}


# ════════════════════════════════════════════════════════════
# 점수 계산
# ════════════════════════════════════════════════════════════

temp = df.copy()
temp['교통_합산'] = (
    W_대중교통 * temp['교통_점수'] + W_서울대 * temp['서울대_점수'])
temp['편의시설_합산'] = (
    W_식당   * temp['식당_점수_정규화'] +
    W_병원   * temp['병원_약국_점수_정규화'] +
    W_카페   * temp['카페_점수_정규화'] +
    W_세탁소 * temp['세탁소_점수_정규화'] +
    W_마트   * temp['마트_슈퍼_점수_정규화'])
temp['최종점수'] = (
    W['주거비']    * temp['주거비_점수'] +
    W['교통']      * temp['교통_합산'] +
    W['안전']      * temp['안전_점수_정규화'] +
    W['편의시설']  * temp['편의시설_합산'])

result = (temp[['dong', '최종점수']]
          .sort_values('최종점수', ascending=False)
          .reset_index(drop=True))
result.index += 1
result.columns = ['행정동', '자취점수']
result['자취점수'] = (result['자취점수'] * 100).round(1)
top3   = result.head(3)['행정동'].tolist()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']


# ════════════════════════════════════════════════════════════
# 결과 요약
# ════════════════════════════════════════════════════════════

st.subheader("🏆 상위 3개 추천 동네")
col1, col2, col3 = st.columns(3)
for col, (i, row_r) in zip([col1, col2, col3], result.head(3).iterrows()):
    col.metric(label=f"{i}위", value=row_r['행정동'], delta=f"{row_r['자취점수']}점")

st.caption(
    f"적용 가중치 — 주거비 {W['주거비']:.0%} | 교통 {W['교통']:.0%} "
    f"| 안전 {W['안전']:.0%} | 편의시설 {W['편의시설']:.0%}  |  "
    f"교통 세부: 대중교통 {W_대중교통:.0%} / 서울대 {W_서울대:.0%}  |  "
    f"편의시설 세부: 식당 {W_식당:.0%} / 병원 {W_병원:.0%} / "
    f"카페 {W_카페:.0%} / 세탁소 {W_세탁소:.0%} / 마트 {W_마트:.0%}"
)

with st.expander("전체 순위 보기"):
    st.dataframe(result, use_container_width=True)

st.divider()


# ════════════════════════════════════════════════════════════
# 방향 1 — 등급표 + 한줄 요약 (상위 3개)
# ════════════════════════════════════════════════════════════

st.subheader("방향 1 — 등급표 + 한줄 요약")
st.caption("⭐ = 내가 중요하게 설정한 상위 3개 항목")

def 점수to등급(v):
    if v >= 0.8: return 'A+'
    elif v >= 0.6: return 'A'
    elif v >= 0.4: return 'B+'
    elif v >= 0.2: return 'B'
    else: return 'C'

중요항목 = sorted(가중치맵, key=가중치맵.get, reverse=True)[:3]
헤더표시 = [c + '⭐' if c in 중요항목 else c for c in 카테고리]

표데이터1 = []
for _, row_r in result.head(3).iterrows():
    dong = row_r['행정동']
    row  = temp[temp['dong'] == dong].iloc[0]
    강점 = max(카테고리, key=lambda c: row[컬럼맵[c]])
    약점 = min(카테고리, key=lambda c: row[컬럼맵[c]])
    등급들 = [점수to등급(row[컬럼맵[c]]) for c in 카테고리]
    표데이터1.append(
        [dong, 점수to등급(row_r['자취점수'] / 100)] + 등급들
        + [f"{강점} 우수 · {약점} 부족"])

df1 = pd.DataFrame(표데이터1, columns=['행정동', '종합'] + 헤더표시 + ['요약'])
st.dataframe(df1.set_index('행정동'), use_container_width=True)

st.divider()


# ════════════════════════════════════════════════════════════
# 방향 2 — 레이더 차트 (상위 3개)
# ════════════════════════════════════════════════════════════

st.subheader("방향 2 — 레이더 차트")

N      = len(카테고리)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0.0]

fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
for dong, color in zip(top3, colors):
    row    = temp[temp['dong'] == dong].iloc[0]
    values = [row[컬럼맵[c]] for c in 카테고리] + [row[컬럼맵[카테고리[0]]]]
    ax2.plot(angles, values, 'o-', linewidth=2, color=color, label=dong)
    ax2.fill(angles, values, alpha=0.1, color=color)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(카테고리, fontsize=9)
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.divider()


# ════════════════════════════════════════════════════════════
# 방향 3 — 평균 대비 색깔 표 (상위 3개)
# ════════════════════════════════════════════════════════════

st.subheader("방향 3 — 평균 대비 색깔 표")
st.caption("🟢 초록 = 관악구 평균보다 높음  |  🔴 빨강 = 관악구 평균보다 낮음")

평균      = {c: temp[컬럼맵[c]].mean() for c in 카테고리}
표데이터3 = []
for dong in top3:
    row = temp[temp['dong'] == dong].iloc[0]
    표데이터3.append({'행정동': dong, **{c: round(row[컬럼맵[c]], 2) for c in 카테고리}})
df3 = pd.DataFrame(표데이터3).set_index('행정동')

def 색칠(col):
    styles = []
    for val in col:
        diff = val - 평균[col.name]
        if diff >= 0.15:    styles.append('background-color: #c8f7c5')
        elif diff >= 0.05:  styles.append('background-color: #e8f8e8')
        elif diff <= -0.15: styles.append('background-color: #f7c5c5')
        elif diff <= -0.05: styles.append('background-color: #f8e8e8')
        else:               styles.append('')
    return styles

st.dataframe(df3.style.apply(색칠, axis=0), use_container_width=True)

st.divider()


# ════════════════════════════════════════════════════════════
# 방향 4 — 가중치 강조 막대그래프 (상위 3개)
# ════════════════════════════════════════════════════════════

st.subheader("방향 4 — 가중치 강조 막대그래프")
st.caption("막대가 진할수록 내가 중요하게 설정한 항목")

max_w  = max(가중치맵.values()) or 1
fig4, axes4 = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
fig4.suptitle('상위 3개 동네 항목별 점수 (진할수록 높은 가중치)', fontsize=12)

for ax, dong, color in zip(axes4, top3, colors):
    row        = temp[temp['dong'] == dong].iloc[0]
    값         = [row[컬럼맵[c]] for c in 카테고리]
    bar_colors = [mcolors.to_rgba(color, alpha=0.3 + 0.7 * (가중치맵[c] / max_w))
                  for c in 카테고리]
    bars = ax.barh(카테고리, 값, color=bar_colors)
    ax.set_xlim(0, 1)
    ax.set_title(dong, fontsize=12, fontweight='bold', color=color)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.4)
    for bar, val in zip(bars, 값):
        ax.text(min(val + 0.02, 0.92), bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=8)

plt.tight_layout()
st.pyplot(fig4)
plt.close(fig4)

st.divider()


# ════════════════════════════════════════════════════════════
# 투표
# ════════════════════════════════════════════════════════════


