import streamlit as st
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple, Callable
from collections import Counter
import copy

# --- 설정 ---
AVAILABLE_PIECE_LENGTHS_MASTER = [1829, 1524, 1219, 914, 610, 305] # 마스터 부재 길이 목록 (큰 값부터 정렬)
# 지정된 부재 길이에 대한 고정 색상 매핑
PIECE_COLOR_MAP_DEFAULT = {
    1829: '#1f77b4', # 파랑
    1524: '#ff7f0e', # 주황
    1219: '#2ca02c', # 초록
    914:  '#d62728', # 빨강
    610:  '#9467bd', # 보라
    305:  '#8c564b', # 갈색
}
PLOTLY_COLORS_FALLBACK = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'] # 대체 색상 팔레트

MARGIN_COLOR = "lightgrey" # 여백 색상
ERROR_COLOR = "rgba(255, 100, 100, 0.3)" # 오류 표시 배경색 (부드러운 빨강)

# --- 최적화 함수 ---
def optimize_dp_max_fill_large_priority(target_capacity: int, piece_types: List[int]) -> Tuple[int, List[int]]:
    """
    전략 2: 여백 최소화 초점 (가용 공간 최대 활용)
    - 부재 길이 합을 최대화합니다. (Internal Waste 최소화)
    - piece_types 리스트가 내림차순으로 정렬되어 있으면 큰 부재를 우선적으로 고려하는 경향이 있습니다.
    """
    if target_capacity <= 0 or not piece_types:
        return 0, []
    sorted_piece_types = sorted(list(set(piece_types)), reverse=True)

    dp_value = [0] * (target_capacity + 1)
    dp_combination = [[] for _ in range(target_capacity + 1)]

    for piece_len in sorted_piece_types:
        for i in range(piece_len, target_capacity + 1):
            if dp_value[i - piece_len] + piece_len > dp_value[i]:
                dp_value[i] = dp_value[i - piece_len] + piece_len
                dp_combination[i] = dp_combination[i - piece_len] + [piece_len]

    return dp_value[target_capacity], sorted(dp_combination[target_capacity], reverse=True)

def optimize_dp_max_fill_min_pieces(target_capacity: int, piece_types: List[int]) -> Tuple[int, List[int]]:
    """
    전략 1: AI 추천 최적 (남은 공간 최소화 후, 사용 부재 수 최소화)
    - 부재 길이 합을 최대화하고, 그 다음으로 부재 수를 최소화합니다.
    """
    if target_capacity <= 0 or not piece_types:
        return 0, []
    dp_state = [(0, float('inf'), []) for _ in range(target_capacity + 1)]
    dp_state[0] = (0, 0, [])

    sorted_piece_types = sorted(list(set(piece_types)), reverse=True)

    for i in range(1, target_capacity + 1):
        for piece_len in sorted_piece_types:
            if i >= piece_len:
                prev_sum, prev_num_pieces, prev_combo = dp_state[i - piece_len]
                current_sum_candidate = prev_sum + piece_len
                current_num_pieces_candidate = prev_num_pieces + 1

                if current_sum_candidate > dp_state[i][0]:
                    dp_state[i] = (current_sum_candidate, current_num_pieces_candidate, prev_combo + [piece_len])
                elif current_sum_candidate == dp_state[i][0]:
                    if current_num_pieces_candidate < dp_state[i][1]:
                        dp_state[i] = (current_sum_candidate, current_num_pieces_candidate, prev_combo + [piece_len])

    final_sum, _, final_combination = dp_state[target_capacity]
    return final_sum, sorted(final_combination, reverse=True)

def optimize_dp_max_fill_max_pieces(target_capacity: int, piece_types: List[int]) -> Tuple[int, List[int]]:
    """
    전략 4: 부재 수 최대화 지향 (작은 부재 적극 활용)
    - 부재 길이 합을 최대화하고, 그 다음으로 부재 수를 최대화합니다.
    """
    if target_capacity <= 0 or not piece_types:
        return 0, []
    dp_state = [(0, 0, []) for _ in range(target_capacity + 1)]

    sorted_piece_types_asc = sorted(list(set(piece_types)))

    for i in range(1, target_capacity + 1):
        for piece_len in sorted_piece_types_asc:
            if i >= piece_len:
                prev_sum, prev_num_pieces, prev_combo = dp_state[i - piece_len]
                current_sum_candidate = prev_sum + piece_len
                current_num_pieces_candidate = prev_num_pieces + 1

                if current_sum_candidate > dp_state[i][0]:
                    dp_state[i] = (current_sum_candidate, current_num_pieces_candidate, prev_combo + [piece_len])
                elif current_sum_candidate == dp_state[i][0]:
                    if current_num_pieces_candidate > dp_state[i][1]:
                        dp_state[i] = (current_sum_candidate, current_num_pieces_candidate, prev_combo + [piece_len])

    final_sum, _, final_combination = dp_state[target_capacity]
    return final_sum, sorted(final_combination, reverse=True)

def optimize_greedy_largest_first(target_capacity: int, piece_types: List[int]) -> Tuple[int, List[int]]:
    """
    전략 3: 부재 수 최소화 지향 (큰 부재 적극 활용)
    - 사용 가능한 가장 큰 부재부터 차례대로 채워 넣습니다. (그리디 방식)
    """
    if target_capacity <= 0 or not piece_types:
        return 0, []

    selected_pieces = []
    current_sum = 0
    remaining_capacity = target_capacity

    sorted_piece_types = sorted(list(set(piece_types)), reverse=True)

    for piece_len in sorted_piece_types:
        while remaining_capacity >= piece_len:
            selected_pieces.append(piece_len)
            current_sum += piece_len
            remaining_capacity -= piece_len
    return current_sum, sorted(selected_pieces, reverse=True)

# --- 레이아웃 계산 ---
def calculate_single_strategy_layout(
    strategy_name: str,
    optimization_func: Callable,
    total_length: float,
    user_selected_piece_types: List[int],
    base_min_end_margin: float,
    input_alpha_for_margin: float,
    internal_alpha_distribution_method: str
) -> Dict[str, Any]:
    """단일 최적화 전략에 대한 레이아웃을 계산합니다."""
    results: Dict[str, Any] = {
        "strategy_name": strategy_name, "status": "오류", "message": "", "plot_elements": [], "summary": {},
        "internal_alpha_waste": 0.0, "final_left_margin": 0.0, "final_right_margin": 0.0,
        "selected_pieces_combination": []
    }
    results["summary"]["전략명"] = strategy_name

    if not user_selected_piece_types:
        results["message"] = "선택된 부재가 없어 배치를 계산할 수 없습니다. (양 끝 여백만 적용됩니다)"
        pass

    current_min_end_margin = base_min_end_margin + input_alpha_for_margin
    usable_space_for_pieces = total_length - (2 * current_min_end_margin)
    results["summary"]["사용자 지정 양 끝 여백 (각각)"] = f"{current_min_end_margin:,.0f}"
    results["summary"]["가용 공간 (부재 배치용)"] = f"{usable_space_for_pieces:,.1f}"

    if usable_space_for_pieces < 0:
        results["message"] = f"배치 오류: 사용자 지정 양 끝 여백의 합({2 * current_min_end_margin:,.0f})이 전체 길이({total_length:,.0f})를 초과합니다."
        plot_elements = [
            {'label': '요구된 좌측 여백', 'start': 0, 'end': current_min_end_margin, 'length': current_min_end_margin, 'type': 'margin', 'color': ERROR_COLOR},
            {'label': '요구된 우측 여백', 'start': total_length - current_min_end_margin, 'end': total_length, 'length': current_min_end_margin, 'type': 'margin', 'color': ERROR_COLOR},
            {'label': '전체 길이 한계', 'start': 0, 'end': total_length, 'length': total_length, 'type': 'limit_line', 'color': 'red'}
        ]
        results["plot_elements"] = plot_elements
        results["status"] = "오류"
        return results

    sum_selected_pieces, selected_pieces_combination = optimization_func(int(usable_space_for_pieces), user_selected_piece_types)

    results["selected_pieces_combination"] = selected_pieces_combination
    results["summary"]["선택된 부재들의 총 길이"] = f"{sum_selected_pieces:,.0f}"

    internal_alpha_waste = usable_space_for_pieces - sum_selected_pieces
    results["internal_alpha_waste"] = internal_alpha_waste
    results["summary"]["가용 공간 내 남은 공간"] = f"{internal_alpha_waste:,.1f}"

    final_left_margin = current_min_end_margin
    final_right_margin = current_min_end_margin

    if internal_alpha_distribution_method == "균등 분배 (양 끝단)":
        final_left_margin += internal_alpha_waste / 2.0
        final_right_margin += internal_alpha_waste / 2.0
    elif internal_alpha_distribution_method == "없음 (최소 여백만)":
        final_right_margin += internal_alpha_waste

    final_left_margin = max(0, final_left_margin)
    final_right_margin = max(0, final_right_margin)

    current_total_calculated = final_left_margin + sum_selected_pieces + final_right_margin
    if abs(current_total_calculated - total_length) > 1e-9:
        diff = total_length - current_total_calculated
        final_right_margin += diff
        final_right_margin = max(0, final_right_margin)

    results["final_left_margin"] = final_left_margin
    results["final_right_margin"] = final_right_margin

    plot_elements = []
    current_pos = 0.0

    active_color_map = {}
    color_idx = 0
    for piece_val in AVAILABLE_PIECE_LENGTHS_MASTER:
        if piece_val in PIECE_COLOR_MAP_DEFAULT:
            active_color_map[piece_val] = PIECE_COLOR_MAP_DEFAULT[piece_val]
        else:
            active_color_map[piece_val] = PLOTLY_COLORS_FALLBACK[color_idx % len(PLOTLY_COLORS_FALLBACK)]
            color_idx +=1

    plot_elements.append({'label': '좌측 여백', 'start': current_pos, 'end': current_pos + final_left_margin, 'length': final_left_margin, 'type': 'margin', 'color': MARGIN_COLOR})
    current_pos += final_left_margin

    for p_len in selected_pieces_combination:
        plot_elements.append({'label': f'부재 ({p_len})', 'start': current_pos, 'end': current_pos + p_len, 'length': p_len, 'type': 'piece', 'color': active_color_map.get(p_len, 'grey')})
        current_pos += p_len

    plot_elements.append({'label': '우측 여백', 'start': current_pos, 'end': total_length, 'length': final_right_margin, 'type': 'margin', 'color': MARGIN_COLOR})

    results["status"] = "성공"
    results["plot_elements"] = plot_elements
    results["summary"]["최종 좌측 여백"] = f"{final_left_margin:,.1f}"
    results["summary"]["최종 우측 여백"] = f"{final_right_margin:,.1f}"
    results["summary"]["배치된 총 부재 개수"] = len(selected_pieces_combination)

    total_unused_space = total_length - sum_selected_pieces
    results["summary"]["총 미사용 공간"] = f"{total_unused_space:,.1f}"
    return results

# --- 시각화 ---
def create_plotly_visualization(total_length: float, plot_elements: List[Dict[str, Any]], strategy_title: str, strategy_summary_dict: Dict[str, Any], selected_piece_types_for_legend: List[int]) -> go.Figure:
    """레이아웃을 시각화하기 위한 Plotly Figure를 생성합니다."""
    fig = go.Figure()
    annotations = []

    y_level = 0.5
    text_size = 18
    min_length_for_text = total_length * 0.01

    fig.add_shape(type="rect", x0=0, y0=0, x1=total_length, y1=1,
                line=dict(color="black", width=3), fillcolor="white", layer="below")

    active_color_map = {}
    color_idx = 0
    for piece_val in AVAILABLE_PIECE_LENGTHS_MASTER:
        if piece_val in PIECE_COLOR_MAP_DEFAULT:
            active_color_map[piece_val] = PIECE_COLOR_MAP_DEFAULT[piece_val]
        else:
            active_color_map[piece_val] = PLOTLY_COLORS_FALLBACK[color_idx % len(PLOTLY_COLORS_FALLBACK)]
            color_idx += 1

    for el in plot_elements:
        if el['type'] == 'limit_line':
            fig.add_shape(type="line", x0=el['start'], y0=-0.1, x1=el['start'], y1=1.1,
                          line=dict(color=el['color'], width=2, dash="dash"), name=el['label'])
            annotations.append(dict(x=el['start'], y=1.15, text=el['label'], showarrow=False,
                                    font=dict(color=el['color'], size=text_size, family="Arial, sans-serif"), xanchor='center'))
            continue

        fig.add_shape(type="rect", x0=el['start'], y0=0, x1=el['end'], y1=1,
                      fillcolor=el['color'], line=dict(color="black", width=3), name=el['label'])

        if el['length'] > min_length_for_text or (el['type'] == 'margin' and el['length'] > 0.01):
            text_color = "white" if el['color'] not in [MARGIN_COLOR, ERROR_COLOR, "yellow", "lightyellow", "lightcyan", "white"] else "black"
            anno_text = f"{el['length']:,.0f}"

            if el['type'] == 'piece' and '(' in el['label'] and ')' in el['label']:
                raw_len_str = el['label'][el['label'].find("(")+1:el['label'].find(")")]
                try:
                    anno_text = f"{int(raw_len_str):,.0f}"
                except ValueError:
                    anno_text = raw_len_str

            annotations.append(dict(x=(el['start'] + el['end']) / 2, y=y_level, text=anno_text,
                                    showarrow=False, font=dict(color=text_color, size=text_size, family="Arial Black, sans-serif"),
                                    align="center"))

    internal_waste_val_str = f"{float(strategy_summary_dict.get('가용 공간 내 남은 공간', '0').replace(',','')):.1f}" if isinstance(strategy_summary_dict.get('가용 공간 내 남은 공간'), str) else f"{strategy_summary_dict.get('가용 공간 내 남은 공간', 0.0):,.1f}"
    num_pieces_val_str = str(strategy_summary_dict.get('배치된 총 부재 개수', "N/A"))

    summary_info_html = f"<span style='font-size: 16px; color: #333;'>남은 공간(가용): <b style='color:#c0392b;'>{internal_waste_val_str} mm</b> | 총 부재: <b style='color:#2980b9;'>{num_pieces_val_str} 개</b></span>"
    title_with_summary = f"<b style='font-size: 26px;'>{strategy_title}</b><br>{summary_info_html}"

    fig.update_layout(
        xaxis=dict(range=[0, total_length], showgrid=False, zeroline=False, title_text="전체 길이 (mm)", tickformat=",,.0f",
                   titlefont=dict(size=18, family="Arial Black, sans-serif"), tickfont=dict(size=16, family="Arial, sans-serif")),
        yaxis=dict(range=[-0.2, 1.2], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        annotations=annotations, height=380,
        margin=dict(l=20, r=20, t=130, b=60),
        title_text=title_with_summary, title_x=0.5, titlefont=dict(size=26, family="Arial Black, sans-serif"),
        plot_bgcolor='white', showlegend=True,
        legend=dict(font=dict(size=14, family="Arial, sans-serif"), itemsizing='constant', orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1)
    )

    legend_items_added = set()
    piece_counts_in_current_layout = Counter(el['length'] for el in plot_elements if el['type'] == 'piece')

    for piece_len_type in sorted(selected_piece_types_for_legend, reverse=True):
        if piece_counts_in_current_layout[piece_len_type] > 0:
            color = active_color_map.get(piece_len_type, 'grey')
            legend_name_key = (color, piece_len_type)
            if legend_name_key not in legend_items_added:
                count = piece_counts_in_current_layout[piece_len_type]
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                         marker=dict(size=16, color=color, line=dict(color='black', width=3)),
                                         name=f"부재: {piece_len_type:,.0f} (x{count})"))
                legend_items_added.add(legend_name_key)

    if any(el['type'] == 'margin' for el in plot_elements):
        if 'margin' not in legend_items_added:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                     marker=dict(size=16, color=MARGIN_COLOR, line=dict(color='black', width=3)),
                                     name="여백 공간"))
            legend_items_added.add('margin')

    # Add tooltips
    for el in plot_elements:
        fig.add_trace(go.Scatter(
            x=[(el['start'] + el['end']) / 2],
            y=[0.5],
            text=[f"{el['label']}: {el['length']:,.0f} mm"],
            mode='text',
            showlegend=False,
            hoverinfo='text'
        ))

    return fig

# --- Streamlit 앱 UI ---
st.set_page_config(layout="wide", page_title="길이 최적화 V5.5 (스타일 혁신)")

# 사이드바 UI 구성
with st.sidebar:
    st.header("⚙️ 입력 파라미터")
    total_length_input = st.number_input("전체 배치 길이 (L):", min_value=1.0, value=9500.0, step=100.0, format="%.0f", help="단위: mm")

    st.markdown("---")
    st.markdown("**사용할 부재 길이 선택 (mm):**")
    selected_piece_types_from_user = []

    active_color_map_sidebar = {}
    color_idx_sidebar = 0
    for piece_val_master_sb in AVAILABLE_PIECE_LENGTHS_MASTER:
        if piece_val_master_sb in PIECE_COLOR_MAP_DEFAULT:
            active_color_map_sidebar[piece_val_master_sb] = PIECE_COLOR_MAP_DEFAULT[piece_val_master_sb]
        else:
            active_color_map_sidebar[piece_val_master_sb] = PLOTLY_COLORS_FALLBACK[color_idx_sidebar % len(PLOTLY_COLORS_FALLBACK)]
            color_idx_sidebar +=1

    for piece_len in AVAILABLE_PIECE_LENGTHS_MASTER:
        col1, col2 = st.columns([2, 5])
        with col1:
            color = active_color_map_sidebar.get(piece_len, 'grey')
            st.markdown(f'<div style="width:22px; height:22px; background-color:{color}; border:2px solid black; margin-top:8px; margin-left:2px; border-radius: 4px;"></div>', unsafe_allow_html=True)
        with col2:
            default_checked = False if piece_len == 305 else True  # 🔹 305이면 기본 체크 해제
            if st.checkbox(f"{piece_len:,.0f} mm", value=default_checked, key=f"piece_cb_{piece_len}"):
                selected_piece_types_from_user.append(piece_len)

    if not selected_piece_types_from_user:
        st.warning("최적화를 위해 하나 이상의 부재 길이를 선택해주세요.")
    st.markdown("---")

    base_end_margin = 300.0
    input_alpha_for_margin_val = st.slider(
        "양쪽 여백 추가값 (alpha):",
        min_value=0.0, max_value=100.0, value=0.0, step=10.0, format="%.0f mm",
        help=f"최종 양 끝 여백은 각각 '{base_end_margin:.0f}mm + 선택된 alpha 값'이 됩니다."
    )

    st.markdown(f"*실제 양 끝 여백 (각각): **{base_end_margin + input_alpha_for_margin_val:,.0f} mm***")
    st.markdown("---")

    internal_alpha_distribution_options = {
        "균등 분배 (양 끝단)": "가용 공간 내 남은 공간을 최종 양 끝 여백에 균등하게 추가합니다.",
        "없음 (최소 여백만)": "최종 양 끝 여백은 '300mm + alpha'로 유지하고, 가용 공간 내 남은 공간은 우측 여백에 추가됩니다."
    }
    selected_internal_alpha_dist_label = st.selectbox(
        "가용 공간 내 남은 공간 분배 방식:",
        options=list(internal_alpha_distribution_options.keys()),
        index=0
    )
    st.caption(internal_alpha_distribution_options[selected_internal_alpha_dist_label])
    st.markdown("---")

    # run_button = st.button("🚀 4가지 전략 실행", use_container_width=True, type="primary")

# 새로운 전략 정의
strategies_config = [
    {"name": "AI 추천 최적 (남은 공간 최소화 후, 사용 부재 수 최소화)", "func": optimize_dp_max_fill_min_pieces}, # 전략 1
    {"name": "여백 최소화 초점 (가용 공간 최대 활용)", "func": optimize_dp_max_fill_large_priority}, # 전략 2
    {"name": "부재 수 최소화 초점 (큰 부재 적극 활용)", "func": optimize_greedy_largest_first}, # 전략 3
    {"name": "부재 수 최대화 초점 (작은 부재 적극 활용)", "func": optimize_dp_max_fill_max_pieces} # 전략 4
]

# 메인 패널 실행 로직
# if run_button:
if not selected_piece_types_from_user:
    st.error("오류: 최적화를 진행하려면 사이드바에서 하나 이상의 '사용할 부재 길이'를 선택해야 합니다.")
else:
    st.markdown("## 📊 부재 배치 최적화 (4가지 전략)")

    all_strategy_results_data = []
    for strategy_conf in strategies_config:
        layout_result = calculate_single_strategy_layout(
            strategy_name=strategy_conf["name"],
            optimization_func=strategy_conf["func"],
            total_length=total_length_input,
            user_selected_piece_types=copy.deepcopy(selected_piece_types_from_user),
            base_min_end_margin=base_end_margin,
            input_alpha_for_margin=input_alpha_for_margin_val,
            internal_alpha_distribution_method=selected_internal_alpha_dist_label
        )
        all_strategy_results_data.append(layout_result)

    # 전략 비교 요약 정보 (스타일 개선 - 카드 디자인)
    # st.subheader("💡 전략 핵심 지표 비교")
    summary_cols = st.columns(len(all_strategy_results_data))
    icons = ["🧠", "🎯", "🔩", "🧩"] # 각 전략에 대한 아이콘 (전략 순서에 맞게)
    for idx, res_sum_data in enumerate(all_strategy_results_data):
        with summary_cols[idx]:
            strategy_display_name = res_sum_data['strategy_name']

            internal_waste_val = res_sum_data.get('internal_alpha_waste',0.0)
            num_pieces_val = res_sum_data['summary'].get('배치된 총 부재 개수',0)

            card_html = f"""
            <div style="background-color: #ffffff; border: 2px solid #e0e0e0; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px; height: 200px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
                <p style="font-weight: bold; font-size: 1.25em; margin-bottom: 15px; color: #1a237e;">{icons[idx]} {strategy_display_name.split(':')[0]}</p>
                <div>
                    <p style="font-size: 1em; margin-bottom: 5px; color: #37474f;">남은 공간 (가용):</p>
                    <p style="font-size: 1.5em; font-weight: bold; margin-bottom: 12px; color: #c62828;">{internal_waste_val:,.1f} mm</p>
                </div>
                <div>
                    <p style="font-size: 1em; margin-bottom: 5px; color: #37474f;">사용 부재 수:</p>
                    <p style="font-size: 1.5em; font-weight: bold; margin-bottom: 0; color: #00796b;">{num_pieces_val} 개</p>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
    st.markdown("---")

    cols = st.columns(2)
    chart_idx = 0
    for i in range(2):
        for j in range(2):
            if chart_idx < len(all_strategy_results_data):
                res = all_strategy_results_data[chart_idx]
                with cols[j]:
                    if res["status"] == "오류":
                        st.error(f"**{res['strategy_name']}**: {res['message']}")
                        if res["plot_elements"]:
                            fig = create_plotly_visualization(total_length_input, res["plot_elements"], res["strategy_name"], res["summary"], selected_piece_types_from_user)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        if res["plot_elements"]:
                            fig = create_plotly_visualization(total_length_input, res["plot_elements"], res["strategy_name"], res["summary"], selected_piece_types_from_user)
                            st.plotly_chart(fig, use_container_width=True)

                        st.markdown(f"##### {res['strategy_name']} 상세 결과:")
                        summary_text_list = [
                            f"- **선택 부재 총 길이:** {res['summary'].get('선택된 부재들의 총 길이', '0')} mm",
                            f"- **총 사용 부재 개수:** {res['summary'].get('배치된 총 부재 개수', 0)} 개",
                            f"- **가용 공간 내 남은 공간:** {res.get('internal_alpha_waste', 0):,.1f} mm",
                            f"- **최종 좌측 여백:** {res.get('final_left_margin', 0):,.1f} mm",
                            f"- **최종 우측 여백:** {res.get('final_right_margin', 0):,.1f} mm",
                            f"- **총 미사용 공간:** {res['summary'].get('총 미사용 공간', '0.0')} mm"
                        ]
                        st.markdown("\n".join(summary_text_list))

                    if res["status"] == "성공" and res["selected_pieces_combination"]:
                        piece_counts = Counter(res["selected_pieces_combination"])
                        detail_parts = [f"{piece:,.0f}mm × {count}" for piece, count in sorted(piece_counts.items(), reverse=True)]
                        st.markdown(f"**사용 부재 상세:** {', '.join(detail_parts)}")
                    elif res["status"] == "성공" and not res["selected_pieces_combination"]:
                            st.markdown("**사용 부재 상세:** 없음")
                    st.markdown("<br>", unsafe_allow_html=True)
                chart_idx += 1

        st.markdown("---")
        st.subheader("🎯 최적 추천")
        successful_results = [r for r in all_strategy_results_data if r['status'] == '성공' and (r['selected_pieces_combination'] or not selected_piece_types_from_user)]

        if successful_results:
            best_for_min_internal_waste = min(successful_results, key=lambda x: x.get('internal_alpha_waste', float('inf')))

            min_waste_value = best_for_min_internal_waste.get('internal_alpha_waste', float('inf'))
            similar_waste_strategies = [
                r for r in successful_results
                if r.get('internal_alpha_waste', float('inf')) <= min_waste_value + 1.0
            ]
            if not similar_waste_strategies : similar_waste_strategies = [best_for_min_internal_waste]

            best_for_min_pieces = min(similar_waste_strategies, key=lambda x: x['summary'].get('배치된 총 부재 개수', float('inf')))
            best_for_max_pieces = max(similar_waste_strategies, key=lambda x: x['summary'].get('배치된 총 부재 개수', 0))

            rec_html = f"""
            <div style="background-color: #f0f8ff; border-left: 6px solid #007bff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h4 style="color: #0056b3; margin-top:0; font-family: 'Arial Black', sans-serif; font-size: 1.5em;">🏆 추천 전략 가이드</h4>
                <p style="font-size: 1.15em; margin-bottom: 12px; line-height: 1.6;">
                    <span style="font-size: 1.3em;">🗑️</span> <strong>가장 적은 내부 공간 낭비:</strong> <strong style="color:#0056b3;">'{best_for_min_internal_waste['strategy_name']}'</strong>
                    <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (남은 공간: <strong>{best_for_min_internal_waste.get('internal_alpha_waste',0):,.1f} mm</strong>)
                </p>
                <p style="font-size: 1.15em; margin-bottom: 12px; line-height: 1.6;">
                    <span style="font-size: 1.3em;">🧩</span> <strong>가장 적은 부재 사용 (효율적):</strong> <strong style="color:#0056b3;">'{best_for_min_pieces['strategy_name']}'</strong>
                    <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ({best_for_min_pieces['summary'].get('배치된 총 부재 개수',0)} 개, 남은 공간: {best_for_min_pieces.get('internal_alpha_waste',0):,.1f} mm)
                </p>
                <p style="font-size: 1.15em; margin-bottom: 0; line-height: 1.6;">
                    <span style="font-size: 1.3em;">🎲</span> <strong>가장 많은 부재 사용 (다양한 활용):</strong> <strong style="color:#0056b3;">'{best_for_max_pieces['strategy_name']}'</strong>
                    <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ({best_for_max_pieces['summary'].get('배치된 총 부재 개수',0)} 개, 남은 공간: {best_for_max_pieces.get('internal_alpha_waste',0):,.1f} mm)
                </p>
            </div>
            """
            st.markdown(rec_html, unsafe_allow_html=True)
        else:
            st.warning("성공적인 배치 결과를 찾을 수 없어 추천을 제공할 수 없습니다.")

# else:
#     st.info("좌측 사이드바에서 파라미터를 설정하고 '4가지 전략 실행' 버튼을 클릭하세요.")
