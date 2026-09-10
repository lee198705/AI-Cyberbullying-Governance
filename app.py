from pathlib import Path
from html import escape

import streamlit as st

from src.moderation import monitor_comments
from src.preprocess import read_csv
from src.recommender import recommend_articles
from src.sentiment_predictor import get_sentiment_predictor
from src.toxicity_predictor import get_predictor, predict_toxicity
from src.user_profile import build_user_profiles


ROOT = Path(__file__).resolve().parent


def h(value) -> str:
    return escape(str(value), quote=True)


st.set_page_config(page_title="AI 网络暴力治理平台", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f7f8f5; color: #17211b; }
    [data-testid="stSidebar"] { background: #eef2eb; }
    [data-testid="stMetric"] {
      background: #ffffff;
      border: 1px solid #dfe6dd;
      border-radius: 8px;
      padding: 12px;
    }
    .comment-card {
      background: #ffffff;
      border: 1px solid #dfe6dd;
      border-radius: 8px;
      padding: 14px 16px;
      margin: 10px 0;
    }
    .risk-high { color: #b42318; font-weight: 700; }
    .risk-low { color: #20734d; font-weight: 700; }
    .muted { color: #66736b; font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return {
        "comments": read_csv(ROOT / "data" / "comments.csv"),
        "users": read_csv(ROOT / "data" / "users.csv"),
        "articles": read_csv(ROOT / "data" / "articles.csv"),
        "interactions": read_csv(ROOT / "data" / "interactions.csv"),
    }


data = load_data()
profiles = build_user_profiles(data["users"], data["comments"], data["interactions"], data["articles"])
report = monitor_comments(data["comments"], profiles)
predictor = get_predictor()
sentiment_predictor = get_sentiment_predictor()

with st.sidebar:
    st.title("AI 网络暴力治理")
    st.caption(f"毒性模型：{predictor.model_name}")
    if predictor.load_error:
        st.warning(f"BiLSTM checkpoint 加载失败：{predictor.load_error}")
    st.caption(f"情感模型：{sentiment_predictor.model_name}")
    if sentiment_predictor.load_error:
        st.warning(f"情感 BiLSTM checkpoint 加载失败：{sentiment_predictor.load_error}")
    selected_user = st.selectbox("用户画像", [u["user_id"] for u in data["users"]])
    top_n = st.slider("推荐数量", min_value=2, max_value=5, value=3)
    st.divider()
    custom_text = st.text_area("实时评论检测", "你这个观点不完整，建议补充数据来源。")
    result = predict_toxicity(custom_text)
    st.metric("攻击风险", f"{result.probability:.0%}")
    st.write(result.action)

st.title("AI 网络暴力治理平台")

c1, c2, c3, c4 = st.columns(4)
c1.metric("今日评论量", report["total"])
c2.metric("风险评论数", report["risk_count"])
c3.metric("阻断候选", report["blocked_count"])
c4.metric("平均风险", f"{report['avg_toxicity']:.0%}")

tab_monitor, tab_rank, tab_profile, tab_recommend = st.tabs(["舆情监测", "评论审核", "用户画像", "内容推荐"])

with tab_monitor:
    st.subheader("风险评论流")
    for row in sorted(report["comments"], key=lambda item: item["toxicity"], reverse=True):
        klass = "risk-high" if row["toxicity"] >= 0.6 else "risk-low"
        st.markdown(
            f"""
            <div class="comment-card">
              <div>{h(row["text"])}</div>
              <div class="muted">用户 {h(row["user_id"])} · 文章 {h(row["article_id"])}</div>
              <div class="{klass}">风险 {row["toxicity"]:.0%} · {h(row["risk_level"])} · {h(row["moderation_action"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_rank:
    st.subheader("高赞理性评论排序")
    ranked = report["ranked_comments"]
    for row in ranked:
        st.markdown(
            f"""
            <div class="comment-card">
              <strong>{row["rank_score"]:.3f}</strong> · {h(row["text"])}
              <div class="muted">点赞 {row["likes"]} · 理性 {row["rationality_score"]:.2f} · 画像加成 {row["profile_bonus"]:.2f} · 质量 {row["quality_score"]:.2f} · 风险 {row["toxicity"]:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_profile:
    profile = profiles[selected_user]
    st.subheader(f"{profile['name']} · {profile['global_user_type']}")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("攻击风险", f"{profile['toxicity_score']:.0%}")
    p2.metric("情绪得分", f"{profile['sentiment_score']:.2f}")
    p3.metric("活跃度", f"{profile['activity_score']:.0%}")
    p4.metric("理性程度", f"{profile['rationality_score']:.0%}")
    st.write("兴趣：", " / ".join(profile["topic_preferences"]))
    st.write("话题级 A1 / B / A2：")
    st.dataframe(
        [
            {"话题": topic, "倾向标签": values["label"], "倾向分": values["score"], "置信度": values["confidence"]}
            for topic, values in profile["topic_stances"].items()
        ],
        hide_index=True,
        width="stretch",
    )

with tab_recommend:
    st.subheader("低风险多样化内容推荐")
    recs = recommend_articles(selected_user, data["articles"], data["interactions"], data["comments"], top_n=top_n)
    for row in recs:
        st.markdown(
            f"""
            <div class="comment-card">
              <strong>{h(row["title"])}</strong>
              <div class="muted">{h(row["category"])} · {h(row["tags"])}</div>
              <div>推荐分 {row["final_score"]:.3f} · 内容风险 {row["content_risk"]:.0%} · 讨论风险 {row["discussion_risk"]:.0%} · {h(row["reason"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
