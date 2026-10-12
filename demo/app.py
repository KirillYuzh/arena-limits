import plotly.express as px
import streamlit as st

from arena_limits import config
from arena_limits.storage import Storage

st.set_page_config(layout="wide", page_title="arena-limits")
st.title("arena-limits: риск A2A-сделок")

storage = Storage(config.DB_PATH)
matches = storage.load_matches()
profiles = storage.load_profiles()

if matches.empty:
    st.warning("В базе нет прогонов: сначала запусти experiments/run_arena.py")
    st.stop()

compliance = float(matches["protocol_ok"].mean())
breach_rate = float(matches["breach"].mean())

left, right = st.columns(2)
with left:
    st.metric("Прогонов", len(matches))
    st.metric("Доля пробоев", f"{breach_rate:.1%}")
    st.metric("Соблюдение протокола", f"{compliance:.1%}")
with right:
    outcomes = matches["outcome"].value_counts().rename_axis("исход").reset_index(name="n")
    st.plotly_chart(
        px.bar(outcomes, x="исход", y="n", title="Исходы сделок"),
        use_container_width=True,
    )

st.subheader("Профили агентов")
st.dataframe(
    profiles[["agent_id", "backbone", "p_hat", "a_i"]].sort_values("p_hat"),
    use_container_width=True,
)

st.subheader("Доля пробоев по бэкбонам")
by_backbone = matches.groupby("backbone")["breach"].mean().rename("доля пробоев").reset_index()
st.plotly_chart(
    px.bar(by_backbone, x="backbone", y="доля пробоев", range_y=[0, 1]),
    use_container_width=True,
)

st.subheader("Корреляции агентов (монокультура)")
corr = storage.load_corr()
if not corr.empty:
    agents = sorted(set(corr["agent_i"]) | set(corr["agent_j"]))
    matrix = corr.pivot(index="agent_i", columns="agent_j", values="rho")
    st.plotly_chart(
        px.imshow(matrix, color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
        use_container_width=True,
    )

st.subheader("E-процесс по кластерам")
eprocess = storage.load_eprocess()
if eprocess.empty:
    st.info("нет наблюдений e-процесса")
else:
    st.dataframe(eprocess, use_container_width=True)
