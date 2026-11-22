"""Streamlit-приложение: стрелки, линии тока и LIC.  Запуск: streamlit run app.py"""
import io
import streamlit as st
from render import render, FIELDS

st.set_page_config(page_title="Визуализация векторных полей", layout="centered")
st.title("Визуализация двумерных векторных полей")

field = st.sidebar.selectbox("Поле", list(FIELDS), format_func=lambda k: FIELDS[k])
mode = st.sidebar.radio("Режим", ["Стрелки", "Линии тока", "LIC"], index=2)

size, length_px, h_px, overlay = 256, 20, 0.5, True
if mode == "LIC":
    size = st.sidebar.select_slider("Разрешение изображения, px", options=[128, 256, 512], value=256)
    length_px = st.sidebar.slider("Длина свёртки в каждую сторону, px", 5, 60, 20)
    h_px = st.sidebar.select_slider("Шаг интегрирования, px", options=[1.0, 0.5, 0.25], value=0.5)
    overlay = st.sidebar.checkbox("Стрелки поверх LIC (показывают направление)", value=True)


@st.cache_data(show_spinner="Считаю LIC…")
def cached(mode, field, size, length_px, h_px, overlay):
    return render(mode, field, size, length_px, h_px, overlay)


fig, elapsed = cached(mode, field, size, length_px, h_px, overlay)
st.pyplot(fig)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150)
st.download_button("Сохранить изображение (PNG)", buf.getvalue(), file_name=f"{field}_{mode}.png", mime="image/png")
if elapsed is not None:
    st.caption(f"Время расчёта LIC: {elapsed:.2f} с")
if mode == "LIC" and field in ("source", "sink"):
    st.info("Источник и сток дают одинаковую LIC-картину: симметричное ядро не различает направление. "
            "Направление показывают стрелки.")
