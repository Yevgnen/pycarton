# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Iterable
from typing import Union

try:
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st
    from pandas import DataFrame
except ModuleNotFoundError:
    raise RuntimeError(
        "Please install `pandas`, `streamlit` and `plotly` library to use gallary."
    )


def display_clustering(
    prediction_files: Iterable[str],
    df_score: Union[str, DataFrame],
    limit: int = 100,
    k_field: str = "k",
    score_fields: Union[str, Iterable[str]] = "score",
    width: int = 1000,
) -> None:
    if isinstance(df_score, str):
        df_score = pd.read_csv(df_score)

    if isinstance(score_fields, str):
        score_fields = [score_fields]
    else:
        score_fields = list(score_fields)  # type: ignore

    # Sidebar
    st.set_page_config(layout="wide")

    st.sidebar.header("Files")
    filename = st.sidebar.selectbox("Select a file", list(prediction_files))
    df = pd.read_csv(filename)

    st.sidebar.header("Label")
    label = st.sidebar.selectbox("Select a label", sorted(df["prediction"].unique()))

    filtered = df[df["prediction"] == label]

    st.sidebar.header("Options")
    limit = st.sidebar.slider(
        "Numers of examples", min_value=1, max_value=len(filtered), value=limit
    )
    hide_overview = st.sidebar.checkbox("Hide overview", False)

    # Body
    st.title("Clustering Report")
    st.header("Overview")

    # Overview
    if not hide_overview:
        w = 0.1
        fig = go.Figure()
        for i, field in enumerate(score_fields):
            fig.add_trace(
                go.Scatter(
                    x=df_score[k_field],
                    y=df_score[field],
                    name=field,
                    yaxis=f"y{i + 1}" if i > 0 else "y",
                )
            )

            layout = {
                "title": field,
            }
            if i > 0:
                layout.update(
                    {
                        "title": field,
                        "overlaying": "y",
                        "side": "right" if i % 2 == 1 else "left",
                        "position": 1 - (i // 2) * w if i % 2 == 1 else (i // 2) * w,
                        "anchor": "x" if i == 1 else "free",
                    }
                )

            fig.update_layout(
                {
                    f'yaxis{i + 1 if i > 0 else ""}': layout,
                }
            )
        fig.update_layout(
            {
                "xaxis": {
                    "domain": [
                        ((len(score_fields) + 1) // 2) * w,
                        1 - (len(score_fields) // 2) * w,
                    ],
                },
                "width": width,
            }
        )
        st.plotly_chart(fig)

    # Predictions
    st.header("Predictions")
    st.table(filtered.iloc[:limit])
