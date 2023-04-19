import itertools
import logging
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# HACK: utils.py is a symlink to the actual file
# I cannot get relative imports to work!
from utils import (
    Experiment,
    MatrixResult,
    load_experiment_pickle,
    save_experiment_pickle,
    exist_experiment_pickle,
    get_cmap_by_metric,
    get_pretty_metric,
    get_pretty_valuation,
    get_pretty_valuation_pair,
)

LOGGER = logging.getLogger('medai.streamlit-app')


def build_suptitle(exp, result_i, metric_i):
    result = exp.results[result_i]
    pretty_metric = get_pretty_metric(result.metric, metric_i=metric_i)
    dataset = "IU X-ray" if exp.dataset == "iu" else "MIMIC-CXR"
    return f"{pretty_metric} in {exp.abnormality} sentences ({dataset} dataset)"


def metric_name_to_index(exp, metric_name, groups):
    if metric_name == "bleurt":
        metric_i = 0
    elif "bleu" in metric_name:
        metric_i = int(metric_name[-1]) - 1
        metric_name = metric_name[:-1]
    else:
        metric_i = 0

    for result_i, result in enumerate(exp.results):
        if result.metric == metric_name and result.groups == groups:
            return result_i, metric_i

    available = [(r.metric, r.groups) for r in exp.results]
    st.write(f"metric not found in experiments, try: {available}")
    return None, None


AVAILABLE_ABNORMALITIES = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged-cardiomediastinum",
    "fracture",
    "lung-lesion",
    "lung-opacity",
    "pleural-effusion",
    "pleural-other",
    "pneumonia",
    "pneumothorax",
    "support-devices",
]

AVAILABLE_METRICS = [
    "bleu1", "bleu2", "bleu3", "bleu4", "rouge", "cider-IDF",
    "bleurt", "bertscore",
]

AVAILABLE_GROUPS = [
    (0, 1),
    (-2, 0, -1, 1),
]


def groups_to_str(groups):
    if groups == (0, 1):
        return "2 (neg / pos)"
    if groups == (-2, 0, -1, 1):
        return "4 (none / neg / unc / pos)"
    return "unk"


def build_available_valuations(groups):
    return list(itertools.product(groups, groups))


def build_experiment_name(dataset, abnormality, metric):
    s = f"{dataset}-{abnormality}"
    if metric in ("bertscore", "bleurt"):
        s += f"-{metric}"
    return s

def main():
    st.sidebar.title("Panel")
    dataset = st.sidebar.selectbox("Dataset", ["iu", "mimic"])
    abnormality = st.sidebar.selectbox("Abnormality", AVAILABLE_ABNORMALITIES, 0)
    metric = st.sidebar.selectbox("NLP metric", AVAILABLE_METRICS)
    groups = st.sidebar.selectbox(
        "N classes",
        AVAILABLE_GROUPS,
        len(AVAILABLE_GROUPS) - 1,
        format_func=groups_to_str,
    )
    # log_scale = st.sidebar.checkbox("Log scale", value=False)
    valuations_chosen = st.sidebar.multiselect(
        "Valuations",
        build_available_valuations(groups),
        default=[(0, 0), (0, 1)],
        format_func=get_pretty_valuation_pair,
    )
    # bins = st.sidebar.slider("Histogram bins", 2, 100, 50)
    opacity = st.sidebar.slider("Histogram opacity", 0.0, 1.0, 1.0, step=0.1)

    exp = load_experiment_pickle(build_experiment_name(dataset, abnormality, metric))
    if exp is None:
        st.write(f"experiment not found: {dataset}-{abnormality}")
        return

    RESULT_I, METRIC_I = metric_name_to_index(exp, metric, groups)
    if RESULT_I is None or METRIC_I is None:
        return

    # With plotly
    result = exp.results[RESULT_I]
    ticks = [get_pretty_valuation(k) for k in result.groups]

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Heatmap(
            z=result.cube[METRIC_I],
            x=ticks,
            y=ticks,
            colorscale=get_cmap_by_metric(metric),
            texttemplate="%{z:.2f}",
            showlegend=False,
            # uid=0,
            colorbar={
                # len: 0.35, # Change size of bar
                # 'title': 'Speed(RPM)<br\><br\>', # set title
                # 'titleside':'top', # set postion
                # tickvals:[0,50,100],
                # "x": -.2,
                #"ticklabelposition": "outside top",
            },
        ),
        row=1,
        col=1,
    )
    for chosen in valuations_chosen:
        values = result.dists[chosen]
        if values.ndim > 1:
            values = values[METRIC_I]  # useful for bleu-1, -2, -3, -4
        fig.add_trace(
            go.Histogram(
                x=values,
                opacity=opacity,
                texttemplate="%{y:.2f}",
                histnorm="probability",
                # ybins={"size": bins},
            ),
            row=1,
            col=2,
        )


    # df = pd.DataFrame(result.cube[METRIC_I], columns=ticks, index=ticks)
    # st.write(df)
    # fig = px.imshow(
    #     df,
    #     text_auto=".2f",
    #     color_continuous_scale=get_cmap_by_metric(metric),
    #     title="something",
    # )
    st.write(build_suptitle(exp, RESULT_I, METRIC_I))
    st.plotly_chart(fig)

    # fig = plt.figure(figsize=(15, 6))
    # shape = (1, 2)  # Axes shape
    # ax_hmap = plt.subplot2grid(shape, (0, 0), fig=fig)
    # ax_hist = plt.subplot2grid(shape, (0, 1), fig=fig)

    # _kw = {
    #     "xlabel_fontsize": 14,
    #     "ylabel_fontsize": 14,
    #     "title_fontsize": 15,
    #     "result_i": RESULT_I,
    #     "metric_i": METRIC_I,
    # }
    # plot_heatmap(exp, ax=ax_hmap, title=False, annot_kws={"fontsize": 13}, **_kw)

    # _kw = {
    #     "add_n_to_label": False,
    #     "bins": bins,
    #     "legend_fontsize": 12,
    #     "range": (0, 1),
    #     **_kw,
    # }
    # plot_hists(exp, valuations_chosen, title=False, xlabel=False, ax=ax_hist, **_kw)

    # # Set suptitle
    # plt.suptitle(build_suptitle(exp, RESULT_I, METRIC_I), fontsize=17)

    # # Set titles
    # ax_hist.set_title("Scores distribution", fontsize=_kw["title_fontsize"])
    # ax_hmap.set_title("Scores matrix", fontsize=_kw["title_fontsize"])

    # if log_scale:
    #     ax_hist.set_yscale("log")

    # # increase fontsize of ticks in the first plot (HACKy way)
    # a = fig.axes[0]  # get the first plot
    # a.set_xticklabels(a.get_xticklabels(), fontsize=12)
    # a.set_yticklabels(a.get_yticklabels(), fontsize=12)

    # # st.write(fig)

    # st.write("Sampler used: ", exp.results[RESULT_I].sampler)


if __name__ == "__main__":
    st.set_page_config(
        page_title="NLP vs CheX demo",
        menu_items={
            "About": "something",
            "Get help": "https://github.com/pdpino/medical-ai",
        },
    )
    st.title("Demo NLP metrics vs CheXpert")

    # st.sidebar.selectbox("something", AVAILABLE_ABNORMALITIES)
    main()
