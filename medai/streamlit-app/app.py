import itertools
import logging
import streamlit as st
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
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


def find_result_index(exp, metric_name, groups):
    for result_i, result in enumerate(exp.results):
        if result.metric == metric_name and result.groups == groups:
            return result_i

    available = [(r.metric, r.groups) for r in exp.results]
    st.write(f"metric not found in experiments, try: {available}")
    return None

AVAILABLE_DATASETS = ["mimic", "iu"]

AVAILABLE_STANDARDS = ["gold-expert1", "gold-expert2", "silver"]

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
    ("bleu", 0),
    ("bleu", 1),
    ("bleu", 2),
    ("bleu", 3),
    ("rouge", 0),
    ("cider-IDF", 0),
    ("bleurt", 0),
    ("bertscore", 2),
    ("chexpert", 0),
]

AVAILABLE_GROUPS = [
    (0, 1),
    (-2, 0, -1, 1),
]


def dataset_to_str(dataset):
    return "MIMIC-CXR" if "mimic" in dataset else "IU X-ray"


def abn_to_str(abn):
    return abn.capitalize().replace("-", " ")


def metric_to_str(metric_pair):
    metric_name, metric_i = metric_pair
    return get_pretty_metric(metric_name, metric_i)


def groups_to_str(groups):
    if groups == (0, 1):
        return "2: healthy / abnormal"
    if groups == (-2, 0, -1, 1):
        return "4: unmention / healthy / uncertain / abnormal"
    return "unk"


def build_available_valuations(groups):
    return list(itertools.product(groups, groups))


def build_experiment_name(dataset, standard, abnormality, metric_name):
    if "gold" in standard:
        dataset += f"-expert{standard[-1]}"
    s = f"{dataset}-{abnormality}"
    if metric_name in ("bertscore", "bleurt") and 'expert' not in dataset:
        s += f"-{metric_name}"
    return s


def main():
    st.sidebar.title("Panel")
    dataset = st.sidebar.selectbox("Dataset", AVAILABLE_DATASETS, format_func=dataset_to_str)
    standard = st.sidebar.selectbox("Standard", AVAILABLE_STANDARDS)
    abnormality = st.sidebar.selectbox("Abnormality", AVAILABLE_ABNORMALITIES, 0, format_func=abn_to_str)
    metric_name, metric_i = st.sidebar.selectbox("NLP metric", AVAILABLE_METRICS, format_func=metric_to_str)
    groups = st.sidebar.selectbox(
        "N classes",
        AVAILABLE_GROUPS,
        0,
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

    exp_name = build_experiment_name(dataset, standard, abnormality, metric_name)
    exp = load_experiment_pickle(exp_name, raise_error=False)
    if exp is None:
        st.write(f"Experiment not found: {exp_name}")
        return

    result_i = find_result_index(exp, metric_name, groups)
    if result_i is None:
        return

    # With plotly
    result = exp.results[result_i]
    ticks = [get_pretty_valuation(k) for k in result.groups]

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Heatmap(
            z=result.cube[metric_i],
            x=ticks,
            y=ticks,
            colorscale=get_cmap_by_metric(metric_name),
            texttemplate="%{z:.3f}",
            showlegend=False,
            colorbar=dict(
                # len=0.35, # Change size of bar
                # title='Speed(RPM)<br\><br\>', # set title
                # titleside='top', # set postion
                # tickvals=[0,50,100],
                # x=-.2,
                # ticklabelposition="outside top",
            ),
            # layout=go.Layout(
            #     xaxis=dict(title="Generated"),
            #     yaxis=dict(title="Ground Truth", autorange="reversed"),
            # ),
        ),
        row=1,
        col=1,
    )
    for chosen in valuations_chosen:
        values = result.dists[chosen]
        if values.ndim > 1:
            values = values[metric_i]  # useful for bleu-1, -2, -3, -4
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


    fig.update_layout(
        title=build_suptitle(exp, result_i, metric_i),
        xaxis=dict(title="Generated"),
        yaxis=dict(title="Ground Truth", autorange="reversed"),
    )


    # df = pd.DataFrame(result.cube[METRIC_I], columns=ticks, index=ticks)
    # st.write(df)
    # fig = px.imshow(
    #     df,
    #     text_auto=".2f",
    #     color_continuous_scale=get_cmap_by_metric(metric),
    #     title="something",
    # )
    st.plotly_chart(fig)

    # fig = plt.figure(figsize=(15, 6))
    # shape = (1, 2)  # Axes shape
    # ax_hmap = plt.subplot2grid(shape, (0, 0), fig=fig)
    # ax_hist = plt.subplot2grid(shape, (0, 1), fig=fig)

    # _kw = {
    #     "xlabel_fontsize": 14,
    #     "ylabel_fontsize": 14,
    #     "title_fontsize": 15,
    #     "result_i": result_i,
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
    # plt.suptitle(build_suptitle(exp, result_i, METRIC_I), fontsize=17)

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

    # st.write("Sampler used: ", exp.results[result_i].sampler)


if __name__ == "__main__":
    st.set_page_config(
        page_title="NLP vs CheX demo",
        menu_items={
            "About": "something",
            "Get help": "https://github.com/pdpino/medical-ai",
        },
    )
    st.title("Demo NLP metrics vs CheXpert")

    main()
