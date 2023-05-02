import os
import pickle
import itertools
import streamlit as st
import pandas as pd
# import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# HACK: utils.py is a symlink to the actual file
# I cannot get relative imports to work!
from utils import (
    Experiment,
    MatrixResult,
    get_cmap_by_metric,
    get_pretty_metric,
    get_pretty_valuation,
    get_pretty_valuation_pair,
)

CHEXPERT_DISEASES_6 = [
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion',
    'Atelectasis',
    'Lung Opacity',
]


def load_experiment(name):
    fpath = os.path.join(os.path.dirname(__file__), "experiments", f"{name}.pickle")
    st.write(f'Loading from file: {name}.pickle')
    if not os.path.isfile(fpath):
        return None

    with open(fpath, "rb") as f:
        raw_exp = pickle.load(f)

    if isinstance(raw_exp, dict):
        # Parse raw dict, avoid depending on medai package
        exp = Experiment(raw_exp['abnormality'], raw_exp['grouped'] or {}, raw_exp['dataset'])
        for raw_result in raw_exp['results']:
            exp.add_result(MatrixResult(
                *raw_result,
            ))

        return exp
    elif isinstance(exp, Experiment):
        return raw_exp
    else:
        st.write(f'Wrong pickle type: {type(raw_exp)}')
        return None


def build_suptitle(exp, result_i, metric_i):
    result = exp.results[result_i]
    pretty_metric = get_pretty_metric(result.metric, metric_i=metric_i)
    dataset = "IU X-ray" if exp.dataset == "iu" else "MIMIC-CXR"
    return f"{pretty_metric} in {exp.abnormality} sentences ({dataset} dataset)"


def find_result_index(exp, metric_name, groups):
    for result_i, result in enumerate(exp.results):
        if result.metric == metric_name and result.groups == groups:
            return result_i

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

AVAILABLE_GROUPSELECTION = [
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


def standard_to_str(standard):
    _d = {
        'gold-expert1': 'Senior radiologist (R1)',
        'gold-expert2': 'Trainee radiologist (R2)',
        'silver': 'CheXpert labeler',
    }
    return _d.get(standard, standard)


def groupselection_to_str(groups):
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
    standard = st.sidebar.selectbox(
        "Gold Standard",
        AVAILABLE_STANDARDS,
        AVAILABLE_STANDARDS.index("gold-expert1" if "mimic" in dataset else "silver"),
        format_func=standard_to_str,
    )
    abnormality = st.sidebar.selectbox("Abnormality", AVAILABLE_ABNORMALITIES, 0, format_func=abn_to_str)
    metric_name, metric_i = st.sidebar.selectbox(
        "NLP metric",
        AVAILABLE_METRICS,
        0,
        format_func=metric_to_str,
    )
    groupselection = st.sidebar.selectbox(
        "N classes",
        AVAILABLE_GROUPSELECTION,
        0,
        format_func=groupselection_to_str,
    )
    log_scale = st.sidebar.checkbox(
        "Histogram: log scale",
        value="cider" in metric_name or ("bleu" in metric_name and metric_i != 0),
    )
    bins = st.sidebar.slider("Histogram: n bins", 2, 100, 50)
    opacity = st.sidebar.slider("Histogram: opacity", 0.0, 1.0, 0.7, step=0.1)

    exp_name = build_experiment_name(dataset, standard, abnormality, metric_name)
    exp = load_experiment(exp_name)
    if exp is None:
        if "expert" in standard and "iu" in dataset:
            msg = "Using a radiologist as gold-standard is only available in MIMIC-CXR"
        elif "expert" in standard and abnormality not in CHEXPERT_DISEASES_6:
            available_labels = '\n'.join(f'* {l}' for l in CHEXPERT_DISEASES_6)
            msg = f"Using a radiologist as gold-standard is only available for these abnormalities:\n{available_labels}"
        else:
            msg = ""
        st.markdown(f"Experiment not found: {msg}")
        return

    result_i = find_result_index(exp, metric_name, groupselection)
    if result_i is None:
        if "silver" in standard and "chexpert" in metric_name:
            msg = "CheXpert metrics are not available using the CheXpert labeler as gold-standard"
        else:
            msg = get_pretty_metric(metric_name, metric_i)
        st.markdown(f"Metric not found: {msg}")
        return

    result = exp.results[result_i]
    ticks = [get_pretty_valuation(k) for k in result.groups]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2)

    fig.add_trace(
        go.Heatmap(
            z=result.cube[metric_i],
            x=ticks,
            y=ticks,
            colorscale=get_cmap_by_metric(metric_name),
            texttemplate="%{z:.3f}",
            showlegend=False,
            showscale=False,
            hovertemplate="%{y}-%{x}",
        ),
        row=1,
        col=1,
    )

    # FIXME: this could use session_state?
    # https://docs.streamlit.io/library/api-reference/session-state
    DEFAULT_VALUATIONS_CHOSEN = [(0, 0), (0, 1)]
    for chosen in build_available_valuations(groupselection):
        values = result.dists[chosen]
        if values.ndim > 1:
            values = values[metric_i]  # useful for bleu-1, -2, -3, -4

        name = "-".join(get_pretty_valuation(v) for v in chosen)
        fig.add_trace(
            go.Histogram(
                x=values,
                opacity=opacity,
                texttemplate="%{y:.2f}" if "chexpert" in metric_name else "",
                histnorm="probability",
                name=name,
                hoverinfo="none",
                visible=True if chosen in DEFAULT_VALUATIONS_CHOSEN else "legendonly",
                nbinsx=2 if "chexpert" in metric_name else bins,
            ),
            row=1,
            col=2,
        )


    fig.update_layout(
        barmode="group" if "chexpert" in metric_name else "overlay",
        title=build_suptitle(exp, result_i, metric_i),
    )

    fig.update_xaxes(title_text="Generated", row=1, col=1)
    fig.update_yaxes(title_text="Ground Truth", autorange="reversed", scaleanchor='x', row=1, col=1)

    fig.update_xaxes(
        title_text=f"{get_pretty_metric(metric_name, metric_i)} score",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="Frequency",
        type="log" if log_scale else None,
        row=1,
        col=2,
    )

    st.plotly_chart(fig)

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

    # # Set titles
    # ax_hist.set_title("Scores distribution", fontsize=_kw["title_fontsize"])
    # ax_hmap.set_title("Scores matrix", fontsize=_kw["title_fontsize"])

    # if log_scale:
    #     ax_hist.set_yscale("log")

    # # increase fontsize of ticks in the first plot (HACKy way)
    # a = fig.axes[0]  # get the first plot
    # a.set_xticklabels(a.get_xticklabels(), fontsize=12)
    # a.set_yticklabels(a.get_yticklabels(), fontsize=12)


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
