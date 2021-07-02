import datetime
import numpy as np
import pandas as pd
import pickle
import re
from typing import Dict, List, Pattern, Union


def AE(ground_truth: np.array, simulated: np.array) -> np.array:
    """
    Absolute Error
    """
    return np.abs(ground_truth - simulated)


def APE(ground_truth: np.array, simulated: np.array) -> np.array:
    """
    Absolute Percentage Error
    """
    return AE(ground_truth, simulated) / np.abs(ground_truth)


def MAPE(ground_truth: np.array, simulated: np.array) -> float:
    """
    Mean Absolute Percentage Error
    """
    return np.mean(APE(ground_truth, simulated))


def SFAE(ground_truth: np.array, simulated: np.array) -> np.array:
    """
    Scale-Free Absolute Error
    """
    return AE(ground_truth, simulated) / np.mean(ground_truth)


def MAD_mean_ratio(ground_truth: np.array, simulated: np.array) -> float:
    """
    MAD/mean ratio
    """
    return np.mean(SFAE(ground_truth, simulated))


def add_cp6_scalar_metrics(measurements_df: pd.DataFrame) -> pd.DataFrame:
    measurements_df["APE"] = APE(
        measurements_df["ground_truth"], measurements_df["simulated"]
    )

    def add_SFAE(df):
        df["SFAE"] = SFAE(df["ground_truth"], df["simulated"])
        return df

    return measurements_df.groupby(
        ["platform", "informationID", "measurement", "model", "sample"]
    ).apply(add_SFAE)


def calc_cp6_scalar_metrics(
    fns: Union[List[str], str],
    meas_list_scalar: List[str] = ["number_of_shares", "activated_users"],
    target_platforms: Union[None, List[str]] = None,
    year: int = datetime.date.today().year,
) -> pd.DataFrame:
    """
    Description:
        Given a list of files containing measurements,
        `calc_cp6_scalar_metrics()` will construct a data frame containing
        columns for ground truth and simulated values, as well as columns for
        the results of:
        - SFAE (Scale-Free Absolute Error)
        - APE (Absolute Percentage Error)

    Input:
        :fp: (str or list(str)) Filepath(s) to file containing measurements.
        :meas_list_scalar: (list(str)) Measurements on which the metrics are calculated.
        :target_platforms: (list(str) or None) Platforms included in resulting data frame.
        :year: (int) Year in which data were collected (used).

    Output:
        Dataframe containing the following columns:
            - platform
            - informationID
            - measurement
            - model
            - uuid
            - sample
            - split_start
            - split_end
            - ground_truth
            - simulated
            - APE
            - SFAE
    """
    if isinstance(fns, str):
        fns = [fns]

    gt_dfs: List[pd.DataFrame] = []
    sim_dfs: List[pd.DataFrame] = []
    model_counts: Dict[str, Dict[str, int]] = {}
    for fn in fns:
        with open(fn, "rb") as f:
            res = pickle.load(f)

        model: str = res["metrics"]["model_identifier"]
        split: str = res["metrics"]["simulation_period"]

        if split not in model_counts.keys():
            model_counts[split] = {}

        if model not in model_counts[split].keys():
            model_counts[split][model] = 0
        else:
            model_counts[split][model] += 1

        gt: pd.DataFrame = res["ground_truth_results"]
        sim: pd.DataFrame = res["simulation_results"]

        if target_platforms is None:
            target_platforms = gt.keys()

        for platform in target_platforms:
            if (
                not isinstance(gt[platform], dict)
                or platform not in gt.keys()
                or gt[platform] is None
                or "multi_platform" not in gt[platform].keys()
            ):
                continue

            for meas in meas_list_scalar:
                if model_counts[split][model] == 0:
                    df = pd.DataFrame.from_dict(
                        gt[platform]["multi_platform"]["node"][meas], orient="index"
                    ).reset_index()
                    df.columns = ["informationID", "value"]
                    df["measurement"] = meas
                    df["platform"] = platform
                    df["split"] = split
                    gt_dfs.append(df)

                df = pd.DataFrame.from_dict(
                    sim[platform]["multi_platform"]["node"][meas], orient="index"
                ).reset_index()
                df.columns = ["informationID", "value"]
                df["measurement"] = meas
                df["sample"] = model_counts[split][model]
                df["platform"] = platform
                df["model"] = model
                df["split"] = split
                df["file"] = fn

                sim_dfs.append(df)

    sim_df: pd.DataFrame = pd.concat(sim_dfs).rename(columns={"value": "simulated"})
    gt_df: pd.DataFrame = pd.concat(gt_dfs).rename(columns={"value": "ground_truth"})

    uuid_pat: Pattern[str] = re.compile(
        r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})"
    )
    sim_df["uuid"] = sim_df["file"].str.extract(uuid_pat)
    sim_df = sim_df.drop(columns=["file"])
    sim_df = sim_df.drop_duplicates()

    out: pd.DataFrame = pd.merge(
        sim_df,
        gt_df,
        how="outer",  # full join
        on=["platform", "informationID", "measurement", "split"],
    )

    """
    The split column is cut into 2 date-time columns in the event other metrics
    are added that require rows values be sorted temporally.
    """
    out[["split_start", "split_end"]] = out["split"].str.split("-", expand=True)
    if year is None:
        year = datetime.date.today().year
    year_str = str(year)
    out["split_start"] = out["split_start"].apply(lambda x: year_str + x)
    out["split_start"] = pd.to_datetime(out["split_start"], format="%Y%B%d")
    out["split_end"] = out["split_end"].apply(lambda x: year_str + x)
    out["split_end"] = pd.to_datetime(out["split_end"], format="%Y%B%d")

    out_cols = [
        "platform",
        "informationID",
        "measurement",
        "nodeTime",
        "model",
        "uuid",
        "sample",
        "nodeTime",
        "split_start",
        "split_end",
        "ground_truth",
        "simulated",
    ]
    out_cols = [col for col in out_cols if col in out.columns.values.tolist()]
    out = out[out_cols].drop_duplicates()
    return add_cp6_scalar_metrics(out).sort_values(
        ["platform", "informationID", "measurement", "model", "sample", "split_start"]
    )
