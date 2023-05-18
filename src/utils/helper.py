
from lightfm.data import Dataset
from lightfm.lightfm import LightFM
from lightfm.evaluation import auc_score
from scipy.sparse import coo_matrix

import gzip
import json
import logging as log
from os import listdir
import numpy as np
from pandas import DataFrame
import pickle
import time
from typing import Optional


def generate_recommendations(
        dataset: Dataset,
        item_ids: list[str],
        model: LightFM,
        user_ids: Optional[list[str]] = None
) -> dict:
    """Generates recommendations based on indicated model, for all or selected users.

    Args:
        dataset (Dataset): Dataset used for model training.
        item_ids (list[str]): List of item ids.
        model (LightFM): LightFm model.
        item_ids (list[str]): List of user ids.

    Returns:
        dict[str, list]: Distionary of user ids with list of recommended items.
    """
    uid_map, _, iid_map, _ = dataset.mapping()

    if user_ids is None:
        users = [i for i in range (0, len(user_ids))]
        user_array = users
    else:
        user_array = [uid_map[user] for user in user_ids]

    item_array = np.arange(len(item_ids), dtype=np.int32)

    recommendations = {}
    for user in user_array:
        scores = model.predict(
            user,
            item_array,
        )

        top_items = np.argsort(-scores)

        recommended_items = [_map_int_to_ext_ids(iid_map, item) for item in top_items[:3]]
        recommendations[_map_int_to_ext_ids(uid_map, user)] = recommended_items
    return recommendations

def get_newest_existing_model_version(path: str) -> int:
    """Gets the newest existing version of model located under given path.

    Args:
        path (str): Path under which model components are located.

    Returns:
        int: The newest model version expressed as a number.
    """
    return int(max(listdir(path))[-5])


def pickle_model_results(
        auc: list[float],
        dataset: Dataset,
        duration: list[float],
        model: LightFM,
        model_name: str,
        path: str,
        version: int
) -> None:
    """Pickles components created during model training.

    Args:
        auc (list[float]): List of AUC metrics per epoch.
        dataset (Dataset): Dataset used for model training.
        duration (list[float]): List of training durations per epoch.
        model (LightFM): LightFm model.
        model_name (str): Model name.
        path (str): Path under which model components will be saved.
        version (int): Model version.

    Returns:
        None.
    """
    log.info(f"Saving {model_name}_v{version} components to pickle file...")
    save_data_to_pkl(f"{path}/{model_name}_v{version}.pkl", model)
    save_data_to_pkl(f"{path}/{model_name}_auc_v{version}.pkl", auc)
    save_data_to_pkl(f"{path}/{model_name}_duration_v{version}.pkl", duration)
    save_data_to_pkl(f"{path}/dataset_v{version}.pkl", dataset)
    log.info("Done")


def read_data_from_gziped_file(path: str) -> list:
    """Returns data retrieved from gziped file located under given path.

    Args:
        path (str): Path to gziped file containing data.

    Returns:
        list[dict]: List of dictionaries, which represent file rows.
    """
    file_name = path.split('/')[-1]
    log.info(f"Reading data from file {file_name}...")
    data = []
    with gzip.open(path) as f:
        for l in f:
            l = l.strip().replace("\\n", "")
            data.append(json.loads(l))
    log.info(f"Retrieved {len(data)} records from file {file_name}")
    return data


def save_data_to_pkl(
        path: str,
        data: None
) -> None:
    """Saves data to pickle file.

    Args:
        directory (str): Name of directory in which data file shoul be stored.
        data (object): Data to be written to file.

    Returns:
        None.
    """
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f'File {path} saved')


def select_random_users(
        df: DataFrame,
        column: str,
        n: int = 3
) -> list:
    """Returns list of randomly chosen values from given DataFrame.

    Args:
        df (DataFrame): DataFrame with all data.
        column (str): Name of column from which data will be chosen.
        n (int): Number od values to return. Default is 3.

    Returns:
        list: List of randomly chosen values from given DataFrame and column.
    """
    return list(df.sample(n, axis='rows')[column])


def train_lightfm_model(
        epochs: int,
        model: LightFM,
        model_name: str,
        test: coo_matrix,
        train: coo_matrix
):
    """Trains model and gathers statisctics.

    Args:
        epochs (int): Number of epochs.
        model (LightFM): LightFm model.
        model_name (str): Model name.
        test (coo_matrix): Test dataset.
        train (coo_matrix): Train dataset.

    Returns:
        tuple(list, list): AUC metrics list and durations list.
    """
    model_auc = []
    model_duration = []
    for _ in range(epochs):
        start = time.time()
        model.fit_partial(train, epochs=1)
        model_duration.append(time.time() - start)
        model_auc.append(auc_score(model, test).mean())

    log.info(f"Model {model_name} has been trained in {epochs} epochs")
    return model_auc, model_duration


def unpickle(
        # file_type: str,
        # model_name: str,
        path: str,
        # version: int
):
    # path = f"{path}/{file_type}_v{version}.pkl" if file_type == "dataset" else f'{path}/{model_name}_v{version}.pkl'
    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def _map_int_to_ext_ids(
        mapping: dict,
        searched_id: int
) -> str:
    """Returns external id for provided internal id.
       Searches for key in mapping dictionary by given value.

    Args:
        mapping (dict): Mapping of internal indices to external ids.
        internal_id (int): Internal id which needs to be mapped to external one.

    Returns:
        ext_id (str): External id for provided internal id.
    """
    for ext_id, int_id in mapping.items():
        if int_id == searched_id:
            return ext_id
