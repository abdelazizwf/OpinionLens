import os

from omegaconf import OmegaConf

conf = OmegaConf.load("./params.yaml")


def get_csv_files(path: str):
    assert os.path.exists(path), f"{path!r} doesn't exist!"

    if os.path.isfile(path):
        return [path]

    paths = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.split(".")[-1] == "csv":
                paths.append(os.path.join(dirname, filename))

    return paths


def save_preprocessed_data(data, preprocessed_data_path):
    if not os.path.exists(preprocessed_data_path):
        os.makedirs(preprocessed_data_path)

    train_frac, val_frac, test_frac = conf.preprocessing.data_splits

    train_index = int((train_frac * len(data)))
    val_index = int((val_frac * len(data)) + train_index)
    test_index = int((test_frac * len(data)) + val_index)

    data.iloc[:train_index - 1].to_csv(os.path.join(preprocessed_data_path, "train.csv"), index=False)
    data.iloc[train_index:val_index - 1].to_csv(os.path.join(preprocessed_data_path, "val.csv"), index=False)
    data.iloc[val_index:test_index - 1].to_csv(os.path.join(preprocessed_data_path, "test.csv"), index=False)


if __name__ == "__main__":
    print(get_csv_files("./data/preprocessed/imdb_dataset.csv"))
    print(get_csv_files("./data/"))
