import os


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


if __name__ == "__main__":
    print(get_csv_files("./data/preprocessed/imdb_dataset.csv"))
    print(get_csv_files("./data/"))
