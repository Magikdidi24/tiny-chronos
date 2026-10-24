"""Real training series from the Monash forecasting archive (all CC BY 4.0). None of them is ETT."""
import argparse
import io
import os
import urllib.request
import zipfile

import numpy as np

from train import TrainConfig

URL = "https://zenodo.org/records/{}/files/{}.zip?download=1"
DATASETS = {  # name: zenodo record
    "electricity_hourly_dataset": 4656140,
    "traffic_hourly_dataset": 4656132,
    "australian_electricity_demand_dataset": 4659727,  # half-hourly
    "pedestrian_counts_dataset": 4656626,  # hourly
    "solar_10_minutes_dataset": 4656144,
}
MIN_LENGTH = TrainConfig.ctx_len + TrainConfig.horizon  # one training window


def download(name):
    path = f"data/{name}.zip"
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(URL.format(DATASETS[name], name), path + ".part")
        os.replace(path + ".part", path)  # no half-downloaded zip if it gets interrupted
    return path


def parse_tsf(lines, min_length):
    """.tsf lines -> (the series without missing values and at least min_length long, number of series in the file)."""
    data, series, total = False, [], 0
    for line in lines:
        line = line.strip()
        if line.startswith("@data"):
            data = True
        elif data and line:
            # "attr1:attr2:...:v1,v2,...", the values come after the last ":" and missing ones are "?"
            x = np.array(line.rsplit(":", 1)[-1].replace("?", "nan").split(","), dtype=np.float32)
            total += 1
            if len(x) >= min_length and np.isfinite(x).all():
                series.append(x)
    return series, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    args = p.parse_args()

    values, lengths, dataset = [], [], []
    for i, name in enumerate(args.datasets):
        with zipfile.ZipFile(download(name)) as z:
            [member] = [m for m in z.namelist() if m.endswith(".tsf")]
            # latin-1 never fails on the comment lines, the values are plain ascii
            with io.TextIOWrapper(z.open(member), encoding="latin-1") as f:
                series, total = parse_tsf(f, MIN_LENGTH)
        print(f"{name}: kept {len(series)} of {total} series, {sum(map(len, series)):,} values")
        values += series
        lengths += map(len, series)
        dataset += [i] * len(series)

    np.savez("data/real.npz", values=np.concatenate(values), lengths=np.array(lengths),
             dataset=np.array(dataset), names=np.array(args.datasets))
    print(f"{len(lengths):,} series, {sum(lengths):,} values -> data/real.npz")


if __name__ == "__main__":
    main()
