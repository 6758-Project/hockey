"""
Splits cleaned data into train, test, and validation datasets.
"""
import argparse
import glob
import os

from typing import List

import numpy as np
import pandas as pd


def split_files(nhl_datadir: os.PathLike) -> (List, List, List):
    """ For an input directory of data files, randomly split the files into
        train, test, and validation datasets.

        Assumes files are named with NHL game IDs. Extracts metadata about season
        and subseason to stratify random sample.

        Returns:
            (train_files, validation_files, test_files) a list of files
    """
    game_files = os.listdir(nhl_datadir)

    game_files = pd.DataFrame({'file': game_files})
    season = game_files['file'].apply(lambda f: int(f[:4]))
    subseason = game_files['file'].apply(lambda f: int(f[4:6]))

    test_files = game_files[season==2019]

    # stratified random sample of season and subseason (regular or post)
    train_files = game_files[~(season==2019)]\
                    .groupby([season, subseason], as_index=False)\
                    .apply(lambda files: files.sample(frac=.8, random_state=1729))
    train_files.index = train_files.index.get_level_values(1)

    train_and_test_idx = np.concatenate([train_files.index, test_files.index])
    validation_files = game_files.loc[~game_files.index.isin(train_and_test_idx)]

    assert len(train_files) + len(validation_files) + len(test_files) == len(game_files), \
           "train/val/test split is not MECE"

    return train_files['file'], validation_files['file'], test_files['file']


def create_split_datasets(clean_datadir, split_datadir):
    train_files, validation_files, test_files = split_files(clean_datadir)

    zipped = zip(['train', 'validation', 'test'], [train_files, validation_files, test_files])
    for name, files in zipped:
        df = pd.concat([pd.read_csv(os.path.join(clean_datadir, n)) for n in files])

        clutter_cols = [col for col in df.columns if "Unnamed" in col]
        df = df.drop(clutter_cols, axis=1).set_index('id')
        df.to_csv(os.path.join(split_datadir, name+".csv"))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save cleaned version of the NHL API Data"
    )

    parser.add_argument(
        "-c",
        "--clean-datadir",
        nargs="+",
        default="./data/cleaned/",
        help="Where the cleaned NHL data is in",
    )

    parser.add_argument(
        "-s",
        "--split-datadir",
        nargs="+",
        default="./data/split/",
        help="Where the split NHL data is in",
    )

    args = parser.parse_args()

    # empty cleaned datadir if exists; otherwise create it
    if os.path.exists(args.split_datadir):
        files = glob.glob(args.split_datadir+"*")
        for f in files:
            os.remove(f)
    else:
        os.makedirs(args.split_datadir, exist_ok=True)

    create_split_datasets(args.clean_datadir, args.split_datadir)
