"""
Processes split data into data ready for training.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def baseline_preprocess(nhl_plays):
    """ Preprocesses an NHL dataset according to the Milestone 2 baseline guidelines """
    nhl_plays = nhl_plays[nhl_plays['type'].isin(['SHOT','GOAL'])]  # Piazza @219

    # losing irrelevant, misleading, and information-leaking (is_winning_goal) columns
    nhl_plays = nhl_plays.drop(
        ['id', 'home_team', 'away_team', 'type', 'code',
         'shooter_name', 'shooter_team_id', 'shooter_team_code', 'goalie_id',
         'strength_name', 'strength_code', 'is_winning_goal'],
        axis=1
    )
    nhl_plays = nhl_plays.set_index(['game_id','event_index'])

    return nhl_plays


def preprocess_data(split_datadir: os.PathLike, processed_datadir: os.PathLike):
    split_files = [n for n in os.listdir(split_datadir) if n.endswith('.csv')]

    split_files.sort()
    if not split_files == ["test.csv", "train.csv", "validation.csv"]:
        raise ValueError(
            f"split dir should have exactly 3 csv files: train, test, and validation. \
               {len(split_files)} were detected"
        )

    for f in split_files:
        nhl_plays = pd.read_csv(os.path.join(split_datadir, f))
        nhl_plays = baseline_preprocess(nhl_plays)

        nhl_plays.to_csv(os.path.join(processed_datadir, f.replace('.csv', '_processed.csv')))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process raw NHL data for training"
    )

    parser.add_argument(
        "-s",
        "--split-datadir",
        nargs="+",
        default="./data/split/",
        help="Where the split NHL data is in",
    )

    parser.add_argument(
        "-p",
        "--processed-datadir",
        nargs="+",
        default="./data/processed/",
        help="Where to write the ready-for-training NHL data",
    )

    args = parser.parse_args()

    # empty cleaned datadir if exists; otherwise create it
    if os.path.exists(args.processed_datadir):
        files = glob.glob(args.processed_datadir+"*")
        for f in files:
            os.remove(f)
    else:
        os.makedirs(args.processed_datadir, exist_ok=True)

    preprocess_data(args.split_datadir, args.processed_datadir)
