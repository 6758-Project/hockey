"""
Get Teams's data from the NHL raw data
"""
import argparse
import os
import json
import logging
import pandas as pd
from tidy_data import parse_game_data

from typing import List

logging.basicConfig(level=logging.INFO)


# slower way
def save_team_events(
    team_name: str,
    data_dir: str = "../data/raw",
    save_dir: str = "../data/",
    years: List[int] = [2016, 2017, 2018, 2019, 2020],
    seasons: List[str] = ["regular", "postseason"],
):
    """
    save the team's events data (currently only SHOT and GOAL events) for a specific team
    The raw data should be already downloaded using the download_data.py script and in ../data directory of the repository
    
        :param str team_name: The unique team name of interest
        :param str data_dir: the raw data directory where the season year directory reside
        :param str years: the years of interest
        :param str seasons: the seasons of interest
        :param str save_dir: the path for the saved team's information
    """
    # check if the save path exists
    if not os.path.isdir(save_dir):
        logging.info("Creating teams directory ...")
        os.makedirs(save_dir)

    # collect the team's information from all years and seasons
    team_games = []

    logging.info(f"Fetching {team_name}'s data ...")
    for game_year in years:
        for season in seasons:
            year_season_games_dir = os.path.join(data_dir, f"{str(game_year)}/{season}")

            # all games in the same year and season
            json_files = [
                f
                for f in os.listdir(year_season_games_dir)
                if os.path.isfile(os.path.join(year_season_games_dir, f))
            ]
            for game_file in json_files:
                game_data_path = os.path.join(year_season_games_dir, game_file)
                game_id = game_file.split(".")[0]
                with open(game_data_path) as f:
                    game_data = json.load(f)
                    game_info_df = parse_game_data(game_id, game_data)

                    # find the team's events
                    for idx, event in game_info_df.iterrows():
                        if event["team_name"] == team_name:
                            team_event = event.to_dict()
                            team_event["year"] = game_year
                            team_event["season"] = season
                            team_games.append(team_event)
    team_df = pd.DataFrame(team_games)
    team_df.to_csv(os.path.join(save_dir, team_name + ".csv"), index=False)
    logging.info(
        f"Successfully saved {team_name}'s data at {os.path.join(save_dir, team_name+'.csv')}"
    )


def save_games_events(
    data_dir: str = "../data/cleaned",
    save_dir: str = "../data/games",
    years: List[int] = [2016, 2017, 2018, 2019, 2020],
    sub_seasons: List[str] = ["regular", "postseason"],
):
    """
    save a game's events data (currently only SHOT and GOAL events) for a specific season/year and subseason
    The raw data should be already downloaded using the download_data.py script and in `../../data` directory in the repository
    
        :param str team_name: The unique team name of interest
        :param str data_dir: the cleaned data directory
        :param str years: the years of interest
        :param str seasons: the seasons of interest
        :param str save_dir: the path for the saved team's information
    """
    # check if the save path exists
    if not os.path.isdir(save_dir):
        logging.info("Creating games directory ...")
        os.makedirs(save_dir)

    # all games in the cleaned directory
    json_files = [
        f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))
    ]

    # collect the game's information from all years and sub-seasons
    games_of_the_year_dfs = []

    for game_year in years:
        for sub_season in sub_seasons:

            games_of_the_year_dfs = []

            for game_json_file in json_files:
                game_file_year = game_json_file[0:4]
                game_file_sub_season = game_json_file[4:6]

                if game_file_sub_season == "02":
                    sub_str = "regular"
                elif game_file_sub_season == "03":
                    sub_str = "postseason"

                if game_file_year == str(game_year) and sub_str == sub_season:
                    game_data_path = os.path.join(data_dir, game_json_file)

                    game_data_df = pd.read_csv(game_data_path)
                    games_of_the_year_dfs.append(game_data_df)

            # saving all the games in one season year and one sub-season
            pd.concat(games_of_the_year_dfs).to_csv(
                os.path.join(save_dir, str(game_year) + "-" + sub_season + ".csv"),
                index=False,
            )
            logging.info(
                f"Successfully saved {game_year}'s data at {os.path.join(save_dir, str(game_year)+'-'+sub_season+'.csv')}"
            )


def get_all_teams(
    data_dir: str = "./data/raw",
    seasons: List[int] = [2016, 2017, 2018, 2019, 2020],
    sub_seasons: List[str] = ["regular", "postseason"],
):
    """
    gets all teams names that participated in the chosen year(s) and season(s)

        :param str data_dir: the raw data directory where the season year directory reside
        :param str years: the years of interest
        :param str seasons: the seasons of interest
        :return: a list of teams ordered alphabetically
        :rtype: list
    """

    teams_set = set()

    for season in seasons:
        for sub_season in sub_seasons:
            season_dir = os.path.join(data_dir, str(season), sub_season)
            json_files = [
                f
                for f in os.listdir(season_dir)
                if os.path.isfile(os.path.join(season_dir, f))
            ]
            for game_file in json_files:
                file_path = os.path.join(season_dir, game_file)
                with open(file_path) as f:
                    game_data = json.load(f)
                    team_away_name = game_data["gameData"]["teams"]["away"]["name"]
                    team_home_name = game_data["gameData"]["teams"]["home"]["name"]
                    teams_set.add(team_home_name)
                    teams_set.add(team_away_name)
    return list(sorted(teams_set))


def main(args):
    seasons = [2016, 2017, 2018, 2019, 2020]
    sub_seasons = ["regular", "postseason"]

    save_games_events(args.data_dir, args.save_dir, seasons, sub_seasons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NHL Teams Games Events")

    parser.add_argument(
        "-d",
        "--data-dir",
        nargs="+",
        default="./data/cleaned/",
        help="Where is the NHL cleaned data",
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        nargs="+",
        default="./data/games",
        help="Where to save game data",
    )

    args = parser.parse_args()

    main(args)
