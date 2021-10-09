"""
Extracts the events information from the downloaded data with the NHL API.
"""
import os
import json
import logging
import pandas as pd

from nhl_proj_tools.data_utils import generate_regular_season_game_ids, generate_postseason_game_ids


def parse_game_data(game_id: str, game_data: dict):
    """
    parse the game data in a json/dictionary format that has all the events information,
    and retrieve the GOAL and SHOT events

    :param str game_id: the unique id of the game
    :param dict game_data: the game data json file as dictionary
    :return: a dataframe of the events information
    :rtype: pd.DataFrame
    """
    events = []
    event_types = set()
    # loop over all events in the game
    for event in game_data["liveData"]["plays"]["allPlays"]:

        # get the event type
        event_result_info = event["result"]
        event_type_id = event_result_info["eventTypeId"]
        # a set for all unique events in the game
        event_types.add(event_type_id)

        # for this milestone we are interested in SHOT or GOAL types
        if event_type_id == "SHOT" or event_type_id == "GOAL":
            event_code = event_result_info["eventCode"]
            event_desc = event_result_info["description"]
            # doesn't exist in other event types
            event_secondary_type = event_result_info["secondaryType"]

            # event information
            event_about_info = event["about"]
            event_id = event_about_info["eventId"]
            # event index inside the allPlays in the json file
            event_index = event_about_info["eventIdx"]
            period_num = event_about_info["period"]
            period_type = event_about_info["periodType"]
            event_date = event_about_info["dateTime"]
            event_time = event_about_info["periodTime"]
            event_time_remaining = event_about_info["periodTimeRemaining"]
            event_goals_home = event_about_info["goals"]["home"]
            event_goals_away = event_about_info["goals"]["away"]

            # shooting/scoring team information
            team_info = event["team"]
            team_id = team_info["id"]
            team_name = team_info["name"]
            team_code = team_info["triCode"]

            # players information (i.e. the shooter/scorer and the goalie)
            # Shooter/scorer information
            players_info = event["players"]
            shooter_info = players_info[0]["player"]
            shooter_role = players_info[0]["playerType"]
            shooter_id = shooter_info["id"]
            shooter_name = shooter_info["fullName"]

            # Goalie information
            # GOAL event has from 2 to 4 players info: scorer, goalie and up to two assists
            # SHOOT event has 2 players info: shooter and goalie
            # in both cases the goalie is always at the end of the list
            goalie_info = players_info[-1]["player"]
            goalie_role = players_info[-1]["playerType"]
            goalie_id = goalie_info["id"]
            goalie_name = goalie_info["fullName"]

            # info specific to GOAL events
            empty_net = None
            game_winning_goal = None
            strength_name = None
            strength_code = None
            if event_type_id == "GOAL":
                empty_net = event_result_info["emptyNet"]
                game_winning_goal = event_result_info["gameWinningGoal"]
                strength_name = event_result_info["strength"]["name"]
                strength_code = event_result_info["strength"]["code"]

            # (x,y) coordinates of the event
            coord_info = event["coordinates"]
            coord_x = coord_info["x"]
            coord_y = coord_info["y"]

            event_entry = {
                "id": event_id,
                "event_index": event_index,
                "game_id": game_id,
                "type": event_type_id,
                "secondary_type": event_secondary_type,
                "description": event_desc,
                "code": event_code,
                "period": period_num,
                "period_type": period_type,
                "time": event_time,
                "time_remaining": event_time_remaining,
                "date": event_date,
                "goals_home": event_goals_home,
                "goals_away": event_goals_away,
                "team_id": team_id,
                "team_name": team_name,
                "team_code": team_code,
                "shooter_name": shooter_name,
                "shooter_id": shooter_id,
                "goalie_name": goalie_name,
                "goalie_id": goalie_id,
                "is_empty_net": empty_net,
                "is_winning_goal": game_winning_goal,
                "strength_name": strength_name,
                "strength_code": strength_code,
                "coordinate_x": coord_x,
                "coordinate_y": coord_y,
            }
            events.append(event_entry)
    return pd.DataFrame(events)


def get_events_information(game_id: str, data_dir: str = "../data/raw"):
    """
    gets the GOAL and SHOT events information from a specific game.
    The data should be downloaded using the download_data.py script and in ../data directory of the repository

        :param str game_id: The game id consisting of 10 digits
        :param str data_dir: the raw data directory where the season year directory reside
        :return: a dataframe of the events information
        :rtype: pd.DataFrame
        :raises FileNotFoundError: if the game file path does not exist
    """
    game_year = game_id[0:4]
    game_type = game_id[4:6]
    game_type_str = ''
    if game_type == '02':
    	game_type_str = 'regular'
    elif game_type == '03':
    	game_type_str = 'postseason'

    game_data_path = os.path.join(data_dir, f"{game_year}/{game_type_str}/{game_id}.json")

    # check if the file exist
    game_data = ""
    if not os.path.isfile(game_data_path):
        import pdb; pdb.set_trace()
        raise FileNotFoundError("file does not exist!")
    else:
        with open(game_data_path) as f:
            game_data = json.load(f)

    return parse_game_data(game_id, game_data)



if __name__ == '__main__':
    datadir_raw = "/home/jake/Projects/mila/6758/hockey/data/raw"
    datadir_tidied = "/home/jake/Projects/mila/6758/hockey/data/tidied/"
    os.makedirs(datadir_tidied, exist_ok=True)

    for season in [2016, 2017, 2018, 2019, 2020]:
        game_ids = generate_regular_season_game_ids(season) + \
                   generate_postseason_game_ids(season)

        for game_id in game_ids[:2]:
            game_cleaned = get_events_information(game_id, data_dir=datadir_raw)
            game_cleaned.to_csv(os.path.join(datadir_tidied, f"{game_id}.csv"))
