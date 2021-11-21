"""
Extracts the events information from the downloaded data with the NHL API.
"""
import argparse
import glob
import os
import json
import logging
import math

import numpy as np
import pandas as pd

from datetime import timedelta

from nhl_proj_tools.data_utils import (
    generate_regular_season_game_ids,
    generate_postseason_game_ids,
)

STANDARDIZED_GOAL_COORDINATES = (89, 0)


def flip_coord_to_one_side(game_events_df, right_team, left_team):
    """
    Flip the (x,y) coordinates of the shots events to the right side of the rink for both teams

    :param pd.DataFrame game_events_df: all the game events
    :param str right_team: the team that started the game on the right side of the rink
    :param str left_team: the team that started the game on the left side of the rink
    :return: a dataframe of the game events data after updating
    :rtype: pd.DataFrame
    """
    for idx, row in game_events_df.iterrows():
        period = row["period"]

        # keep the team who started on the right to the right always
        if row["shooter_team_name"] == right_team and period % 2 == 0:
            game_events_df.at[idx, "coordinate_x"] = row["coordinate_x"] * -1
            game_events_df.at[idx, "coordinate_y"] = row["coordinate_y"] * -1

        # flip the team who started on the left to the right always
        elif row["shooter_team_name"] == left_team and period % 2 != 0:
            game_events_df.at[idx, "coordinate_x"] = row["coordinate_x"] * -1
            game_events_df.at[idx, "coordinate_y"] = row["coordinate_y"] * -1
    return game_events_df


def add_milestone2_advanced_metrics(events_df):
    """
    Add advanced features related to the preceding event for each one
    """
    # 1. elapsed time since the game started
    tmp = events_df["time"].str.split(":", expand=True).astype(int)
    events_df["game_sec"] = (events_df["period"] - 1) * 20 * 60 + tmp[0] * 60 + tmp[1]

    # getting information about previous events using df.shift(periods=1)
    prev_events_df = events_df.shift(periods=1)

    # 2. previous event type
    events_df["prev_event_type"] = prev_events_df["type"]

    # 3. time difference in seconds
    events_df["prev_event_time_diff"] = (
        events_df["game_sec"] - prev_events_df["game_sec"]
    )

    # 4. distance between current and previous event
    # first we calcualte the angle between the current and previous events (in degrees)
    events_df["angle_between_prev_event"] = (
        (events_df["angle"] - prev_events_df["angle"]).abs().astype(float).round(4)
    )
    a = events_df["distance_from_net"]
    b = prev_events_df["distance_from_net"]
    # then with the knowledge of the two sides of a triangle and its angle, we can get the third side length
    events_df["distance_from_prev_event"] = np.sqrt(
        a ** 2
        + b ** 2
        - (2 * a * b * np.cos(events_df["angle_between_prev_event"] * np.pi / 180.0))
    )
    events_df["distance_from_prev_event"] = (
        events_df["distance_from_prev_event"].astype(float).round(4)
    )
    # 5. rebound angle is the change in angle between current and previous shot events = [0,180]
    rebound_angle_mask = (
        (events_df["type"] == "SHOT")
        & (events_df["prev_event_type"] == "SHOT")
        & (events_df["shooter_team_name"] == prev_events_df["shooter_team_name"])
        & (events_df["period"] == prev_events_df["period"])
    )
    events_df["rebound_angle"] = events_df["angle_between_prev_event"]
    events_df.loc[~rebound_angle_mask, "rebound_angle"] = 0.0

    # 6. see if the current event is a rebound
    events_df['is_rebound'] = False
    events_df.loc[rebound_angle_mask, 'is_rebound'] = True

    # 7. speed of the puck
    speed_mask = events_df["prev_event_time_diff"] > 0
    events_df["speed"] = (
        events_df[speed_mask]["distance_from_prev_event"]
        / events_df[speed_mask]["prev_event_time_diff"]
    )
    events_df["speed"] = events_df["speed"].astype(float).round(4)
    events_df.loc[np.isnan(events_df["speed"]) | (events_df["period"] != prev_events_df["period"]), "speed"] = 0.0

    
    
   
    
    # 8. Time since the power-play started
    # Flags are temporary storage variables
    Home_power_startstamp = 0
    Guest_power_startstamp = 0

    Guest_minor_exp_end = np.empty(0)
    Guest_DM_exp_end = np.empty(0)
    Guest_major_exp_end = np.empty(0)

    Home_minor_exp_end = np.empty(0)
    Home_DM_exp_end = np.empty(0)
    Home_major_exp_end = np.empty(0)

    HomeTeamPowerPlayFlag = 0
    GuestTeamPowerPlayFlag = 0

    # Loop through all rows, not efficient, better method investigating
    for i in range(events_df.shape[0]):

        # Guest team got a penalty, home team powerplay
        if (events_df.at[i,"type"]=="PENALTY") and (events_df.at[i,"shooter_team_name"]==events_df.at[i,"away_team"]):

            # Penalty is a doulbe minor penalty
            if (events_df.at[i,"penaltyMinutes"] == 4):
                # Expected ending time of the double minor penalty
                Guest_DM_exp_end = np.append(Guest_DM_exp_end,(events_df.at[i,"game_sec"] + (60*4)))

            # Penalty is a major penalty
            elif (events_df.at[i,"penaltyMinutes"] == 5):
                # Expected ending time of the major penalty
                Guest_major_exp_end = np.append(Guest_major_exp_end,(events_df.at[i,"game_sec"] + (60*5)))

            # Penalty is a minor penalty
            elif (events_df.at[i,"penaltyMinutes"] == 2):
                # Expected ending time of the minor penalty      
                Guest_minor_exp_end = np.append(Guest_minor_exp_end,(events_df.at[i,"game_sec"] + (60*2)))
           
         
        # Home team got a penalty, guest team powerplay   
        elif (events_df.at[i,"type"]=="PENALTY") and (events_df.at[i,"shooter_team_name"]==events_df.at[i,"home_team"]):
            
            # Penalty is a doulbe minor penalty
            if (events_df.at[i,"penaltyMinutes"] == 4):
                # Expected ending time of the double minor penalty
                Home_DM_exp_end = np.append(Home_DM_exp_end,(events_df.at[i,"game_sec"] + (60*4)))

            # Penalty is a major penalty
            elif (events_df.at[i,"penaltyMinutes"] == 5):
                # Expected ending time of the major penalty           
                Home_major_exp_end = np.append(Home_major_exp_end,(events_df.at[i,"game_sec"] + (60*5)))

            # Penalty is a minor penalty
            elif (events_df.at[i,"penaltyMinutes"] == 2):
                # Expected ending time of the minor penalty       
                Home_minor_exp_end = np.append(Home_minor_exp_end,(events_df.at[i,"game_sec"] + (60*2)))
                              
                
        # Goal occured, if guest team scored, clear home team penalty
        elif (events_df.at[i,"type"]=="GOAL") and (events_df.at[i,"shooter_team_name"]==events_df.at[i,"away_team"]):

            # Detele the first minor penalty timer
            if Home_minor_exp_end.size != 0:
                Home_minor_exp_end = np.delete(Home_minor_exp_end,0)
            
            # Reset the double minor penalty timer, remove 1 minor penalty and start the next one
            Home_DM_exp_end = Home_DM_exp_end[Home_DM_exp_end > (2*60) ]
            if Home_DM_exp_end.size != 0:
                Home_DM_exp_end[:] = (2*60) + events_df.at[i,"game_sec"]
        
        # Goal occured, if home team scored, clear guest team penalty
        elif (events_df.at[i,"type"]=="GOAL") and (events_df.at[i,"shooter_team_name"]==events_df.at[i,"home_team"]):    
            
            # Detele the first minor penalty timer
            if Guest_minor_exp_end.size != 0:
                Guest_minor_exp_end = np.delete(Guest_minor_exp_end,0)
                
            # Reset the double minor penalty timer, remove 1 minor penalty and start the next one
            Guest_DM_exp_end = Guest_DM_exp_end[Guest_DM_exp_end > (2*60) ]
            if Guest_DM_exp_end.size != 0:
                Guest_DM_exp_end[:] = (2*60) + events_df.at[i,"game_sec"]
        
        
        # Always update penalty timer, remove expired ones
        Home_minor_exp_end = Home_minor_exp_end[Home_minor_exp_end > events_df.at[i,"game_sec"]]
        Home_DM_exp_end = Home_DM_exp_end[Home_DM_exp_end > events_df.at[i,"game_sec"]]
        Home_major_exp_end = Home_major_exp_end[Home_major_exp_end > events_df.at[i,"game_sec"]]

        Guest_minor_exp_end = Guest_minor_exp_end[Guest_minor_exp_end > events_df.at[i,"game_sec"]]
        Guest_DM_exp_end = Guest_DM_exp_end[Guest_DM_exp_end > events_df.at[i,"game_sec"]]
        Guest_major_exp_end = Guest_major_exp_end[Guest_major_exp_end > events_df.at[i,"game_sec"]]   
        
            
        # Update skater number
        # 9. Number of friendly non-goalie skaters
        if (Home_minor_exp_end.size + Home_DM_exp_end.size + Home_major_exp_end.size) > 2:
            events_df.at[i,'Home_Skaters'] = 3
        else:
            events_df.at[i,'Home_Skaters'] = 5 - (Home_minor_exp_end.size + Home_DM_exp_end.size + Home_major_exp_end.size)
        
        # 10. Number of opposing non-goalie skaters
        if (Guest_minor_exp_end.size + Guest_DM_exp_end.size + Guest_major_exp_end.size) > 2:
            events_df.at[i,'Guest_Skaters'] = 3
        else:
            events_df.at[i,'Guest_Skaters'] = 5 - (Guest_minor_exp_end.size + Guest_DM_exp_end.size + Guest_major_exp_end.size)


        # Update the power play timer
        # If the guest team has a player advantage, start the power play 
        if (events_df.at[i,'Home_Skaters'] > events_df.at[i,'Guest_Skaters']) and (HomeTeamPowerPlayFlag == 0):
            Home_power_startstamp = events_df.at[i,"game_sec"]
            HomeTeamPowerPlayFlag = 1
        elif ((events_df.at[i,'Home_Skaters'] <= events_df.at[i,'Guest_Skaters'])):   
            Home_power_startstamp = events_df.at[i,"game_sec"]
            HomeTeamPowerPlayFlag = 0
        
        # If the home team has a player advantage, start the power play   
        if (events_df.at[i,'Guest_Skaters'] > events_df.at[i,'Home_Skaters']) and (GuestTeamPowerPlayFlag == 0):
            Guest_power_startstamp = events_df.at[i,"game_sec"]
            GuestTeamPowerPlayFlag = 1
        elif ((events_df.at[i,'Guest_Skaters'] <= events_df.at[i,'Home_Skaters'])):  
            Guest_power_startstamp = events_df.at[i,"game_sec"]
            GuestTeamPowerPlayFlag = 0
                  
        events_df.at[i,"PowerPlayTimerHome"] = events_df.at[i,"game_sec"] - Home_power_startstamp
        events_df.at[i,"PowerPlayTimerGuest"] = events_df.at[i,"game_sec"] - Guest_power_startstamp   
        
    return events_df


def add_milestone2_metrics(events):
    events["distance_from_net"] = (
        (STANDARDIZED_GOAL_COORDINATES[0] - events["coordinate_x"]) ** 2
        + (STANDARDIZED_GOAL_COORDINATES[1] - events["coordinate_y"]) ** 2
    ) ** (0.5)

    events["angle"] = np.arcsin(
        (events["coordinate_y"] / events["distance_from_net"].replace(0, 999)).values
    )  # assumes shots at distance=0 have angle 0

    events["angle"] = (events["angle"] / (np.pi / 2)) * 90  # radians to degrees

    events["is_goal"] = events["type"] == "GOAL"
    events["is_empty_net"] = events["is_empty_net"] == True  # assumes NaN's are False

    return events


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

    # get the home and away teams
    home_team = game_data["gameData"]["teams"]["home"]["name"]
    away_team = game_data["gameData"]["teams"]["away"]["name"]

    # loop over all events in the game
    for event in game_data["liveData"]["plays"]["allPlays"]:
        # get the event type
        event_result_info = event.get("result", None)
        event_type_id = event_result_info.get("eventTypeId", None)
        # a set for all unique events in the game
        event_types.add(event_type_id)

        event_code = event_result_info.get("eventCode", None)
        event_desc = event_result_info.get("description", None)
        event_secondary_type = event_result_info.get("secondaryType", None)
        
        # Adding penalty related information
        penaltyMinutes = event_result_info.get("penaltyMinutes", None)

        # event information
        event_about_info = event.get("about", None)
        event_id = event_about_info.get("eventId", None)
        # event index inside the allPlays in the json file
        event_index = event_about_info.get("eventIdx", None)
        period_num = event_about_info.get("period", None)
        period_type = event_about_info.get("periodType", None)
        event_date = event_about_info.get("dateTime", None)
        event_time = event_about_info.get("periodTime", None)
        event_time_remaining = event_about_info.get("periodTimeRemaining", None)
        event_goals_home = event_about_info["goals"]["home"]
        event_goals_away = event_about_info["goals"]["away"]

        # shooting/scoring team information
        shooter_team_info = event.get("team", None)
        shooter_team_id = (
            shooter_team_info.get("id", None) if shooter_team_info else None
        )
        shooter_team_name = (
            shooter_team_info.get("name", None) if shooter_team_info else None
        )
        shooter_team_code = (
            shooter_team_info.get("triCode", None) if shooter_team_info else None
        )

        # players information (i.e. the shooter/scorer and the goalie)
        # Shooter/scorer information
        players_info = event.get("players", None)
        shooter_info = players_info[0].get("player", None) if players_info else None
        shooter_role = players_info[0].get("playerType", None) if players_info else None
        shooter_id = shooter_info.get("id", None) if shooter_info else None
        shooter_name = shooter_info.get("fullName", None) if shooter_info else None

        # Goalie information
        # GOAL event has from 2 to 4 players info: scorer, goalie and up to two assists
        # SHOOT event has 2 players info: shooter and goalie
        # in both cases the goalie is always at the end of the list
        goalie_info = players_info[-1].get("player", None) if players_info else None
        goalie_role = players_info[-1].get("playerType", None) if players_info else None
        goalie_id = goalie_info.get("id", None) if goalie_info else None
        goalie_name = goalie_info.get("fullName", None) if goalie_info else None

        # info specific to GOAL events
        empty_net = None
        game_winning_goal = None
        strength_name = None
        strength_code = None
        empty_net = event_result_info.get("emptyNet", None)
        game_winning_goal = event_result_info.get("gameWinningGoal", None)
        strength_name = (
            event_result_info["strength"]["name"]
            if "strength" in event_result_info.keys()
            else None
        )
        strength_code = (
            event_result_info["strength"]["code"]
            if "strength" in event_result_info.keys()
            else None
        )

        # (x,y) coordinates of the event
        coord_info = event.get("coordinates", None)

        coord_x = coord_info.get("x", None) if coord_info else None
        coord_y = coord_info.get("y", None) if coord_info else None

        event_entry = {
            "id": event_id,
            "event_index": event_index,
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
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
            "shooter_team_id": shooter_team_id,
            "shooter_team_name": shooter_team_name,
            "shooter_team_code": shooter_team_code,
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
            
            # Adding penalty related information
            "penaltyMinutes": penaltyMinutes,
            
        }
        events.append(event_entry)

    events_df = pd.DataFrame(events)

    # calculate the median of the SHOT x_coordinate to see where did the teams start from (left or right)
    if not events_df.empty:
        median_df = (
            events_df[
                ((events_df["period"] == 1) | (events_df["period"] == 3))
                & (events_df["type"] == "SHOT")
            ]
            .groupby(["shooter_team_name", "home_team"])[
                ["coordinate_x", "coordinate_y"]
            ]
            .median()
            .reset_index()
        )
        for idx, row in median_df.iterrows():
            if row["home_team"] == row["shooter_team_name"]:
                if (
                    row["coordinate_x"] > 0
                ):  # means the home team started on the right side
                    events_df = flip_coord_to_one_side(events_df, home_team, away_team)
                else:
                    events_df = flip_coord_to_one_side(events_df, away_team, home_team)

        events_df = add_milestone2_metrics(events_df)
        events_df = add_milestone2_advanced_metrics(events_df)

    return events_df


def get_events_information(game_id: str, data_dir: str = "../data/raw"):
    """
    gets the GOAL and SHOT events information from a specific game.
    The data should be downloaded using the download_data.py script and in ../data directory of the repository

        :param str game_id: The game id consisting of 10 digits
        :param str data_dir: the raw data directory where the season year directory reside
        :return: a dataframe of the events information
        :rtype: pd.DataFrame
    """
    game_year = game_id[0:4]
    game_type = game_id[4:6]
    game_type_str = ""
    if game_type == "02":
        game_type_str = "regular"
    elif game_type == "03":
        game_type_str = "postseason"

    game_data_path = os.path.join(
        data_dir, f"{game_year}/{game_type_str}/{game_id}.json"
    )

    # check if the file exist
    game_data = ""
    if not os.path.isfile(game_data_path):
        logging.warning(f"game ID: {game_id} doesn't exist, so skipping.")
        return None
    else:
        with open(game_data_path) as f:
            game_data = json.load(f)
            return parse_game_data(game_id, game_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Save cleaned version of the NHL API Data"
    )

    parser.add_argument(
        "-d",
        "--raw-datadir",
        nargs="+",
        default="./data/raw/",
        help="Where the raw NHL data is in",
    )
    parser.add_argument(
        "-c",
        "--clean-datadir",
        nargs="+",
        default="./data/cleaned/",
        help="Where the cleaned NHL data is in",
    )

    args = parser.parse_args()

    # empty cleaned datadir if exists; otherwise create it
    if os.path.exists(args.clean_datadir):
        files = glob.glob(args.clean_datadir + "*")
        for f in files:
            os.remove(f)
    else:
        os.makedirs(args.clean_datadir, exist_ok=True)

    for season in range(2015, 2019 + 1):
        game_ids = generate_regular_season_game_ids(
            season
        ) + generate_postseason_game_ids(season)

        for game_id in game_ids:
            game_cleaned = get_events_information(game_id, data_dir=args.raw_datadir)
            if game_cleaned is not None:
                game_cleaned.to_csv(os.path.join(args.clean_datadir, f"{game_id}.csv"))
