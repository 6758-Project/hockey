 """
 Downloads data from the NHL API.
 """
import argparse
import requests

from typing import List


def download_games(game_ids: List, dest="./data/raw/"):
    """ Downloads data for a set of NHL Game IDs.

        Applies a data directory structure of year/type/ID.json
        Logs warnings if any game ID are misspecified.
    """
    pass  # TODO


def main(args):
    seasons = [2016, 2017, 2018, 2019, 2020]

    requested_game_ids = []

    for season in seasons:
        regular_season_games = 1230 if season <= 2016 else 1271

        # regular season
        for game_num in range(1,regular_season_games):
            game_num_padded = str(game_num).zfill(4)
            regular_game_id = str(season) + game_num_padded
            requested_game_ids += regular_game_id

        # playoffs
        for rd in range(1, 4+1):
            series_per_round = [8, 4, 2, 1][rd-1]
            for series_num in range(1, series_per_round+1):
                for game_num in range(1,7+1):
                    playoff_game_id = str(season) + str("03") + str(series_num)+str(game_num)
                    requested_game_ids += playoff_game_id

    download_games(requested_game_ids)




 if __name__ == '__main__':
     main()
