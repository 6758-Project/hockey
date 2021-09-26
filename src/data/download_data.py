"""
Downloads data from the NHL API.
"""
import argparse
import os
import logging
import requests

from typing import List


game_url_from_id = lambda game_id: f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/"


def download_games(game_ids: List, datadir: os.PathLike):
    """ Downloads data for a set of NHL Game IDs.

        Applies a data directory structure of year/type/ID.json
        Logs warnings if any game ID are misspecified.
    """
    for game_id in game_ids:
        response = requests.get(game_url_from_id(game_id))

        if response.status_code == 404:
            logging.warning(f"No data returned for game ID {game_id} (404), so skipping")
        else:
            path_to_datafile = os.path.join(datadir, game_id[:4], game_id[4:6], f"{game_id}.json")
            with open(path_to_datafile, "wb") as f:
                f.write(response.content)


def main(args):
    requested_game_ids = []

    for season in args.seasons:
        regular_season_games = 1230 if season <= 2016 else 1271

        # regular season
        if not args.postseason_only:
            for game_num in range(1, regular_season_games):
                game_num_padded = str(game_num).zfill(4)
                regular_game_id = str(season) + "02" + game_num_padded
                requested_game_ids.append(regular_game_id)

        if not args.regular_season_only:
            for rd in range(1, 4+1):
                series_per_round = [8, 4, 2, 1][rd-1]
                for series_num in range(1, series_per_round+1):
                    for game_num in range(1,7+1):
                        playoff_game_id = str(season) + str("03") + str(series_num)+str(game_num)
                        requested_game_ids.append(playoff_game_id)

        os.makedirs(os.path.join(args.datadir, str(season), "02"), exist_ok=True)
        os.makedirs(os.path.join(args.datadir, str(season), "03"), exist_ok=True)

    download_games(requested_game_ids, args.datadir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download NHL API Data')

    parser.add_argument('-d', '--datadir', nargs='+', default='./data/raw/',
                        help='Which directory to download games')

    parser.add_argument('-s', '--seasons', nargs='+', type=int,
                        default=[2016, 2017, 2018, 2019, 2020],
                        help='Starting year of NHL seasons for which to download games')

    parser.add_argument('-r', '--regular-season-only', dest="regular_season_only",
                        help='(boolean) if passed, download only regular season data',
                        action='store_true')
    parser.set_defaults(regular_season_only=False)

    parser.add_argument('-p', '--postseason-only', dest="postseason_only",
                        help='(boolean) if passed, download only postseason data',
                        action='store_true')
    parser.set_defaults(postseason_only=False)

    args = parser.parse_args()

    if args.regular_season_only and args.postseason_only:
        raise ValueError("If both regular and postseason data is desired, do not pass any filter flags")

    main(args)
