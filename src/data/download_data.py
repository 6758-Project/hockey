"""
Downloads data from the NHL API.
"""
import argparse
import os
import logging
import requests

from nhl_proj_tools.conventions import create_nhl_data_directory
from nhl_proj_tools.data_utils import *

logging.basicConfig(level=logging.INFO)


def download_games(game_ids: List, datadir: os.PathLike):
    """ Downloads data for a set of NHL Game IDs.

        Applies a data directory structure of year/type/ID.json
        Logs warnings if no data associated with game_id.
    """
    for game_id in game_ids:
        response = requests.get(get_game_url(game_id))

        if response.status_code == 404:
            logging.warning(f"No data returned for game ID {game_id} (404), so skipping. " + \
                             "(does it encode an optional postseason game #5, #6, or #7)?")
        else:
            subseason = "regular" if game_id[4:6] == "02" \
                          else "postseason" if game_id[4:6] == "03" \
                          else None
            if not subseason:
                raise ValueError(f"{game_id[4:6]} is not a recognized subseason code")

            path_to_datafile = os.path.join(datadir, game_id[:4], subseason, f"{game_id}.json")
            with open(path_to_datafile, "wb") as f:
                f.write(response.content)


def main(args):
    logging.info("Creating directory structure...")
    create_nhl_data_directory(args.datadir, args.seasons)

    for season in args.seasons:
        logging.info(f"Generating requests for {season} season...")

        requested_game_ids = []
        if not args.postseason_only:
            requested_game_ids.extend(generate_regular_season_game_ids(season))

        if not args.regular_season_only:
            requested_game_ids.extend(generate_postseason_game_ids(season))

        download_games(requested_game_ids, args.datadir)
        logging.info(f"...{season} season successfully downloaded")




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
