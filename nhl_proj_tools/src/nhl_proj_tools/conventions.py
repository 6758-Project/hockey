""" Codifications of project conventions. """
import os

from typing import List

def create_nhl_data_directory(datadir: os.PathLike, seasons: List[int]):
    """ Creates a directory tree for NHL data.

        Pattern: datadir/season/subseason/
        Ex:  ./data/raw/2016/regular
    """
    for season in seasons:
        os.makedirs(os.path.join(datadir, str(season), "regular"), exist_ok=True)
        os.makedirs(os.path.join(datadir, str(season), "postseason"), exist_ok=True)
