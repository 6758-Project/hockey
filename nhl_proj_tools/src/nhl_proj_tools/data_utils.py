""" A collection of utility functions for working with NHL data. """

def generate_regular_season_game_ids(season: int):
    """ For an NHL season starting in year SEASON, return all regular season game IDs.

        See the unofficial documentation for information about game ID conventions:
        https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
    """
    regular_season_games = 1230 if season <= 2016 else 1271

    regular_game_ids = []
    for game_num in range(1, regular_season_games+1):
        game_num_padded = str(game_num).zfill(4)
        regular_game_id = str(season) + "02" + game_num_padded
        regular_game_ids.append(regular_game_id)

    return regular_game_ids


def generate_postseason_game_ids(season: int):
    """ For an NHL season starting in year SEASON, returns all postseason game IDs.

        See the unofficial documentation for information about game ID conventions:
        https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
    """
    postseason_game_ids = []
    for rd in range(1, 4+1):
        series_per_round = [8, 4, 2, 1][rd-1]
        for series_num in range(1, series_per_round+1):
            for game_num in range(1,7+1):
                playoff_game_id = str(season) + str("03") + str(rd).zfill(2) + str(series_num)+str(game_num)
                postseason_game_ids.append(playoff_game_id)

    return postseason_game_ids


def get_game_url(game_id: str):
    """ Returns an NHL stats API URL based on an NHL GAME_ID """
    return f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/"

