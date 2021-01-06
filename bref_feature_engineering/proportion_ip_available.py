from API_fixes.getgamelogsfix import get_game_logs_fix
from bref_feature_engineering.rollingavg_seasonlog_teamrecord import teamrecord
from bref_feature_engineering.players_all import get_vorps, get_players
from bref_feature_engineering.players_impact_star import get_impact_players
import pandas as pd
import numpy as np

def get_games_played(team_record, year, players):
	"""
	Returns a dataframe whose rows are games and whose columns are `players`.
	Each cell is a binary variable indicating whether the player played.
	1 = player played. 0 = player did not play.
	:param team_record: a dataframe consisting of all games played by a team 
		and their outcomes.
	:param year: a year (integer).
	:param players: a list of player names. 
	:return: dataframe of indicators.
	"""
    start_date = team_record["DATE"].iloc[0]
    end_date = team_record["DATE"].iloc[-1]
    player_logs = {player: get_game_logs_fix(player, start_date, end_date) for player in players}
    games_played = pd.DataFrame(0, index=team_record.index, columns=player_logs.keys())
    for player, game_logs in player_logs.items():
        played = np.array(game_logs.index).astype(int)
        games_played[player][played] = 1
    return games_played


def append_proportion_ip_avail(team_record, team_name):
	"""
	Updates the dataframe team_record in place to include one new column:
		`PROP_IP_AVAIL`, which indicates the proportion of impact players 
		who played in a given game. Since we assume the impact players
		play when available, we use this as a proxy for availability of
		impact players. 
	:param team_record: a dataframe consisting of all games played by a team 
		and their outcomes.
	:param team_name: a NBA team. 
	:return: none.
	"""
	year = int(team_record["DATE"].iloc[-1][0:4])
	vorps = get_vorps(year)
	players = get_players(year, vorps)
	impact_players = get_impact_players(players)
	gp_ip = get_games_played(team_record, year, impact_players[team_name])
	prop_ip_avail = np.mean(gp_ip, axis=1)
	team_record["PROP_IP_AVAIL"] = prop_ip_avail




