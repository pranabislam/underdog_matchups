import pandas as pd
import numpy as np
from datetime import date

teams = {
    "ATL":"Atlanta Hawks",
    "BRK":"Brooklyn Nets",
    "BOS":"Boston Celtics",
    "CHA":"Charlotee Bobcats",
    "CHO":"Charlotte Hornets",
    "CHI":"Chicago Bulls",
    "CLE":"Cleveland Cavaliers",
    "DAL":"Dallas Mavericks",
    "DEN":"Denver Nuggets",
    "DET":"Detroit Pistons",
    "GSW":"Golden State Warriors",
    "HOU":"Houston Rockets",
    "IND":"Indiana Pacers",
    "LAC":"Los Angeles Clippers",
    "LAL":"Los Angeles Lakers",
    "MEM":"Memphis Grizzlies",
    "MIA":"Miami Heat",
    "MIL":"Milwaukee Bucks",
    "MIN":"Minnesota Timberwolves",
    "NJN":"New Jersey Nets",
    "NOH":"New Orleans Hornets",
    "NOP":"New Orleans Pelicans",
    "NYK":"New York Knicks",
    "OKC":"Oklahoma City Thunder",
    "ORL":"Orlando Magic",
    "PHI":"Philadelphia 76ers",
    "PHO":"Phoenix Suns",
    "POR":"Portland Trail Blazers",
    "SAC":"Sacramento Kings",
    "SAS":"San Antonio Spurs",
    "SEA":"Seattle Supersonics",
    "TOR":"Toronto Raptors",
    "UTA":"Utah Jazz",
    "WAS":"Washington Wizards"
}


def get_impact_players(players):
    # Get rows of data
    impact_players = {}
    
    for team in teams.values():
        roster = players[players['TEAM'] == team]
        
        if roster.empty:
            continue
        else:
            impact_players[team] = roster.sort_values('MP', ascending=False)[:7]['PLAYER']

    return impact_players


def get_all_stars(players):
    # Create a dataframe of likely NBA all-stars based on 95% percentile of VORPs
    prod_threshold = np.percentile(players['VORP'], 95)
    all_stars = players[players['VORP'] >= prod_threshold]
    
    # Generate the distribution of likely NBA all-stars on each team
    all_stars_count = all_stars[['TEAM','VORP']].groupby('TEAM').count()

    for team in teams.values():
        if team not in all_stars_count.index:
            all_stars_count.loc[team] = {'VORP': 0}
    
    return all_stars_count