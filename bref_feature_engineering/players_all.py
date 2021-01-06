from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from datetime import date

from basketball_reference_scraper.teams import get_roster_stats

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


def get_vorps(year):
    # Get rows of data
    page = requests.get('https://www.basketball-reference.com/leagues/NBA_{}_advanced.html'.format(year))
    soup = BeautifulSoup(page.text, "html.parser")
    table = soup.find("table", attrs={"class":"sortable stats_table"})
    rows = table.find_all("tr")
    
    players, vorps = [], []

    # Iterate through each row to extract the player's name and their VORP
    for row in rows:
        player = row.find('a')
        vorp = row.find("td", attrs={"data-stat":"vorp"})

        # Only extract non-null data
        if player != None:
            players.append(player.text)
            vorps.append(vorp.text)

    data = {
        'PLAYER': players,
        'VORP': pd.Series(vorps).astype(float)
    }

    return pd.DataFrame(data=data).drop_duplicates('PLAYER')


def get_players(year, vorps):
    players = pd.DataFrame()

    # Retrieve all active NBA players in a chosen year    
    for team in teams.keys():
        team_roster = get_roster_stats(team, year)
        players = pd.concat([players, team_roster], ignore_index=True)
        
    # Merge players with their respective VORPs
    players = players.merge(vorps, on='PLAYER').drop_duplicates('PLAYER')
    
    players['MP'] = players['MP'].astype(float)
        
    # Change team naming convention to match bookies CSVs using 'teams' dictionary
    players['TEAM'] = players['TEAM'].apply(lambda team: teams[team])
    
    return players[['TEAM', 'PLAYER', 'AGE', 'MP', 'VORP']]


def add_player_features(bookie_df, players, all_stars_count):
    # Features indicating the number of all-stars on both the home & away teams
    bookie_df['Home All Stars'] = all_stars_count.loc[bookie_df['Home Team']].values
    bookie_df['Away All Stars'] = all_stars_count.loc[bookie_df['Away Team']].values
    
    # Calculate the median age of each NBA team
    players['AGE'] = players['AGE'].astype(int)
    med_team_age = players[['TEAM', 'AGE']].groupby('TEAM').median()

    bookie_df['Home Median Age'] = med_team_age.loc[bookie_df['Home Team']].values
    bookie_df['Away Median Age'] = med_team_age.loc[bookie_df['Away Team']].values
    
    return bookie_df