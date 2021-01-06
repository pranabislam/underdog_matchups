# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:35:11 2020

@author: sanjs
"""

import pandas as pd
from merge_players_to_bookies_master_ss import make_clean_dfs
from basketball_reference_scraper.box_scores import get_box_scores
from schedule import get_clean_schedule
import os

clean_bookie_dfs = make_clean_dfs()

teams = {
    "ATL":"Atlanta Hawks",
    "BRK":"Brooklyn Nets",
    "BOS":"Boston Celtics",
    "CHA":"Charlotte Bobcats",
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

# returns cleaned dataframe of box scores for a given game
def clean_box_scores(date, team1, team2, stat_type = 'BASIC'):
    if stat_type == 'ADVANCED':
        box = get_box_scores(date, team1, team2, stat_type = 'ADVANCED')
        for b in box:
            box[b].drop(['USG%', 'BPM'], axis = 1, inplace = True)
    else:
        box = get_box_scores(date, team1, team2)
        for b in box:
            box[b].drop('+/-', axis = 1, inplace = True)
    for b in box:
        box[b] = box[b][box[b].MP != 'Did Not Play']
        box[b] = box[b][box[b].MP != 'Player Suspended']
        box[b] = box[b][box[b].MP != 'Not With Team']
        box[b] = box[b][box[b].MP != 'Did Not Dress']
        box[b] = box[b].drop(['PLAYER', 'MP'], axis = 1).apply(pd.to_numeric)
        #box[b] = box[b][:-1]
        #avg = box[b].mean(axis = 0)
        #box[b] = box[b].append(avg, ignore_index = True)
        #box[b] = box[b].append(box[b].tail(1).drop('PLAYER', axis = 1), ignore_index = True)
    
    return box

# box = cleanboxscores('2019-10-22', 'LAC', 'LAL', stat_type = 'ADVANCED')



# team strings must be in key (abbreviated) format
# merge basic + advanced stats and then output team average
# assuming team1 is home team

def merge_box_scores(date, team1, team2):
    box_bas = clean_box_scores(date, team1, team2)
    box_adv = clean_box_scores(date, team1, team2, stat_type = 'ADVANCED')
    box = {}
    for team in zip(team1, team2):    
        box[team1] = pd.concat([box_bas[team1], box_adv[team1]], axis = 1, sort = False)
        box[team1].columns = ['Home ' + str(col) for col in box[team1].columns]
        
        box[team2] = pd.concat([box_bas[team2], box_adv[team2]], axis = 1, sort = False)
        box[team2].columns = ['Away ' + str(col) for col in box[team2].columns]
    
    avgteam1 = box[team1].tail(1)
    avgteam2 = box[team2].tail(1)
    avgteam1.index, avgteam2.index = [0], [0]
    avg = pd.concat([avgteam1, avgteam2], axis = 1)
    return avg


stats = ['Home FG', 'Home FGA', 'Home FG%', 'Home 3P', 'Home 3PA', 'Home 3P%',
       'Home FT', 'Home FTA', 'Home FT%', 'Home ORB', 'Home DRB', 'Home TRB',
       'Home AST', 'Home STL', 'Home BLK', 'Home TOV', 'Home PF', 'Home PTS',
       'Home TS%', 'Home eFG%', 'Home 3PAr', 'Home FTr',
       'Home ORB%', 'Home DRB%', 'Home TRB%', 'Home AST%', 'Home STL%',
       'Home BLK%', 'Home TOV%', 'Home ORtg', 'Home DRtg',
       'Away FG', 'Away FGA', 'Away FG%', 'Away 3P', 'Away 3PA',
       'Away 3P%', 'Away FT', 'Away FTA', 'Away FT%', 'Away ORB', 'Away DRB',
       'Away TRB', 'Away AST', 'Away STL', 'Away BLK', 'Away TOV', 'Away PF',
       'Away PTS', 'Away TS%', 'Away eFG%', 'Away 3PAr',
       'Away FTr', 'Away ORB%', 'Away DRB%', 'Away TRB%', 'Away AST%',
       'Away STL%', 'Away BLK%', 'Away TOV%', 'Away ORtg',
       'Away DRtg']


# need to write each season to csv since function takes long time to run

    

def team_check(team, year):
    if year < 2015:
        if team == 'CHO':
            team = 'CHA'
    if year < 2014:
        if team == 'NOP':
            team = 'NOH'
    if year < 2013:
        if team == 'BRK':
            team = 'NJN'
    return team


def sched_stats(year):
    sched = get_clean_schedule(year)
    sched['DATE'] = sched['DATE'].dt.strftime('%Y-%m-%d')
    teamstats = pd.DataFrame(columns = stats)
    for index, row in sched.iterrows():
        date = row['DATE']
        team1 = list(teams.keys())[list(teams.values()).index(row['HOME'])]
        team1 = team_check(team1, year)
        team2 = list(teams.keys())[list(teams.values()).index(row['VISITOR'])]
        team2 = team_check(team2, year)
        avg = merge_box_scores(date, team1, team2)    
        avg.index = [index]
        teamstats = teamstats.append(avg)
    sched_merge = pd.concat([sched, teamstats], axis = 1, sort = True)
    return sched_merge


def merge_team_stats(year):
    filepath = os.getcwd() + '\\bookie merge files\\'
    bookie_df = pd.read_csv(filepath + '{}_bookie_merge.csv'.format(year)).drop('Unnamed: 0', axis = 1)
    bookie_df = pd.concat([bookie_df,pd.DataFrame(columns = stats)], sort = False)
    sched = sched_stats(year)
    for index, row in bookie_df.iterrows():
        date = row['DATE']
        team1 = row['Home Team']
        team2 = row['Away Team']
        app = sched[(sched['DATE'] == date) & (sched['HOME'] == team1) & (sched['VISITOR'] == team2)][stats]
        app.index = [index]
        bookie_df.loc[index, stats] = app.loc[index, stats]
    bookie_df.to_csv(os.getcwd() + f'\\final merge files\\{year}_stats.csv')
    return



