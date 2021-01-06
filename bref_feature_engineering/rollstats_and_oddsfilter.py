# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:00:51 2020

@author: sanjs
"""

import pandas as pd
import copy

stats = ['Home FG', 'Home FG%', 'Home 3P', 'Home 3P%','Home FT', 'Home FT%', 
         'Home ORB', 'Home DRB','Home AST', 'Home STL', 'Home BLK', 'Home TOV', 
         'Home PF','Home TS%', 'Home eFG%', 'Home 3PAr', 'Home FTr','Home ORB%', 
         'Home DRB%', 'Home AST%', 'Home STL%','Home BLK%', 'Home TOV%', 
         'Home ORtg', 'Home DRtg','Away FG', 'Away FG%', 'Away 3P', 'Away 3P%', 
         'Away FT', 'Away FT%', 'Away ORB', 'Away DRB', 'Away AST', 'Away STL', 
         'Away BLK', 'Away TOV', 'Away PF', 'Away TS%', 'Away eFG%', 'Away 3PAr',
         'Away FTr', 'Away ORB%', 'Away DRB%', 'Away AST%', 'Away STL%', 
         'Away BLK%', 'Away TOV%', 'Away ORtg','Away DRtg']

# function to calculate rolling stats for a given season and rolling window
bookie_df = pd.read_csv('..\\{}_stats.csv'.format(year), parse_dates=['DATE'])
def roll_stats(bookie_df, n = 5):   
    dummy_df = copy.deepcopy(bookie_df)
    dummy_df['DATE'] = dummy_df['DATE'].dt.strftime('%Y-%m-%d')
    for idx, row in dummy_df.iterrows():
        home = row['Home Team']
        away = row['Away Team']
        date = row['DATE']
        homesched = (bookie_df[(bookie_df['DATE'] < date) & ((bookie_df['Home Team'] == home) | (bookie_df['Away Team'] == home))]).tail(n)
        awaysched = (bookie_df[(bookie_df['DATE'] < date) & ((bookie_df['Home Team'] == away) | (bookie_df['Away Team'] == away))]).tail(n)
        
        #rolling stats
        home_homerolling = homesched[homesched['Home Team'] == home][stats]
        home_homerolling = home_homerolling[home_homerolling.columns.drop(list(home_homerolling.filter(regex='Away')))]
        home_awayrolling = homesched[homesched['Away Team'] == home][stats]
        home_awayrolling = home_awayrolling[home_awayrolling.columns.drop(list(home_awayrolling.filter(regex='Home')))]
        home_awayrolling.columns = home_awayrolling.columns.str.replace('Away', 'Home')
        total_homerolling = home_homerolling.append(home_awayrolling).mean()
        
        away_homerolling = awaysched[awaysched['Home Team'] == away][stats]
        away_homerolling = away_homerolling[away_homerolling.columns.drop(list(away_homerolling.filter(regex='Away')))]
        away_homerolling.columns = away_homerolling.columns.str.replace('Home', 'Away')
        away_awayrolling = awaysched[awaysched['Away Team'] == away][stats]
        away_awayrolling = away_awayrolling[away_awayrolling.columns.drop(list(away_awayrolling.filter(regex='Home')))]
        total_awayrolling = away_homerolling.append(away_awayrolling).mean()
        
        total_rolling = (total_homerolling.append(total_awayrolling)).to_frame().T
        total_rolling.index = [idx]
        dummy_df.loc[idx, stats] = total_rolling.loc[idx, stats]
        
        
        #rolling win pct
        if n != 5:
            home_wins = 0
            home_homerolling_win = homesched[homesched['Home Team'] == home][['Away Underdog', 'Underdog Win']]
            for ind, ro in home_homerolling_win.iterrows():
                if (home_homerolling_win.loc[ind, 'Away Underdog'] + home_homerolling_win.loc[ind, 'Underdog Win']) == 1:
                    home_wins += 1
            home_awayrolling_win = homesched[homesched['Away Team'] == home][['Away Underdog', 'Underdog Win']]
            for ind, ro in home_awayrolling_win.iterrows():
                if (home_awayrolling_win.loc[ind, 'Away Underdog'] + home_awayrolling_win.loc[ind, 'Underdog Win']) != 1:
                    home_wins += 1
            if len(homesched) == 0:
                dummy_df.loc[idx, 'HOME_ROLL_WIN_PCT'] = 0
            else:
                dummy_df.loc[idx, 'HOME_ROLL_WIN_PCT'] = home_wins/len(homesched)
            
            away_wins = 0
            away_homerolling_win = awaysched[awaysched['Home Team'] == away][['Away Underdog', 'Underdog Win']]
            for ind, ro in away_homerolling_win.iterrows():
                if (away_homerolling_win.loc[ind, 'Away Underdog'] + away_homerolling_win.loc[ind, 'Underdog Win']) == 1:
                    away_wins += 1
            away_awayrolling_win = awaysched[awaysched['Away Team'] == away][['Away Underdog', 'Underdog Win']]
            for ind, ro in away_awayrolling_win.iterrows():
                if (away_awayrolling_win.loc[ind, 'Away Underdog'] + away_awayrolling_win.loc[ind, 'Underdog Win']) != 1:
                    away_wins += 1
            if len(awaysched) == 0:
                dummy_df.loc[idx, 'AWAY_ROLL_WIN_PCT'] = 0
            else:
                dummy_df.loc[idx, 'AWAY_ROLL_WIN_PCT'] = away_wins/len(awaysched)
                    
        
        odds_filter = ((dummy_df['Home Odds Close'].between(1, 1.5)) | (dummy_df['Home Odds Close'] >= 3) | 
                   (dummy_df['Away Odds Close'].between(1, 1.5)) | (dummy_df['Away Odds Close'] >= 3))
        final = dummy_df[odds_filter]
        final.dropna(axis = 0, inplace = True)
        final.reset_index(drop = True, inplace = True)
    
    return final




