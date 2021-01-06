# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:58:34 2020

@author: sanjs
"""

import numpy as np

# function to pull prior record vs. opponent up to a given date
def get_season_series(sched):
    sched['DATE'] = sched['DATE'].dt.strftime('%Y-%m-%d')
    sched['RESULT'] = ''
    sched['HOME_PRIOR_WIN_V_OPP'] = ''
    sched['AWAY_PRIOR_WIN_V_OPP'] = ''
    sched['PRIOR_GAME_V_OPP'] = ''
    for idx, row in sched.iterrows():
        date = sched['DATE'].iloc[idx]
        home = sched['HOME'].iloc[idx] 
        away = sched['VISITOR'].iloc[idx]
        if sched['HOME_PTS'].iloc[idx] > sched['VISITOR_PTS'].iloc[idx]:
            sched['RESULT'].iloc[idx] = home
        else:
            sched['RESULT'].iloc[idx] = away
        
        opponent_filter = ((sched['VISITOR'] == away) & (sched['HOME'] == home)) | ((sched['HOME'] == away) & (sched['VISITOR'] == home))
        team_vs_opponent_prior = sched[(sched["DATE"] < date) & opponent_filter]
        sched['HOME_PRIOR_WIN_V_OPP'].iloc[idx] = np.sum(team_vs_opponent_prior['RESULT'] == home)
        sched['AWAY_PRIOR_WIN_V_OPP'].iloc[idx] = np.sum(team_vs_opponent_prior['RESULT'] == away)
        sched['PRIOR_GAME_V_OPP'].iloc[idx] = len(team_vs_opponent_prior)
        
    return sched
