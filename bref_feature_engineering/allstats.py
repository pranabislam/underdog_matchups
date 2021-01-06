import pandas as pd
from basketball_reference_scraper.teams import get_roster, get_team_stats, get_opp_stats, get_roster_stats
from basketball_reference_scraper.players import get_stats, get_game_logs
from basketball_reference_scraper.seasons import get_schedule, get_standings
from basketball_reference_scraper.box_scores import get_box_scores
from gamelogsfix import get_game_logs_fix


# combining basic + advanced stats for a player
def allstats(name, year, start_date = None, end_date = None):
    if start_date is None:
        start_date, end_date = f'{year-1}-10-01', f'{year}-04-30'
        
    gamelogbasic = get_game_logs_fix(name, start_date, end_date, stat_type = 'BASIC', playoffs=False)
    gamelogadv = get_game_logs_fix(name, start_date, end_date, stat_type = 'ADVANCED', playoffs=False)
    gamelogadv = gamelogadv.drop(['G','DATE','AGE','TEAM','HOME/AWAY','OPPONENT','RESULT','GS','MP','GAME_SCORE'], axis = 1)
    merge = pd.concat([gamelogbasic, gamelogadv], axis=1, sort=False)
        
    return merge


# combining basic + advanced stats for a team
def allstatsrolling(team, year, start_date = None, end_date = None):
    names = get_roster(team, year).PLAYER
    # some players have two-way contract marker (TW) - removing that
    for n in names:
        n = n.strip(' (TW)')
    if start_date is None:
        start_date, end_date = f'{year-1}-10-01', f'{year}-04-30'    
    # counting stats as rows and players as columns - to store rolling avgs
    rollingavg = pd.DataFrame(index = stats[9:42], columns = names)
    
    # pull game log for each player, remove irrelevant columns, convert to numeric, average, and put in rollingavg DF
    for n in names:
        gamelog = allstats(n, year, start_date, end_date)
        gamelog = gamelog.drop(['G','DATE','AGE','TEAM','HOME/AWAY','OPPONENT','RESULT','GS','MP'], axis = 1).apply(pd.to_numeric).mean(axis = 0)
        rollingavg[n] = gamelog    
    # add column of team rolling average for each stat    
    rollingavg['Team Average'] = rollingavg.mean(axis = 1)
    # drop NaN columns (players without an appearance over that stretch of games)    
    return rollingavg.dropna(axis = 1, how = 'all')

# should output stats for 15 players who had playing time over this stretch plus team rolling average
allrolling = allstatsrolling('LAL', 2019, '2019-02-20', '2019-02-25')
