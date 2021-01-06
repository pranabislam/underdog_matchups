import pandas as pd
import numpy as np
from API_fixes.getschedulefix import get_schedule_fix


def get_clean_schedule(year):
    sched = get_schedule_fix(year)

    sched.replace({'New Jersey Nets': 'Brooklyn Nets', 
                     'Charlotte Bobcats': 'Charlotte Hornets',
                     'New Orleans Hornets': 'New Orleans Pelicans',
                     'Seattle Supersonics': 'Oklahoma City Thunder'}, inplace=True)
    
    return sched


def merge_sched_to_bookies(bookie_df, sched):
    # Create extra potential game dates
    bookie_df['D-1'] = pd.to_datetime(bookie_df['Game Date']) + pd.Timedelta(days=-1)
    bookie_df['D+1'] = pd.to_datetime(bookie_df['Game Date']) + pd.Timedelta(days=1)
    
#     bookie_df['unique_games_test'] = bookie_df['Game Date'].astype(str) + bookie_df['Home Team'] + bookie_df['Away Team']    
    
    # Perform inner joins on all scheduled games with the possible dates created above
    merged_scores_d0 = sched.merge(bookie_df, left_on=['DATE', 'HOME', 'VISITOR'],
                                    right_on=['Game Date', 'Home Team', 'Away Team'],
                                    how='inner')

    merged_scores_dless1 = sched.merge(bookie_df, left_on=['DATE', 'HOME', 'VISITOR'],
                                    right_on=['D-1', 'Home Team', 'Away Team'],
                                    how='inner')
    
    merged_scores_dplus1 = sched.merge(bookie_df, left_on=['DATE', 'HOME', 'VISITOR'],
                                    right_on=['D+1', 'Home Team', 'Away Team'],
                                    how='inner')

    # Concatentate dataframes created above to generate an accurately dated bookies df
    frames = [merged_scores_d0, merged_scores_dless1, merged_scores_dplus1]
    merged_scores = pd.concat(frames, ignore_index=True)
    
    merged_scores.sort_values(by=['DATE', 'HOME'], inplace=True)
    merged_scores = merged_scores.reset_index(drop = True)
    merged_scores.drop_duplicates(inplace=True)

    return merged_scores