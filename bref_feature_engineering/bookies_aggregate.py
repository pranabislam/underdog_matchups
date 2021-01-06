import pandas as pd
import numpy as np


def underdogs_faves_aggregrates(bookie_df):
    # Create a unique game ID to identify games on which ot perform groupby operation
    bookie_df['unique_game_id'] = bookie_df['DATE'].astype(str) + '_' + bookie_df['Home Team'] + '_' + bookie_df['Away Team']    
    main_game_data = bookie_df['unique_game_id'].drop_duplicates().str.split('_', expand=True)
    
    # Perform groupby to get mean bookie odds per game
    bookie_means = bookie_df.groupby(['unique_game_id']).mean()

    # Add unique game data back onto new df
    bookie_means.insert(1, 'DATE', list(pd.to_datetime(main_game_data[0])))
    bookie_means.insert(2, 'Home Team', list(main_game_data[1]))
    bookie_means.insert(3, 'Away Team', list(main_game_data[2]))
    bookie_means.insert(0, 'unique_game_id', list(bookie_means.index))
    
    # Modify age feature for simplicity
    bookie_means.insert(15, 'Home Relative Age Diff', list(bookie_means['Home Median Age'] - bookie_means['Away Median Age']))

    
    # Filter odds based on significant underdogs & favorites
    odds_filter = ((bookie_means['Home Odds Close'].between(1, 1.5)) | (bookie_means['Home Odds Close'] >= 3) | 
                   (bookie_means['Away Odds Close'].between(1, 1.5)) | (bookie_means['Away Odds Close'] >= 3))
    underdogs_faves = bookie_means[odds_filter]
    
    # Feature cleanup
    underdogs_faves.drop(['Unnamed: 0', 'Bookie Number', 'PRIOR_GAME_V_OPP', 
                          'Away Win', 'Home Median Age', 'Away Median Age'], axis=1, inplace=True)
    underdogs_faves.index = range(len(underdogs_faves))
    
    return underdogs_faves


# Read in & write out all relevant CSV files
for year in range(2009, 2021):
    bookie_df = pd.read_csv('DS-GA-1001-Odd-Sharks/final_merged_csvs_v2/{}_stats.csv'.format(year), parse_dates=['DATE'])

    agg_df = underdogs_faves_aggregrates(bookie_df)
    agg_df.to_csv('DS-GA-1001-Odd-Sharks/avg_odds_csvs/{}_stats.csv'.format(year), index=False)