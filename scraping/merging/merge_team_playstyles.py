import pandas as pd
import numpy as np
import os

## Master dictionary for team lookups
## ADD ANY TEAM ABBREVIATION TO THIS DICT AND OUR MAPPED DICTIONARIES WILL BE GOOD
team_name_to_num = {
'Atlanta Hawks':1,
'Boston Celtics':2,
'Brooklyn Nets':3,
'New Jersey Nets':3,
'Charlotte Bobcats':4,
'Charlotte Hornets':4,
'Chicago Bulls':5,
'Cleveland Cavaliers':6,
'Dallas Mavericks':7,
'Denver Nuggets':8,
'Detroit Pistons':9,
'Golden State Warriors':10,
'Houston Rockets':11,
'Indiana Pacers':12,
'LA Clippers':13,
'Los Angeles Clippers':13,
'Los Angeles Lakers':14,
'Memphis Grizzlies':15,
'Miami Heat':16,
'Milwaukee Bucks':17,
'Minnesota Timberwolves':18,
'New Orleans Hornets':19,
'New Orleans Pelicans':19,
'New York Knicks':20,
'Oklahoma City Thunder':21,
'Orlando Magic':22,
'Philadelphia 76ers':23,
'Phoenix Suns':24,
'Portland Trail Blazers':25,
'Sacramento Kings':26,
'San Antonio Spurs':27,
'Toronto Raptors':28,
'Utah Jazz':29,
'Washington Wizards':30
}

num_to_team_name = {i:[] for i in range(1,31)}

for key, value in team_name_to_num.items():
    num_to_team_name[value].append(key)


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
    
    # Filter odds based on significant underdogs & favorites
    odds_filter = ((bookie_means['Home Odds Close'].between(1, 1.5)) | (bookie_means['Home Odds Close'] >= 3) | 
                   (bookie_means['Away Odds Close'].between(1, 1.5)) | (bookie_means['Away Odds Close'] >= 3))
    underdogs_faves = bookie_means[odds_filter]
    
    # Feature cleanup
    underdogs_faves.drop(['Unnamed: 0', 'Bookie Number'], axis=1, inplace=True)
    underdogs_faves.index = range(len(underdogs_faves))
    
    return underdogs_faves

def getDfSummary(input_data):
    output_data_as_dict = {'number_nan': [],
                           'number_distinct': []}
    
    for column in input_data.columns:
        output_data_as_dict['number_nan'].append(input_data[column].isna().sum())
        output_data_as_dict['number_distinct'].append(len(input_data[column].unique()))  
    
    output_data = pd.DataFrame(output_data_as_dict, index=input_data.columns)
    
    return output_data


def drop_columns_and_rows(df):
    '''
    We drop the columns discussed and then rows with NA values (in order due to some rows having NA vals in the 
    columns in question)
    '''
    columns_drop = ['Home FGA', 'Home 3PA', 'Home FTA', 'Home TRB', 'Home PTS', 'Home TRB%',
                    'Away FGA', 'Away 3PA', 'Away FTA', 'Away TRB', 'Away PTS', 'Away TRB%']
    
    df.drop(columns=columns_drop, inplace=True)
    cleaned_df = df.dropna(axis=0)
    
    return cleaned_df


#%%
## Get all filenames and set up vars
#all_files_master = os.listdir('../../final merged csvs/')  # old path
all_files_master = os.listdir('../../final merge no rolling/') # new path
files_to_remove = set(['2009_stats.csv', '2010_stats.csv', '2011_stats.csv', 
                   '2012_stats.csv', '2013_stats.csv'])
all_files_master = list(set(all_files_master) - files_to_remove)
all_files_master.sort()

all_files_playstyles = os.listdir('../team_playtype_stats/')

## Iterate through all files and add the playstyle stats
for master_csv in all_files_master:
    path = '../../final merge no rolling/' + master_csv
    playstyle_year = str(int(master_csv[:4]) - 1) ## We are matching playstyle year to prior year performance 

    ## Old function used to need to clean, now we read in a cleaned df
    # current_master_df = underdogs_faves_aggregrates(pd.read_csv(path, parse_dates=['DATE']))
    current_master_df = drop_columns_and_rows(pd.read_csv(path, parse_dates=['DATE']))

    current_master_df['Home Team Num'] = current_master_df['Home Team'].apply(lambda x: team_name_to_num[x])
    current_master_df['Away Team Num'] = current_master_df['Away Team'].apply(lambda x: team_name_to_num[x])

    for playstyle_filename in all_files_playstyles:
        path = '../team_playtype_stats/' + playstyle_filename
        team_play_stats = pd.read_excel(path, sheet_name=playstyle_year)[['TEAM', 'PERCENTILE']].loc[1:]
        team_play_stats['Team Num'] = team_play_stats['TEAM'].apply(lambda x: team_name_to_num[x])
        percentiles = {row['Team Num']:row['PERCENTILE']/100 for row_num, row in team_play_stats.iterrows()}
        stat = playstyle_filename[:-5]

        current_master_df['Home Team ' + stat] = current_master_df['Home Team Num'].apply(lambda x: percentiles[x])
        current_master_df['Away Team ' + stat] = current_master_df['Away Team Num'].apply(lambda x: percentiles[x])

        output_filepath = 'cleaned_dfs_11-23/' + master_csv
        
        ## Commenting this out because no need to run again after we have written the files already
    current_master_df.drop(columns=['Home Team Num', 'Away Team Num'], inplace=True)
    if master_csv == '2020_stats.csv':
        current_master_df.drop(columns=['index'], inplace=True)
    # uncomment when you want to output
    current_master_df.to_csv(output_filepath, index=False)




################### FUNCTION CALLS ###################
'''
#%%
# For 2009 - 2013
remaining_files = ['2009_stats.csv', '2010_stats.csv', '2011_stats.csv', '2012_stats.csv', '2013_stats.csv']

for f in remaining_files:
    path = '../../final merge no rolling/' + f
    current_master_df = drop_columns_and_rows(pd.read_csv(path, parse_dates=['DATE']))
    output_filepath = 'cleaned_dfs_11-23/' + f
    # uncomment when you want to output
    current_master_df.to_csv(output_filepath, index=False)
# %%
'''

