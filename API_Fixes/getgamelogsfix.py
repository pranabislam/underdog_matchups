# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 23:08:39 2020

@author: sanjs
"""
#%%
import pandas as pd
from basketball_reference_scraper.utils import get_player_suffix
from requests import get
from bs4 import BeautifulSoup
import unicodedata

def get_game_logs_fix(name, start_date, end_date, stat_type = 'BASIC', playoffs=False):
    # Changed to get_player_suffix_fix
    suffix = get_player_suffix_fix(name).replace('/', '%2F').replace('.html', '')
    start_date_str = start_date
    end_date_str = end_date
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    years = list(range(start_date.year, end_date.year+2))
    if stat_type == 'BASIC':
        if playoffs:
            selector = 'div_pgl_basic_playoffs'
        else:
            selector = 'div_pgl_basic'
    elif stat_type == 'ADVANCED':
        if playoffs:
            selector = 'div_pgl_advanced_playoffs'
        else:
            selector = 'div_pgl_advanced'
    final_df = None
    for year in years:
        if stat_type == 'BASIC':         
            r = get(f'https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url={suffix}%2Fgamelog%2F{year}&div={selector}')
        elif stat_type == 'ADVANCED':
            r = get(f'https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url={suffix}%2Fgamelog-advanced%2F{year}&div={selector}')
        if r.status_code==200:
            soup = BeautifulSoup(r.content, 'html.parser')
            table = soup.find('table')
            if table:
                df = pd.read_html(str(table))[0]
                df.rename(columns = {'Date': 'DATE', 'Age': 'AGE', 'Tm': 'TEAM', 'Unnamed: 5': 'HOME/AWAY', 'Opp': 'OPPONENT',
                        'Unnamed: 7': 'RESULT', 'GmSc': 'GAME_SCORE'}, inplace=True)
                df['HOME/AWAY'] = df['HOME/AWAY'].apply(lambda x: 'AWAY' if x=='@' else 'HOME')
                df = df[df['Rk']!='Rk']
                #df = df.drop(['Rk', 'G'], axis=1)
                df = df.loc[(df['DATE'] >= start_date_str) & (df['DATE'] <= end_date_str)]
                active_df = pd.DataFrame(columns = list(df.columns))
                for index, row in df.iterrows():
                    if row['GS']=='Inactive' or row['GS']=='Did Not Play' or row['GS']=='Not With Team' or row['GS']=='Did Not Dress':
                        continue
                    active_df = active_df.append(row)
                if final_df is None:
                    final_df = pd.DataFrame(columns=list(active_df.columns))
                final_df = final_df.append(active_df)
    final_df.set_index(['Rk'], inplace = True)
    return final_df

def get_player_suffix_fix(name):
    
    if name.lower() in repeat_players_list:
        return get_repeat_player_suffix(name)
    
    normalized_name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8")
    names = normalized_name.split(' ')[1:]
    potential_suffixes = [] ## adding this for now
    for last_name in names:
        initial = last_name[0].lower()
        r = get(f'https://www.basketball-reference.com/players/{initial}')
        if r.status_code==200:
            soup = BeautifulSoup(r.content, 'html.parser')
            for table in soup.find_all('table', attrs={'id': 'players'}):
                suffixes = []              
                for anchor in table.find_all('a'):                    
                    if unicodedata.normalize('NFD', anchor.text).encode('ascii', 'ignore').decode("utf-8").lower() in normalized_name.lower():
                        suffix = anchor.attrs['href']
                        player_r = get(f'https://www.basketball-reference.com{suffix}')
                        if player_r.status_code==200:
                            player_soup = BeautifulSoup(player_r.content, 'html.parser')
                            h1 = player_soup.find('h1', attrs={'itemprop': 'name'})
                            if h1:
                                page_name = h1.find('span').text
                                norm_page_name = unicodedata.normalize('NFD', page_name).encode('ascii', 'ignore').decode("utf-8")
                                if norm_page_name.lower()==normalized_name.lower():

                                    return suffix

def get_repeat_player_suffix(name):
    ''' Get repeat player suffix. Might want to add in other params to deal with 
    the repeat players who are still ambiguous such as :
    Chris Johnson (x2), Chris Wright (x2), Gary Payton (maybe not since the 
    older version stoppped playing in 2007, Marcus Williams (x2), Mike James (x2), 
    Tony Mitchell (x2)'''

    ## Note that this function assumes no repeat players have accent marks 
    ## in their names 

    if name.lower() in repeat_players_dict['duplicated']:
        ## We must actually address this situation differently or figure solution
        return repeat_players_dict['duplicated'][name.lower()]

    return repeat_players_dict['unique'][name.lower()]

    
    return repeat_players[name]



repeat_players_dict = {'unique': {'dee brown': '/players/b/brownde03.html',
 'bobby jones': '/players/j/jonesbo02.html',
 'chris smith': '/players/s/smithch05.html',
 'david lee': '/players/l/leeda02.html',
 'gary payton': '/players/p/paytoga02.html',
 'gary trent': '/players/t/trentga02.html',
 'george king': '/players/k/kingge03.html',
 'gerald henderson': '/players/h/hendege02.html',
 'glen rice': '/players/r/ricegl02.html',
 'glenn robinson': '/players/r/robingl02.html',
 'greg smith': '/players/s/smithgr02.html',
 'jaren jackson': '/players/j/jacksja02.html',
 'kevin porter': '/players/p/porteke02.html',
 'larry drew': '/players/d/drewla02.html',
 'larry nance': '/players/n/nancela02.html',
 'luke jackson': '/players/j/jackslu02.html',
 'mike dunleavy': '/players/d/dunlemi02.html',
 'patrick ewing': '/players/e/ewingpa02.html',
 'reggie williams': '/players/w/willire02.html',
 'tim hardaway': '/players/h/hardati02.html',
 'walker russell': '/players/r/russewa02.html'},
 
 'duplicated' : {'chris johnson': '/players/j/johnsch04.html',
 'chris wright': '/players/w/wrighch02.html',
 'marcus williams': '/players/w/willima04.html',
 'mike james': '/players/j/jamesmi02.html',
 'tony mitchell': '/players/m/mitchto02.html',
 }}

repeat_players_list = ['dee brown', 'bobby jones', 'chris johnson', 
                       'chris smith', 'chris wright', 'david lee', 
                       'gary payton', 'gary trent', 'george king', 
                       'gerald henderson', 'glen rice', 'glenn robinson', 
                       'greg smith', 'jaren jackson', 'kevin porter', 
                       'larry drew', 'larry nance', 'luke jackson', 
                       'marcus williams', 'mike dunleavy', 'mike james', 
                       'patrick ewing', 'reggie williams', 'tim hardaway', 
                       'tony mitchell', 'walker russell']


