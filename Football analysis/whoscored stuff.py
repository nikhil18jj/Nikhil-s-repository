# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:23:58 2020

@author: nikhi
"""
from selenium import webdriver 
import time 
from bs4 import BeautifulSoup
import pandas as pd
import os; 
import sys;
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException 
import time
import json
from matplotlib.patches import Ellipse
import matplotlib.patheffects as path_effects
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi

from shapely.ops import cascaded_union
from shapely.geometry import Polygon, MultiPoint
from shapely import affinity

options = Options()
options.add_argument("start-maximized")
options.add_argument("disable-infobars")
options.add_argument("--disable-extensions")
options.headless=True
urls = ["https://1xbet.whoscored.com/Matches/1492112/Live/Spain-LaLiga-2020-2021-Granada-Barcelona"]

driver = webdriver.Chrome() 
def check_exists_by_xpath():
    try:
        driver.find_element_by_xpath("//script[contains(.,'matchCentreData')]")
    except NoSuchElementException:
        return False
    return True

# for i in range(len(urls)):
website_URL = urls[0]
driver.get(website_URL)
if check_exists_by_xpath():
    matchdict = driver.execute_script("return matchCentreData;")
    matchdict['matchId'] = driver.execute_script("return matchId;")
    with open(str(matchdict['matchId'])+'.json', 'w') as fp:
        json.dump(matchdict, fp, sort_keys=True, indent=4)

def filereader(filename):
   
    with open('1492112.json',encoding="unicode-escape") as f:
        matchdict=json.loads(f.read())
   

  
   
    match=pd.json_normalize(matchdict['events'],sep="_")
    hometeam=matchdict['home']['name']
    awayteam=matchdict['away']['name']
    homeid=matchdict['home']['teamId']
    awayid=matchdict['away']['teamId']
    players=pd.DataFrame()
    homepl=pd.json_normalize(matchdict['home']['players'],sep='_')[['name','position','shirtNo','playerId']]
    #[['name','position','shirtNo','playerId']]
    awaypl=pd.json_normalize(matchdict['away']['players'] ,sep='_')[['name','position','shirtNo','playerId']] 
    #[['name','position','shirtNo','playerId']]  
    players=players.append(homepl)
    players=players.append(awaypl) 
    
    #match.dtypes  
    #players.dtypes     
    match=match.merge(players,how='left')#match=match.append(players)
    #match=pd.concat([match,players])
    homedf=match[match.teamId==homeid].reset_index(drop=True)
    awaydf=match[match.teamId==awayid].reset_index(drop=True) 
    homedf['receiver']=np.where((homedf.type_displayName=='Pass')&(homedf.outcomeType_displayName=="Successful"),
                             homedf.name.shift(-1),'').tolist()
    awaydf['receiver']=np.where((awaydf.type_displayName=='Pass')&(awaydf.outcomeType_displayName=="Successful"),
                             awaydf.name.shift(-1),'').tolist()
    match['receiver']=['' for _ in range(len(match))]
    match.loc[match.teamId==homeid,'receiver']=homedf['receiver'].tolist()
    match.loc[match.teamId==awayid,'receiver']=awaydf['receiver'].tolist()
    match['gameid']=[matchdict['matchId'] for i in range(len(match))]
        
        
    return match,matchdict
match,matchdict=filereader('1492112.json')
min_dribble_length=3
max_dribble_length=60
max_dribble_duration=10
match['second'] = match['second'].fillna(value=0)
match['time_seconds'] = match['expandedMinute']*60+match['second']   
        

def add_dribbles(actions):
    
    next_actions=actions.shift(-1)
    same_team=actions.teamId == next_actions.teamId
    dx=actions.endX-next_actions.x
    dy=actions.endY-next_actions.y
    far_enough=dx**2+dy**2>= min_dribble_length**2
    not_too_far=dx**2+ dy**2 <= max_dribble_length **2
    
    dt=next_actions.time_seconds-actions.time_seconds
    same_phase=dt<max_dribble_duration
    same_period=actions.period_value == next_actions.period_value
    dribble_idx=same_team & far_enough & not_too_far & same_phase & same_period
    dribbles=pd.DataFrame()
    prev=actions[dribble_idx]
    nex=next_actions[dribble_idx]
    dribbles["gameid"]=nex.gameid
    dribbles["period_value"]=nex.period_value
    dribbles["id"]=prev.id+0.1
    dribbles["time_seconds"]=(prev.time_seconds+nex.time_seconds)/2
    dribbles["teamId"]=nex.teamId
    dribbles["playerId"]=nex.playerId
    dribbles['name']=nex.name
    dribbles["receiver"]=''
    dribbles["x"]=prev.endX
    dribbles["y"]=prev.endY
    dribbles["endX"]=nex.x
    dribbles["endY"]=nex.y
    dribbles["type_displayName"]=['Carry' for _ in range(len(dribbles))] 
    dribbles["outcomeType_displayName"]=['Successful' for _ in range(len(dribbles))]
    actions=pd.concat([actions,dribbles],ignore_index=True,sort=False)
    actions=actions.sort_values(["gameid","period_value","id"]).reset_index(drop=True)
    actions["id"]=range(len(actions))
    
    return actions
    
match=add_dribbles(match)     
barcamoves = match.query("(teamId==65)&(type_displayName in ['Pass','Carry'])&(outcomeType_displayName == 'Successful')")   
TYPE = "xT" ##xT/EPV
if TYPE == "xT":    
    with open("xT.json", "r") as f:
        xtd = json.load(f) 
    xtd = np.array(xtd)
elif TYPE == "EPV":
    xtd = pd.read_csv("epv_grid.csv").to_numpy()

#with open("xT.json", "r") as f:
#    xtd = json.load(f)

#xtd = np.array(xtd) 
n_rows,n_cols=xtd.shape

barcamoves['x_bin'] = barcamoves.x.apply(lambda val: int(val/(100/n_cols)) if val != 100 else int(val/(100/n_cols)) - 1 )
#pass_df.x1.apply(lambda val: int(val/(100/n_cols)) if val != 100 else int(val/(100/n_cols)) - 1 )
barcamoves['endX_bin'] = barcamoves.endX.apply(lambda val: int(val/(100/n_cols)) if val != 100 else int(val/(100/n_cols)) - 1 )

barcamoves['y_bin'] = barcamoves.y.apply(lambda val: int(val/(100/n_rows)) if val != 100 else int(val/(100/n_rows)) - 1 )
barcamoves['endY_bin'] = barcamoves.endY.apply(lambda val: int(val/(100/n_rows)) if val != 100 else int(val/(100/n_rows)) - 1 )
#%time
barcamoves['start_zone_value'] = barcamoves[['x_bin', 'y_bin']].apply(lambda x: xtd[x[1]][x[0]], axis=1)
barcamoves['end_zone_value'] = barcamoves[['endX_bin', 'endY_bin']].apply(lambda x: xtd[x[1]][x[0]], axis=1)
barcamoves['xT_value'] = barcamoves['end_zone_value'] - barcamoves['start_zone_value'] ##value of any pass is just value at end_zone - value at start_zone
barcamoves = barcamoves[[col for col in barcamoves.columns if 'bin' not in col]] ##remove the bins/indices since they're useless now
 

xT_Players=barcamoves.groupby('name') [['xT_value','end_zone_value','start_zone_value']].sum().sort_values('xT_value',ascending=False).round(4)
#barcamoves=barcamoves.merge(xT_Players, on=['playerId'], how='left', validate='m:1')
from mplsoccer.pitch import Pitch
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

#barcamoves[barcamoves['xT_value']>0 & barcamoves['xT_value']<=0.05)]=np.nan
#barcamoves[barcamoves['xT_value']>=-0.05]=np.nan

# setup pitch
pitch = Pitch(pitch_type='statsbomb', figsize=(20, 12), line_zorder=2,pitch_color='black', line_color='white', orientation='horizontal')    
fig, ax = pitch.draw()
plt.style.use('dark_background')
#bin_statistic = pitch.bin_statistic(barcamoves['x'], barcamoves['y'], statistic='sum',values=barcamoves['xT_value'].round(4),bins=(8,12))
xstart=[]
ystart=[]
xstart=barcamoves['x'].values
ystart=100-barcamoves['y'].values
bin_statistic = pitch.bin_statistic(xstart,ystart, statistic='sum',values=barcamoves['xT_value'].round(4),bins=(8,12))
bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 0)
pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b',vmin=0)
cbar = fig.colorbar(pcm, ax=ax)
ax.set_title('Barcelona xT locations vs Granada')

xT_Players['xT_value'].plot(kind='barh')
plt.title("xT values for Barca players vs Granada")
plt.xlabel("Players")
plt.ylabel("xT")



epv = pd.read_csv("epv_grid.csv")
epv = np.array(epv)
n_rows,n_cols=epv.shape

barcamoves['x_bin'] = barcamoves.x.apply(lambda val: int(val/(100/n_cols)) if val != 100 else int(val/(100/n_cols)) - 1 )
#pass_df.x1.apply(lambda val: int(val/(100/n_cols)) if val != 100 else int(val/(100/n_cols)) - 1 )
barcamoves['endX_bin'] = barcamoves.endX.apply(lambda val: int(val/(100/n_cols)) if val != 100 else int(val/(100/n_cols)) - 1 )

barcamoves['y_bin'] = barcamoves.y.apply(lambda val: int(val/(100/n_rows)) if val != 100 else int(val/(100/n_rows)) - 1 )
barcamoves['endY_bin'] = barcamoves.endY.apply(lambda val: int(val/(100/n_rows)) if val != 100 else int(val/(100/n_rows)) - 1 )
#%time
barcamoves['start_zone_value'] = barcamoves[['x_bin', 'y_bin']].apply(lambda x: epv[x[1]][x[0]], axis=1)
barcamoves['end_zone_value'] = barcamoves[['endX_bin', 'endY_bin']].apply(lambda x: epv[x[1]][x[0]], axis=1)
barcamoves['epv_value'] = barcamoves['end_zone_value'] - barcamoves['start_zone_value'] ##value of any pass is just value at end_zone - value at start_zone
barcamoves = barcamoves[[col for col in barcamoves.columns if 'bin' not in col]] ##remove the bins/indices since they're useless now


          

epv_Players=barcamoves.groupby('name') [['epv_value','end_zone_value','start_zone_value']].sum().sort_values('epv_value',ascending=False).round(4)
'''
passes_df = barcamoves.loc[barcamoves['teamId'] == 65]

passes_df = passes_df.loc[passes_df['type_displayName']=='Pass']
passes_df=passes_df.loc[passes_df['outcomeType_displayName']=="Successful"]
#passes_df.insert(29, column='passRecipientName', value=passes_df['name'].shift(-1))  
passes_df.dropna(subset=["receiver"], inplace=True)
#passes_df = passes_df[passes_df.columns[~passes_df.isnull().all()]]




passes_df = passes_df[passes_df['position'] != 'Sub']

passes_df['playerKitNumber']=passes_df['shirtNo'].astype(np.int)
passes_df['playerKitNumberReceipt']=passes_df['shirtNo'].shift(-1).fillna(0).astype(np.int)

passes_formation = passes_df[['id', 'playerKitNumber', 'playerKitNumberReceipt']].copy()
location_formation = passes_df[['playerKitNumber', 'x', 'y']]

average_locs_and_count = location_formation.groupby('playerKitNumber').agg({'x': ['mean'], 'y': ['mean', 'count']})
average_locs_and_count.columns = ['x', 'y', 'count']



passes_formation['kitNo_max'] = passes_formation[['playerKitNumber',
                                                'playerKitNumberReceipt']].max(axis='columns')
passes_formation['kitNo_min'] = passes_formation[['playerKitNumber',
                                                'playerKitNumberReceipt']].min(axis='columns')


passes_between = passes_formation.groupby(['kitNo_max', 'kitNo_min']).id.count().reset_index()
passes_between.rename({'id': 'pass_count'}, axis='columns', inplace=True)

# add on the location of each player so we have the start and end positions of the lines
passes_between = passes_between.merge(average_locs_and_count, left_on='kitNo_min', right_index=True)
passes_between = passes_between.merge(average_locs_and_count, left_on='kitNo_max', right_index=True,
                                      suffixes=['', '_end'])

##############################################################################
# Calculate the line width and marker sizes relative to the largest counts

max_line_width = 18
passes_between['width'] = passes_between.pass_count / passes_between.pass_count.max() * max_line_width
#average_locs_and_count['marker_size'] = (average_locs_and_count['count']
#                                         / average_locs_and_count['count'].max() * max_marker_size)
marker_color='#6a009c'
marker_size=2000
pitch_color='#000000'
##############################################################################
# Set color to make the lines more transparent when fewer passes are made

min_transparency = 0.3
color = np.array(to_rgba('white'))
color = np.tile(color, (len(passes_between), 1))
c_transparency = passes_between.pass_count / passes_between.pass_count.max()
c_transparency = (c_transparency * (1 - min_transparency)) + min_transparency
color[:, 3] = c_transparency

##############################################################################
# Plotting


pitch = Pitch(pitch_type='statsbomb', orientation='horizontal',
              pitch_color='#000000', line_color='#c7d5cc', figsize=(16, 11),
              constrained_layout=True, tight_layout=False)
fig, ax = pitch.draw()

pitch.lines(passes_between.x/100*120, 80-passes_between.y/100*80,
            passes_between.x_end/100*120, 80-passes_between.y_end/100*80, lw=passes_between.width,
            color=color, zorder=1, ax=ax)
pitch.scatter(average_locs_and_count.x/100*120, 80-average_locs_and_count.y/100*80, s=marker_size,
              color=marker_color, edgecolors='black', linewidth=1, alpha=1, ax=ax)

for index, row in average_locs_and_count.iterrows():
    pitch.annotate(row.name, xy=(row.x/100*120, 80-row.y/100*80), c='white', va='center', ha='center', size=20, weight='bold', ax=ax)
ax.set_title("Pass Network", size=15, y=0.97, color='#c7d5cc')
fig.set_facecolor(pitch_color)

'''

passes_df = barcamoves.loc[barcamoves['teamId'] == 65]

passes_df = passes_df[passes_df['position'] != 'Sub']
#passes_df['passer'] = passes_df['playerId']
#passes_df['recipient'] = passes_df['passer'].shift(-1)
passes_df = passes_df[passes_df.columns[~passes_df.isnull().all()]]
passes_df['playerKitNumber']=passes_df['shirtNo'].fillna(0).astype(np.int)
passes_df['playerKitNumberReceipt']=passes_df['shirtNo'].shift(-1).fillna(0).astype(np.int)

passes_df = passes_df.loc[passes_df['type_displayName']=='Pass']
passes_df=passes_df.loc[passes_df['outcomeType_displayName']=="Successful"]
#passes_df.insert(29, column='passRecipientName', value=passes_df['name'].shift(-1))  
passes_df.dropna(subset=["receiver"], inplace=True)

passer_avg = passes_df.groupby('playerKitNumber').agg({'x': ['median'], 'y': ['median','count'],'epv_value':['sum']})

passer_avg.columns = ['x', 'y', 'count','epv']
passer_avg.index = passer_avg.index.astype(int)
passes_formation = passes_df[['id', 'playerKitNumber', 'playerKitNumberReceipt']].copy()

passes_formation['kitNo_max'] = passes_formation[['playerKitNumber',
                                                'playerKitNumberReceipt']].max(axis='columns')
passes_formation['kitNo_min'] = passes_formation[['playerKitNumber',
                                                'playerKitNumberReceipt']].min(axis='columns')

passes_between = passes_formation.groupby(['kitNo_max', 'kitNo_min']).id.count().reset_index()
passes_between.rename({'id': 'pass_count'}, axis='columns', inplace=True)

# add on the location of each player so we have the start and end positions of the lines
passes_between = passes_between.merge(passer_avg, left_on='kitNo_min', right_index=True)
passes_between = passes_between.merge(passer_avg, left_on='kitNo_max', right_index=True,
                                      suffixes=['', '_end'])
'''
#Between Passer and Recipient
passes_between = passes_df.groupby(['passer', 'recipient']).id.count().reset_index()
passes_between.rename({'id': 'pass_count'}, axis='columns', inplace=True)

passes_between = passes_between.merge(passer_avg, left_on='passer', right_index=True)
passes_between = passes_between.merge(passer_avg, left_on='recipient', right_index=True,
                                      suffixes=['', '_end'])
'''
#Minimum No. of Passes
passes_between = passes_between.loc[(passes_between['pass_count']>=3)]

#Scaling for StatsBomb
passes_between['x']=passes_between['x']*1.2
passes_between['y']=passes_between['y']*0.8
passer_avg['x']=passer_avg['x']*1.2
passer_avg['y']=passer_avg['y']*0.8
passes_between['x_end']=passes_between['x_end']*1.2
passes_between['y_end']=passes_between['y_end']*0.8

#Width Variable
yo = passes_between.pass_count / passes_between.pass_count.max()

#WITHIN .PITCH THERE IS NO NEED TO FLIP X AND Y

#Plot Pitch
fig, ax = plt.subplots()
fig.set_facecolor('#F8F8FF')
fig.patch.set_facecolor('#F8F8FF')

pitch = Pitch(pitch_type='statsbomb', orientation='vertical',
              pitch_color='#F8F8FF', line_color='#132257',
              constrained_layout=True, tight_layout=False,
              linewidth=0.5)

pitch.draw(ax=ax)
'''
pitch = Pitch(pitch_type='statsbomb', orientation='horizontal',
              pitch_color='#000000', line_color='#c7d5cc', figsize=(16, 11),
              constrained_layout=True, tight_layout=False)
pitch.draw(ax=ax)
'''
b = passer_avg.epv
min_transparency = 0.3
color = np.array(to_rgba('#132257'))
color = np.tile(color, (len(passes_between), 1))
c_transparency = passes_between.pass_count / passes_between.pass_count.max()
c_transparency = (c_transparency * (1 - min_transparency)) + min_transparency
color[:, 3] = c_transparency
a = plt.scatter(passer_avg.y, passer_avg.x, s=100,c=b,facecolor='none',lw=1,
                cmap="cool", alpha=1, zorder=2, vmin=0 ,vmax=0.6, marker='h')
c = plt.scatter(passer_avg.y, passer_avg.x, s=60,c='#FFA500',
                alpha=1, zorder=3, marker='h')
pitch.arrows(passes_between.x, passes_between.y, passes_between.x_end, 
            passes_between.y_end, color=color, ax=ax, zorder=1, width=1.5)
cbar = plt.colorbar(a, orientation="horizontal",shrink=0.3, pad=0,
             ticks=[0, 0.2, 0.4, 0.6])
cbar.set_label('Expected Possession Value (EPV)', color='#132257', size=6)
cbar.outline.set_edgecolor('#132257')
cbar.ax.xaxis.set_tick_params(color='#132257')
cbar.ax.xaxis.set_tick_params(labelcolor='#132257')
cbar.ax.tick_params(labelsize=5)
plt.gca().invert_xaxis()
for index, row in passer_avg.iterrows():
    pitch.annotate(row.name, xy=(row.x, row.y), 
                   c='#132257', va='center', ha='center', size=5, ax=ax)
plt.text(79,2,"Positions = Median Location of Successful Passes\nArrows = Pass Direction\nTransparency = Frequency of Combination\nMinimum of 3 Passes ", color='#132257',
               fontsize=5, alpha=0.5, zorder=1)
plt.text(80,122,"Minutes 0-65 (First Substitution)", color='#132257',
               fontsize=5)
plt.text(18,122,"@nikhilrajesh231", color='#132257', fontsize=5)
ax.set_title("Barcelona PV Pass Network\n4-0 vs Granada (A)", 
             fontsize=8, color="#132257", fontweight = 'bold', y=1.01)
plt.savefig('pn.png',bbox_inches="tight",facecolor="#F8F8FF",dpi=500)


epv_Players['epv_value'].plot(kind='barh')
plt.title("PV values for Barca players vs Granada")
plt.xlabel("Players")
plt.ylabel("PV")


passact=match.query("(teamId==65)&(type_displayName in ['Pass'])&(outcomeType_displayName == 'Successful')")   
caract=match.query("(teamId==65)&(type_displayName in ['Carry'])&(outcomeType_displayName == 'Successful')")   
def binnings(df):
    TYPE=input("Enter the metric")
    if TYPE == "xT":    
        with open("xT.json", "r") as f:
            xtd = json.load(f) 
        xtd = np.array(xtd)
        n_rows,n_cols=xtd.shape
    elif TYPE == "EPV":
        xtd = pd.read_csv("epv_grid.csv")
        xtd=np.array(xtd)
        n_rows,n_cols=xtd.shape
    df['x_bin'] =df.x.apply(lambda val: int(val/(100/n_cols)) if val != 100 else int(val/(100/n_cols)) - 1 )
#pass_df.x1.apply(lambda val: int(val/(100/n_cols)) if val != 100 else int(val/(100/n_cols)) - 1 )
    df['endX_bin'] = df.endX.apply(lambda val: int(val/(100/n_cols)) if val != 100 else int(val/(100/n_cols)) - 1 )
    
    df['y_bin'] = df.y.apply(lambda val: int(val/(100/n_rows)) if val != 100 else int(val/(100/n_rows)) - 1 )
    df['endY_bin'] = df.endY.apply(lambda val: int(val/(100/n_rows)) if val != 100 else int(val/(100/n_rows)) - 1 )
    #%time
    df['start_zone_value'] = df[['x_bin', 'y_bin']].apply(lambda x: xtd[x[1]][x[0]], axis=1)
    df['end_zone_value'] = df[['endX_bin', 'endY_bin']].apply(lambda x: xtd[x[1]][x[0]], axis=1)
    df['epv_value'] = df['end_zone_value'] - df['start_zone_value'] ##value of any pass is just value at end_zone - value at start_zone
    df= df[[col for col in df.columns if 'bin' not in col]] ##remove the bins/indices since they're useless now
    epv_Players=df.groupby('name') [['epv_value','end_zone_value','start_zone_value']].sum().sort_values('epv_value',ascending=False).round(4)
    return epv_Players
PV_passes=pd.DataFrame()
PV_passes=binnings(passact)
PV_carries=pd.DataFrame()
PV_carries=binnings(caract)