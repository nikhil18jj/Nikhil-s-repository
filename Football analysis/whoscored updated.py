# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:09:53 2021

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
from scipy import stats
from highlight_text import fig_text
from tqdm import tqdm
from mplsoccer.pitch import Pitch
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib.pyplot as plt

options = Options()
options.add_argument("start-maximized")
options.add_argument("disable-infobars")
options.add_argument("--disable-extensions")
options.headless=True
folder="C:\\Users\nikhi\\"
driver=webdriver.Chrome()
website_URL="https://1xbet.whoscored.com/Matches/1492169/Live/Spain-LaLiga-2020-2021-Barcelona-Getafe"
driver.get(website_URL)
element=driver.find_element_by_xpath('//*[@id="layout-wrapper"]/script[1]')
script_content=element.get_attribute('innerHTML')
script_ls=script_content.split(sep="  ")
script_ls=list(filter(None,script_ls))
script_ls=[name for name in script_ls if name.strip()]
dictstring=script_ls[2][17:-2]
matchdict=json.loads(dictstring)
matchdict["id"]=script_ls[1][8:-2]
with open(str(matchdict["id"])+'.json','w') as fp:
    json.dump(matchdict, fp, sort_keys=True, indent=4)

        

def filereader(filename):
   
    with open('1492169.json',encoding="unicode-escape") as f:
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
    match['gameid']=[matchdict['id'] for i in range(len(match))]
        
        
    return match,matchdict
match,matchdict=filereader('1492169.json')
min_dribble_length=6.0
max_dribble_length=100.0
max_dribble_duration=20.0
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



def passmap(Df,teamid,teamname,min1,max1):
	pitch = Pitch(pitch_type='statsbomb', orientation='vertical',
	  pitch_color='#000000', line_color='#a9a9a9',
	  constrained_layout=True, tight_layout=False,
	  linewidth=0.5)

	fig, ax = pitch.draw()
	df = Df.copy()
	df = df[(df.expandedMinute>=min1)&(df.expandedMinute<=max1)]    
	allplayers = df[(df.teamId==teamid)&(df.name.notna())].name.tolist()
	playersubbedoff = df[(df.type_displayName == 'SubstitutionOff')&(df.teamId==teamid)]['name'].tolist()
	timeoff = df[(df.type_displayName == 'SubstitutionOff')&(df.teamId==teamid)]['expandedMinute'].tolist()
	playersubbedon = df[(df.type_displayName == 'SubstitutionOn')&(df.teamId==teamid)]['name'].tolist()
	timeon = df[(df.type_displayName == 'SubstitutionOn')&(df.teamId==teamid)]['expandedMinute'].tolist()
	majoritylist = []
	minoritylist = []
	for i in range(len(timeon)):
		if((timeon[i]>=min1)&(timeon[i]<=max1)):
			player1min = timeon[i] - min1
			player2min = max1 - timeon[i]
			if(player1min >= player2min):
				majoritylist.append(playersubbedoff[i])
				minoritylist.append(playersubbedon[i])
			else:
				majoritylist.append(playersubbedon[i])
				minoritylist.append(playersubbedoff[i])
	players = list(set(allplayers) - set(minoritylist))
	#return players
   
	shirtNo = []
	for p in players:
		shirtNo.append(int(df[df.name==p]['shirtNo'].values[0]))
	passes_df = df.query("(type_displayName=='Pass')&(name in @players)&(receiver in @players)&\
						 (outcomeType_displayName == 'Successful')&(teamId==@teamid)")
	#passes_df.insert(29, column='passRecipientName', value=passes_df['name'].shift(-1))  
	passes_df.dropna(subset=["receiver"], inplace=True)
	
	#passes_df['passer'] = passes_df['playerId']
	#passes_df['recipient'] = passes_df['passer'].shift(-1)
	passes_df = passes_df[passes_df.columns[~passes_df.isnull().all()]]
	passes_df['playerKitNumber']=passes_df['shirtNo'].fillna(0).astype(np.int)
	passes_df['playerKitNumberReceipt']=passes_df['shirtNo'].shift(-1).fillna(0).astype(np.int)
	
	
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
	yo= passes_between.pass_count / passes_between.pass_count.max()
	b = 1
	min_transparency = 0.3
	color = np.array(to_rgba('#00bfff'))
	color = np.tile(color, (len(passes_between), 1))
	c_transparency = passes_between.pass_count / passes_between.pass_count.max()
	c_transparency = (c_transparency * (1 - min_transparency)) + min_transparency
	color[:, 3] = c_transparency
	a = plt.scatter(passer_avg.y, passer_avg.x, s=100,c="#111111",facecolor='none',lw=1,
					cmap="winter", alpha=1, zorder=2, vmin=0 ,vmax=0.6, marker='h')
	c = plt.scatter(passer_avg.y, passer_avg.x, s=60,c='#FF0000',
					alpha=1, zorder=3, marker='h')
	pitch.arrows(passes_between.x, passes_between.y, passes_between.x_end, 
				passes_between.y_end, color=color, ax=ax, zorder=1, width=1.5)
	cbar = plt.colorbar(a, orientation="horizontal",shrink=0.3, pad=0,
				 ticks=[0, 0.2, 0.4, 0.6])
	cbar.set_label('Expected Possession Value (EPV)', color='#a9a9a9', size=6)
	cbar.outline.set_edgecolor('#a9a9a9')
	cbar.ax.xaxis.set_tick_params(color='#a9a9a9')
	cbar.ax.xaxis.set_tick_params(labelcolor='#a9a9a9')
	cbar.ax.tick_params(labelsize=5)
	plt.gca().invert_xaxis()
	
	for index, row in passer_avg.iterrows():
		pitch.annotate(row.name, xy=(row.x, row.y), 
					   c='#a9a9a9', va='center', ha='center', size=5, ax=ax)
	plt.text(79,2,"Positions = Median Location of Successful Passes\nArrows = Pass Direction\nTransparency = Frequency of Combination\nMinimum of 3 Passes ", color='#a9a9a9',
				   fontsize=5, alpha=0.5, zorder=1)
	plt.text(80,122,"Minutes 45-90", color='#a9a9a9',
				   fontsize=5)
	plt.text(18,122,"@nikhilrajesh231", color='#a9a9a9', fontsize=5)
	ax.set_title("Barcelona PV Pass Network\n5-2 vs Getafe (H)", 
				 fontsize=8, color="#a9a9a9", fontweight = 'bold', y=1.01)
	fig=plt.savefig('pn2.png',bbox_inches="tight",facecolor="#000000",dpi=600)
	return fig

players = passmap(barcamoves,65,"Barcelona",0,45)
#players1= passmap(barcamoves,65,"Barcelona",45,64)

##### Define individual heatmaps
def ind_heatmap(Df,player):
 
    from matplotlib.colors import LinearSegmentedColormap
    import cmasher as cmr
    #from mplsoccer import VerticalPitch
    df = Df.copy()
    pitch = Pitch(pitch_type='uefa', figsize=(10.5,6.8), line_zorder=2, 
                  line_color='#636363', orientation='horizontal',
                  constrained_layout=True,tight_layout=False,pitch_color='black')
    fig, ax = pitch.draw()
    df = df[df.name==player]
    touchdf = df[(df.isTouch==True)&(df.name==player)].reset_index()
    #pitch.kdeplot(touchdf.x, touchdf.y, ax=ax, cmap=cmap,
                  #linewidths=0.3,fill=True,levels=1000)
    pitch.kdeplot(touchdf.x, touchdf.y, ax=ax, cmap=cmr.voltage, shade=True,levels=1000)
    ax.set_title(player+' Touch-based heatmap',fontsize=25,color='white')
    fig.text(0.2, 0.0, "Created by Nikhil Rajesh / @nikhilrajesh231",
             fontstyle="italic",fontsize=15,color='black')
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    return fig
heatmap=ind_heatmap(barcamoves,"Lionel Messi")
heatmap1=ind_heatmap(barcamoves,"Frenkie de Jong")
barcamoves.name.unique()

#ind_heatmap()
# setup pitch

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
pitch = Pitch(pitch_type='uefa', figsize=(6.8, 10.5), line_zorder=2,
              line_color='white', orientation='vertical')
# draw
fig, ax = pitch.draw()
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
cmaplist = ['#082630', '#0682fe', "#eff3ff"]
cmap = LinearSegmentedColormap.from_list("", cmaplist)
bin_statistic = pitch.bin_statistic(barcamoves.x, barcamoves.y, values = barcamoves.xT_value, statistic='sum', bins=(38,25))
bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
vm = bin_statistic['statistic'].min()
vma = bin_statistic['statistic'].max()
pitch.heatmap(bin_statistic, ax=ax, cmap='inferno', edgecolors=None, vmin = bin_statistic['statistic'].min(),
              vmax = bin_statistic['statistic'].max())
ax.set_title('Barcelona'+'\n'+'Open-play Threat-generation hotspots',fontsize=25)
fig.set_facecolor('white')  
plt.savefig('xt.png',dpi=600)  



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
    return df

passact=match.query("(teamId==65)&(type_displayName in ['Pass'])&(outcomeType_displayName == 'Successful')")   
caract=match.query("(teamId==65)&(type_displayName in ['Carry'])&(outcomeType_displayName == 'Successful')")   

PV_passes=binnings(passact)
PV_carries=binnings(caract)

PV_players_passes=PV_passes.groupby('name')[['epv_value']].agg('sum')
PV_players_passes['name']=PV_players_passes.index
PV_players_passes['pass_epv']=PV_players_passes['epv_value'].tolist()
PV_players_passes=PV_players_passes.reset_index(drop=True)
PV_players_passes=PV_players_passes.drop(['epv_value'],axis=1)
PV_players_passes=PV_players_passes.sort_values(by='name')


PV_players_carries=PV_carries.groupby('name')[['epv_value']].agg('sum')
PV_players_carries['name']=PV_players_carries.index
PV_players_carries['carry_epv']=PV_players_carries['epv_value'].tolist()
PV_players_carries=PV_players_carries.reset_index(drop=True)
PV_players_carries=PV_players_carries.drop(['epv_value'],axis=1)
PV_players_carries=PV_players_carries.sort_values(by='name')

PV_combo=PV_players_passes.merge(PV_players_carries,how='left')
PV_combo['total PV']=PV_combo['pass_epv']+PV_combo['carry_epv']
PV_combo=PV_combo.sort_values('total PV',ascending=False)
names=PV_combo['name'].tolist()
passpv=PV_combo['pass_epv'].tolist()
carrypv=PV_combo['carry_epv'].tolist()
totalpv=PV_combo['total PV'].tolist()
width=0.35
fig, ax = plt.subplots()

ax.barh(names, passpv, width, label='Pass')
ax.barh(names, carrypv, width,
       label='Carry',left=passpv)

ax.set_ylabel('PV for players')
ax.set_title('PV for each player vs Getafe')
ax.legend()
plt.savefig('pv.png',bbox_inches="tight",dpi=500)


'''
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
    df['xt_value'] = df['end_zone_value'] - df['start_zone_value'] ##value of any pass is just value at end_zone - value at start_zone
    df= df[[col for col in df.columns if 'bin' not in col]] ##remove the bins/indices since they're useless now
    return df

xt_actions=binnings(barcamoves)
def xTplotter(Df):
    df = Df.copy()
    hometeamid = 65
    awayteamid = 819
    hometeam = "Barcelona"
    awayteam = "Getafe"
    homedf = df.query("(teamId==@hometeamid)&(events in ['Pass','Carry'])&\
                            (outcomeType_displayName in 'Successful')")
    awaydf = df.query("(teamId==@awayteamid)&(events in ['Pass','Carry'])&\
                            (outcomeType_displayName in 'Successful')")
    homemoves = binnings(homedf,f).reset_index(drop=True)
    awaymoves = binnings(awaydf,f).reset_index(drop=True)
    homemoves['xt_cumu'] = homemoves.xt_value.cumsum()
    awaymoves['xt_cumu'] = awaymoves.xt_value.cumsum()
    from scipy.ndimage.filters import gaussian_filter1d
    homexTlist = [homemoves.query("(time_seconds>=300*@i)&(time_seconds<=300*(@i+1))").\
            xt_value.sum() for i in range(round(df.time_seconds.max()/60//15)+1)]
    awayxTlist = [awaymoves.query("(time_seconds>=300*@i)&(time_seconds<=300*(@i+1))").\
            xt_value.sum() for i in range(round(df.time_seconds.max()/60//15)+1)]
    homexTlist = gaussian_filter1d(homexTlist, sigma=1)
    awayxTlist = gaussian_filter1d(awayxTlist, sigma=1)
    timelist = [5*i for i in range(round(df.time_seconds.max()/60//5)+1)]
    difflist = [homexTlist[i] - awayxTlist[i] for i in range(len(homexTlist))]
    fig,ax = plt.subplots(1,2, figsize=(15,5))
    fig.set_facecolor('#000a0d')
    ax[0].set_facecolor('#000a0d')
    ax[1].set_facecolor('#000a0d')
    hc = '#d7191c'
    ac = '#ffffbf'
    ax[0].plot(timelist,np.cumsum(homexTlist),hc,timelist, np.cumsum(awayxTlist),ac)
    ax[0].set_ylabel("Cumulative xT",fontsize=15,color='#edece9')
    ax[0].set_xlabel("Time intervals - every 15 minutes",fontsize=15,color='#edece9')
    ax[0].yaxis.label.set_color('#e0dfdc')
    ax[0].tick_params(axis='y', colors='#e0dfdc')
    # ax[0].fill_between(timelist,np.cumsum(homexTlist),color=hc,alpha=0.3)
    n_lines = 10
    diff_linewidth = 1.05
    alpha_value = 0.1
    for n in range(1, n_lines+1):
        ax[0].plot(timelist,np.cumsum(homexTlist),c=hc,
                linewidth=2+(diff_linewidth*n),
                alpha=alpha_value)
    # ax[0].fill_between(timelist,np.cumsum(oppoxTlist),color=hc,alpha=0.3)
    n_lines = 10
    diff_linewidth = 1.05
    alpha_value = 0.1
    for n in range(1, n_lines+1):
        ax[0].plot(timelist,np.cumsum(awayxTlist),c=ac,
                linewidth=2+(diff_linewidth*n),
                alpha=alpha_value)
    homegoaltimes = df[(df.period_value<=4)&(df.teamId==df.hometeamid)&
                (df.events.isin(['Shot','Freekick','Penalty']))&
            (df.outcome=='Successful')].expandedMinute.tolist() + df[(df.period_value<=4)&
                    (df.teamId==df.awayteamid)&
                    (df.outcome=='OwnGoal')].expandedMinute.tolist()
    awaygoaltimes = df[(df.period_value<=4)&(df.teamId==df.awayteamid)&
                    (df.events.isin(['Shot','Freekick','Penalty']))&
        (df.outcome=='Successful')].expandedMinute.tolist() + df[(df.period_value<=4)&
                    (df.teamId==df.hometeamid)&
                    (df.outcome=='OwnGoal')].expandedMinute.tolist()
    ax[1].plot(timelist,difflist,'lightgrey')
    y1positive=(np.asarray(difflist)+1e-7)>=0
    y1negative=(np.asarray(difflist)-1e-7)<0
    ax[1].set_ylabel("xT difference",fontsize=15,color='#edece9')
    ax[1].set_xlabel("Time intervals - every 5 minutes",fontsize=15,color='#e0dfdc')
    ax[1].yaxis.label.set_color('#edece9')
    ax[1].tick_params(axis='y', colors='#edece9')
    ax[1].fill_between(timelist,difflist,where=y1positive,color=hc,alpha=0.7,
                       interpolate=True)
    ax[1].fill_between(timelist,difflist,where=y1negative,color=ac,alpha=0.7,
                       interpolate=True)
    ax[1].scatter(homegoaltimes,[0 for _ in range(len(homegoaltimes))],
                  marker='o',s=100,zorder=5,facecolors='#082630',
                  edgecolors=hc,linewidth=3)
    ax[1].scatter(awaygoaltimes,[0 for _ in range(len(awaygoaltimes))],
                  marker='o',s=100,zorder=5,facecolors='#082630',
                  edgecolors=ac,linewidth=3)
    fig_text(s = f"<{hometeam}>" +' - '+
                f"<{awayteam}>"+' xT cumulative'+'\n'+
             'xT comes from successful passes and carries',
        x = 0.05, y = 0.97, highlight_colors = [hc,ac],
            highlight_weights=['bold','bold'],fontweight='bold',fontsize=20,color='#e0dfdc')

    fig_text(s = f"<{hometeam}>" +' - '+
                f"<{awayteam}>"+' xT gameflow'+'\n'+
             'Calculated as the difference in xT in every 15 min interval',
        x = 0.55, y = 0.97, highlight_colors = [hc,ac],
            highlight_weights=['bold','bold'],fontweight='bold',fontsize=20,color='#e0dfdc')
    plt.tight_layout()
    spines = ['top','right','bottom','left']
    for s in spines:
        ax[0].spines[s].set_color('#e0dfdc')
        ax[1].spines[s].set_color('#e0dfdc')
    return fig

xttim=xTplotter(xt_actions)
'''

'''
min_dribble_length: float = 2.0
max_dribble_length: float = 100.0
max_dribble_duration: float = 20.0
def _add_dribbles(actions):
    next_actions = actions.shift(-1)
    same_team = actions.teamId == next_actions.teamId
    dx = actions.endX - next_actions.x
    dy = actions.endY - next_actions.y
    far_enough = dx ** 2 + dy ** 2 >= min_dribble_length ** 2
    not_too_far = dx ** 2 + dy ** 2 <= max_dribble_length ** 2
    dt = next_actions.time_seconds - actions.time_seconds
    same_phase = dt < max_dribble_duration
    same_period = actions.period_value == next_actions.period_value
    dribble_idx = same_team & far_enough & not_too_far & same_phase & same_period

    dribbles = pd.DataFrame()
    prev = actions[dribble_idx]
    nex = next_actions[dribble_idx]
    dribbles['game_id'] = nex.game_id
    dribbles['period_value'] = nex.period_value
    for cols in ['season_id','competition_id','expandedMinute']:
        dribbles[cols] = nex[cols]
    for cols in ['KP','Assist','TB']:
        dribbles[cols] = [0 for _ in range(len(dribbles))]
    dribbles['isTouch'] = [True for _ in range(len(dribbles))]
    morecols = ['position', 'shirtNo', 'playerId', 'hometeamid', 'awayteamid',
       'hometeam', 'awayteam', 'team', 'competition_name']
    for cols in morecols:
        dribbles[cols] = nex[cols]
    dribbles['action_id'] = prev.action_id + 0.1
    dribbles['time_seconds'] = (prev.time_seconds + nex.time_seconds) / 2
    dribbles['teamId'] = nex.teamId
    dribbles['playerId'] = nex.playerId
    dribbles['name'] = nex.name
    dribbles['receiver'] = [' ' for _ in range(len(dribbles))]
    dribbles['x'] = prev.endX
    dribbles['y'] = prev.endY
    dribbles['endX'] = nex.x
    dribbles['endY'] = nex.y
    dribbles['bodypart'] = ['foot' for _ in range(len(dribbles))]
    dribbles['events'] = ['Carry' for _ in range(len(dribbles))]
    dribbles['outcome'] = ['Successful' for _ in range(len(dribbles))]
    dribbles['type_displayName'] = ['Carry' for _ in range(len(dribbles))]
    dribbles['outcomeType_displayName'] = ['Successful' for _ in range(len(dribbles))]
    dribbles['quals'] = [{} for _ in range(len(dribbles))]
    actions = pd.concat([actions, dribbles], ignore_index=True, sort=False)
    actions = actions.sort_values(['game_id', 'period_value',
                                   'action_id']).reset_index(drop=True)
    actions['action_id'] = range(len(actions))
    return actions

gamedf = awaysdf[awaysdf.awayteam==awayteams.value].reset_index(drop=True)
gamedf['quals'] = gamedf.quals.apply(lambda x:ast.literal_eval(x))
gamedf['name'] = gamedf['name'].fillna(value='')
gamedf['action_id'] = range(len(gamedf))
gamedf.loc[gamedf.type_displayName=='BallRecovery','events'] = 'NonAction' 
gameactions = (
        gamedf[gamedf.events != 'NonAction']
        .sort_values(['game_id', 'period_value', 'time_seconds'])
        .reset_index(drop=True)
    )
gameactions = _add_dribbles(gameactions)
gameactions['poss'] = np.where(gameactions.period_value.diff(-1) != 0,0,
                            np.where(gameactions.teamId==gameactions.hometeamid,1,2))
gameactions['nextposs'] = gameactions.poss.shift(-1)
gameactions['prevposs'] = gameactions.poss.shift(1)
gameactions['prevx'] = gameactions.x.shift(1)
gameactions['prevy'] = gameactions.y.shift(1)
gameactions['Possend'] = np.where(gameactions.poss!=gameactions.nextposs,
                                  gameactions['events'],'')
gameactions['Possbeg'] = np.where(gameactions.poss!=gameactions.prevposs,
                                  gameactions['events'],'')
gameactions['x_beg'] = np.where(gameactions.poss!=gameactions.prevposs,
                                  gameactions['x'],0.0)
gameactions['y_beg'] = np.where(gameactions.poss!=gameactions.prevposs,
                                  gameactions['y'],0.0)
gameactions['homeposs'] = np.where((gameactions.poss==1)&\
                                   (gameactions.type_displayName!='Foul'),1,0)
gameactions['awayposs'] = np.where((gameactions.poss==2)&\
                                   (gameactions.type_displayName!='Foul'),1,0)
gameactions['homecount'] = (((gameactions['homeposs'].diff(1) != 0)).\
                            astype('int').cumsum()*gameactions['homeposs']+1)//2
gameactions['awaycount'] = (((gameactions['awayposs'].diff(1) != 0)).\
                            astype('int').cumsum()*gameactions['awayposs']+1)//2



gamedf['redcard'] = gamedf.quals.apply(lambda x:int(33 in x or 32 in x))
gamedf[gamedf.type_displayName.isin(['SubstitutionOff', 'SubstitutionOn'])|
       gamedf.redcard==1][['name','expandedMinute',
                           'team','type_displayName']].reset_index(drop=True)
'''                          
