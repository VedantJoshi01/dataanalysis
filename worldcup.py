# -*- coding: utf-8 -*-
"""FIFA WorldCup Data Analysis.py

Original file is located at
    https://colab.research.google.com/drive/1WHkQXfVieiojx5QKTbHQcCc5VTvtA416
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
match=pd.read_csv("WorldCupMatches.csv")
player=pd.read_csv("WorldCupPlayers.csv")
worldcup= pd.read_csv("WorldCups.csv")

"""1) Looking at the content"""

worldcup.head()

match.head()

player.head()

"""2) Cleaning the data


"""

match.dropna(subset=['Year'],inplace=True)

match['Home Team Name'].value_counts()

name= match[match['Home Team Name'].str.contains('rn">')]['Home Team Name'].value_counts()
name

wrong_names=list(name.index)
wrong_names

correct_names=[name.split(">")[1] for name in wrong_names]
correct_names

old = ['Germany FR', 'Maracan� - Est�dio Jornalista M�rio Filho', 'Estadio do Maracana']
new = ['Germany', 'Maracan Stadium', 'Maracan Stadium']

wrong_names=wrong_names+old
correct_names=correct_names+new

correct_names

wrong_names

for index,wr in enumerate(wrong_names):
  worldcup=worldcup.replace(wrong_names[index],correct_names[index])
for index,wr in enumerate(wrong_names):
  match=match.replace(wrong_names[index],correct_names[index])
for index,wr in enumerate(wrong_names):
  player=player.replace(wrong_names[index],correct_names[index])

names= match[match['Home Team Name'].str.contains('rn">')]['Home Team Name'].value_counts()
names

"""This shows that there are no wrong values anymore.

3) Most number of Worldcup Winning Title
"""

winner=worldcup['Winner'].value_counts()
winner

runnerup=worldcup['Runners-Up'].value_counts()
runnerup

third=worldcup['Third'].value_counts()
third

Top3=pd.concat([winner,runnerup,third],axis=1)
Top3.fillna('0',inplace=True)
Top3=Top3.astype(int)
Top3

import plotly as py
import cufflinks as cf
from plotly.offline import iplot
py.offline.init_notebook_mode(connected=True)
cf.go_offline()

Top3.iplot(kind = 'bar', xTitle='Teams', yTitle='Count', title='FIFA World Cup Winning Count')

"""Number of Goals per Country"""

home=match[['Home Team Name','Home Team Goals']].dropna()
away=match[['Away Team Name','Away Team Goals']].dropna()

home.columns = ['Countries', 'Goals']
away.columns = home.columns

goals = home.append(away, ignore_index = True)
goals = goals.groupby('Countries').sum()
goals

goals = goals.sort_values(by = 'Goals', ascending=False)
goals

goals[:20].iplot(kind='bar', xTitle = 'Country Names', yTitle = 'Goals', title = 'Countries Hits Number of Goals')

"""Attendance,Number of Teams,Goals and Matches per Cup"""

worldcup['Attendance']=worldcup['Attendance'].str.replace('.','')

worldcup.head()

fig, ax = plt.subplots(figsize = (10,5))
sns.despine(right = True)
g = sns.barplot(x = 'Year', y = 'Attendance', data = worldcup)
g.set_xticklabels(g.get_xticklabels(), rotation = 80)
g.set_title('Attendance Per Year')

fig, ax = plt.subplots(figsize = (10,5))
sns.despine(right = True)
g = sns.barplot(x = 'Year', y = 'QualifiedTeams', data = worldcup)
g.set_xticklabels(g.get_xticklabels(), rotation = 80)
g.set_title('Qualified Teams Per Year')

fig, ax = plt.subplots(figsize = (10,5))
sns.despine(right = True)
g = sns.barplot(x = 'Year', y = 'GoalsScored', data = worldcup)
g.set_xticklabels(g.get_xticklabels(), rotation = 80)
g.set_title('Goals Scored by Teams Per Year')

"""Goals per Team per World Cup"""

home = match.groupby(['Year', 'Home Team Name'])['Home Team Goals'].sum()
home

away = match.groupby(['Year', 'Away Team Name'])['Away Team Goals'].sum()
away

goals = pd.concat([home, away], axis=1)
goals.fillna(0, inplace=True)
goals['Goals'] = goals['Home Team Goals'] + goals['Away Team Goals']
goals = goals.drop(labels = ['Home Team Goals', 'Away Team Goals'], axis = 1)
goals

goals = goals.reset_index()

goals.columns = ['Year', 'Country', 'Goals']
goals = goals.sort_values(by = ['Year', 'Goals'], ascending = [True, False])
goals

top5 = goals.groupby('Year').head()
top5.head(10)

import plotly.graph_objects as go
x, y = goals['Year'].values, goals['Goals'].values
data = []
for team in top5['Country'].drop_duplicates().values:
    year = top5[top5['Country'] == team]['Year']
    goal = top5[top5['Country'] == team]['Goals']
    
    data.append(go.Bar(x = year, y = goal, name = team))
layout = go.Layout(barmode = 'stack', title = 'Top 5 Teams with most Goals', showlegend = False)

fig = go.Figure(data = data, layout = layout)
fig.show()

"""Matches with Highest Number of Attendance"""

match['Datetime'] = pd.to_datetime(match['Datetime'])
match['Datetime'] = match['Datetime'].apply(lambda x: x.strftime('%d %b, %y'))
top10 = match.sort_values(by = 'Attendance', ascending = False)[:10]
top10['vs'] = top10['Home Team Name'] + " vs " + top10['Away Team Name']

plt.figure(figsize = (12,10))

ax = sns.barplot(y = top10['vs'], x = top10['Attendance'])
sns.despine(right = True)
plt.ylabel('Match Teams')
plt.xlabel('Attendence')
plt.title('Matches with the highest number of Attendence')

for i, s in enumerate("Stadium: " + top10['Stadium'] +", Date: " + top10['Datetime']):
    ax.text(2000, i, s, fontsize = 12, color = 'white')
plt.show()

"""Which countries have won the world cup"""

gold = worldcup["Winner"]
silver = worldcup["Runners-Up"]
bronze = worldcup["Third"]

gold_count = pd.DataFrame.from_dict(gold.value_counts())
silver_count = pd.DataFrame.from_dict(silver.value_counts())
bronze_count = pd.DataFrame.from_dict(bronze.value_counts())
podium_count = gold_count.join(silver_count, how='outer').join(bronze_count, how='outer')
podium_count = podium_count.fillna(0)
podium_count.columns = ['WINNER', 'SECOND', 'THIRD']
podium_count = podium_count.astype('int64')
podium_count = podium_count.sort_values(by=['WINNER', 'SECOND', 'THIRD'], ascending=False)

podium_count.plot(y=['WINNER', 'SECOND', 'THIRD'], kind="bar", 
                  color =['gold','silver','brown'], figsize=(15, 6), fontsize=14,
                 width=0.8, align='center')
plt.xlabel('Countries')
plt.ylabel('Number of podium')
plt.title('Number of podium by country')

"""Number of Goals per Country"""

home = match[['Home Team Name', 'Home Team Goals']].dropna()
away = match[['Away Team Name', 'Away Team Goals']].dropna()

goal_per_country = pd.DataFrame(columns=['countries', 'goals'])
goal_per_country = goal_per_country.append(home.rename(index=str,columns={'Home Team Name': 'countries', 'Home Team Goals': 'goals'}))
goal_per_country = goal_per_country.append(away.rename(index=str, columns={'Away Team Name': 'countries', 'Away Team Goals': 'goals'}))
goal_per_country['goals'] = goal_per_country['goals'].astype('int64')

goal_per_country = goal_per_country.groupby(['countries'])['goals'].sum().sort_values(ascending=False)

goal_per_country[:10].plot(x=goal_per_country.index,
                           y=goal_per_country.values, kind="bar", figsize=(12, 6), fontsize=14)
plt.xlabel('Countries')
plt.ylabel('Number of goals')
plt.title('Top 10 of Number of goals by country')
