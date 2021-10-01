import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#sys.path.insert(0,'../../ift6758')
#from data import get_player_stats








# Something wrong with my pathing above, included the whole function from example.
def get_player_stats(year: int, player_type: str) -> pd.DataFrame:
    """

    Uses Pandas' built in HTML parser to scrape the tabular player statistics from
    https://www.hockey-reference.com/leagues/ . If the player played on multiple 
    teams in a single season, the individual team's statistics are discarded and
    the total ('TOT') statistics are retained (the multiple team names are discarded)

    Args:
        year (int): The first year of the season to retrieve, i.e. for the 2016-17
            season you'd put in 2016
        player_type (str): Either 'skaters' for forwards and defensemen, or 'goalies'
            for goaltenders.
    """

    if player_type not in ["skaters", "goalies"]:
        raise RuntimeError("'player_type' must be either 'skaters' or 'goalies'")
    
    url = f'https://www.hockey-reference.com/leagues/NHL_{year}_{player_type}.html'

    print(f"Retrieving data from '{url}'...")

    # Use Pandas' built in HTML parser to retrieve the tabular data from the web data
    # Uses BeautifulSoup4 in the background to do the heavylifting
    df = pd.read_html(url, header=1)[0]

    # get players which changed teams during a season
    players_multiple_teams = df[df['Tm'].isin(['TOT'])]

    # filter out players who played on multiple teams
    df = df[~df['Player'].isin(players_multiple_teams['Player'])]
    df = df[df['Player'] != "Player"]

    # add the aggregate rows
    df = df.append(players_multiple_teams, ignore_index=True)

    return df








def Save_DataFrame(Data,FileName:str="GoalieData",DirFd:str="data/WarmUp"):
    # Save the data frame in csv format, create directories if they do not exist
    
    if os.path.isdir(DirFd) == 0:
        os.makedirs(DirFd)
        print("Created Directry")
    if os.path.isfile(f"{DirFd}/{FileName}") == 0:
        Data.to_csv(f"{DirFd}/{FileName}")
    else:
        return print("File exist")
    return print("Data saved")

def Splice_DataFrame(DataFrameLarge, Col=["SV%"]):
    # Reduce the size of the data frame by keeping the selected columns
    DataFrameSmall = DataFrameLarge[Col]   
    return DataFrameSmall
        

    
def AddNewDataColumn(DataFrame, Col1: str="SV%", Col2: str="GP",NewCol: str="ModSV%"):
    # Adding a new data column by multiplying two existing columns, replace NaN with 0
    #print(DataFrame.isna().sum())
    DataFrame = DataFrame.fillna(0)
    #print(DataFrame.isna().sum())
    DataFrame[NewCol] = pd.to_numeric(DataFrame[Col1]) * pd.to_numeric(DataFrame[Col2]) 
    #print(DataFrame.isna().sum())
    return DataFrame    



def PlotBar(DataFrame, Row:str="ModSV%", Col:str="Player", Count:int=20, xlab:str="Modified Safe Value",ylab:str="Player Name",Tit:str="Goalie Performance Ranking", DirFd:str="data/WarmUp"):
    # Plot two of the data frame columns on a horizontal bar chart, descending sort automatically, save output image
    
    Plot_Data = DataFrame[[Col,Row]].sort_values(Row, ascending=False)

    f, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(x=Row, y=Col, data=Plot_Data[0:Count],
                color="b")

    sns.despine(left=True, bottom=True)
    ax.set(title=Tit,ylabel=ylab,
           xlabel=xlab)    
    
    print('Save plot as ' + DirFd+"/"+Tit+'.jpg')
    plt.savefig(DirFd+"/"+Tit+'.jpg', bbox_inches = "tight")
    
    return print("Plot Created")


def main_test():

    DataFrameAll = get_player_stats(2017, 'goalies')
    Save_DataFrame(DataFrameAll)
    DataFrameFilt = Splice_DataFrame(DataFrameAll,['Player','GP','GA','SA','SV%','MIN'])
    NewDataFrame = AddNewDataColumn(DataFrameFilt, "SV%", "GP","ModSV%")
    PlotBar(NewDataFrame)
    
    return print("Warm Up Complete")



if __name__ == '__main__':
    main_test()    