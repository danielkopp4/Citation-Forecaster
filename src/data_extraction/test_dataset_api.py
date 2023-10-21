import os
from typing import Set 
import pandas as pd
import datetime
import time
import logging

from src.data_api import HistoricalBettingDataAPI
from src.betting_env.odds import Odds


folder_name = "./nba-historical-stats-and-betting-data"
money_line_file = "nba_betting_money_line.csv"
games_file = "nba_games_all.csv"
book_name = "Bovada"


def download_data():
    if not os.path.exists(folder_name):
        import opendatasets as od
        dataset_link = "https://www.kaggle.com/datasets/ehallmar/nba-historical-stats-and-betting-data"
        od.download(dataset_link)


def load_data(book_name=None):
    bets = pd.read_csv(os.path.join(folder_name, money_line_file))
    games = pd.read_csv(os.path.join(folder_name, games_file))
    
    if book_name:
        bets = bets.loc[bets["book_name"] == book_name]

    

    bets = bets.drop(["book_name", "book_id"], axis=1)
    # add more column names to first array to include in final data
    bets["game_id"] = bets["game_id"].astype(int)
    bets["team_id"] = bets["team_id"].astype(int)
    bets["a_team_id"] = bets["a_team_id"].astype(int)

    bets = bets.merge(games[["game_id", "team_id", "a_team_id", "is_home", "wl", "season_year"]], on=["game_id", "team_id", "a_team_id"])
    bets = bets.drop(["is_home", "game_id"], axis=1)

    def conv_odds(row):
        return Odds.from_american(row)

    bets["price1"] = bets["price1"].apply(conv_odds)
    bets["price2"] = bets["price2"].apply(conv_odds)

    def conv_wl(row):
        if row == "W":
            return 0
        elif row == "L":
            return 1
        else:
            logging.critical("UNEXPECTED VALUE in 'wl' column", row)
            exit()

    def conv_year(row):
        date = datetime.datetime(row, 1, 1)
        return time.mktime(date.timetuple())

    bets["wl"] = bets["wl"].apply(conv_wl)
    bets["date"] = bets["season_year"].apply(conv_year)

    bets["team_a"] = bets["team_id"]
    bets["team_b"] = bets["a_team_id"]
    bets["o_a"] = bets["price1"]
    bets["o_b"] = bets["price2"]

    bets = bets.drop(["season_year", "team_id", "a_team_id", "price1", "price2"], axis=1)

    return bets

class NBAHistoricalBettingAPI(HistoricalBettingDataAPI):
    def __init__(self):
        self._data = load_data(book_name=book_name)
        print("loaded", len(self._data), "points")
        super(NBAHistoricalBettingAPI, self).__init__()

    def _get_item_raw(self, index):
        dp = self._data.iloc[index]
        return (
            dp["o_a"], 
            dp["o_b"], 
            dp["team_a"],
            dp["team_b"], 
            dp["date"], 
            dp["wl"]
        )

    @property
    def _length_raw(self) -> int:
        return self._data.shape[0]

    def get_unique_teams(self) -> Set[str]:
        return set(self._data["team_a"]).union(self._data["team_b"])

if __name__ == "__main__":
    download_data()
    print(load_data().head())