import pandas as pd
import numpy as np
from pymongo import MongoClient

from sumo.pandas_util import nan_rows

PORT_NUM = 27017
BANZUKE_RANGE = [201701, 201703, 201705, 201707, 201709]
SUMO_RANKS = ['Y', 'O', 'S', 'K', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16']

def connect():
    client = MongoClient('mongo', 27017)
    db = client.sumo
    return db


def load_banzuke(db, banzuke_range):
    if banzuke_range is None:
        banzuke_range = {}
    else:
        banzuke_range = {'_id': {"$in" : banzuke_range}}

    collection = db.banzuke.find(banzuke_range)

    def wins_last_basho(x):
        return x.groupby('rid')['wins'].apply(lambda x: x.shift().fillna(0))

    def absent_last_basho(x):
        return x.groupby('rid')['absent'].apply(lambda x: x.shift().fillna(0))

    df = (pd.DataFrame(list(collection))
            .groupby('_id')
            .apply(lambda x: pd.concat([pd.DataFrame(y) for y in x['rikishi']], axis=0))
            .assign(wins=lambda x: pd.DataFrame(x.score.values.tolist(), index=x.index)[0],
                     loss=lambda x: pd.DataFrame(x.score.values.tolist(), index=x.index)[1],
                     absent=lambda x: pd.DataFrame(x.score.values.tolist(), index=x.index)[2],
                     bid=lambda x: x.index.get_level_values(0))
            .drop('score', axis=1)
            .rename(columns={'rikishi_id': 'rid'})
            .assign(rid=lambda x: x.rid.astype(int))
            .assign(wins_last_basho=lambda x: wins_last_basho(x),
                    absent_last_basho=lambda x: absent_last_basho(x),
                    rank=lambda x: pd.Categorical(x['rank'], categories=SUMO_RANKS, ordered=True))
            .set_index(['bid', 'rid'])
         )
    return df


def load_history(db, banzuke_range):
    if banzuke_range is None:
        banzuke_range = {}
    else:
        banzuke_range = {'bid': {"$in" : banzuke_range}}

    collection = db.rikishi_banzuke.find(banzuke_range)

    df = pd.DataFrame(list(collection))
    df = df.set_index(['rid', 'bid'])

    def make_single_rikishi_tournament(idx, row):
        d = pd.DataFrame(row.history)
        d['rid'] = idx[0]
        d['bid'] = idx[1]
        return d

    def convert_result(results):
        results = results.copy()
        results[results != 'shiro'] = 0
        results[results == 'shiro'] = 1
        return results

    def day_to_numeric(days):
        return days.apply(lambda x: x.split(' ')[1]).astype(int)

    history_rows = [make_single_rikishi_tournament(*item) for item in df.iterrows()]
    history_df = (pd.concat(history_rows, sort=True)
                      .assign(bid=lambda x: x.bid.astype(int))
                      .assign(rid=lambda x: x.rid.astype(int))
                      .assign(opponent=lambda x: x.opponent.astype(int))
                      .assign(day=lambda x: day_to_numeric(x.day))
                      .assign(result=lambda x: convert_result(x.result))
                      .assign(kimarite=lambda x: pd.Categorical(x.kimarite).codes)
                      .set_index(['bid', 'day', 'rid'])
                )

    # history_df.index =history_df.index.droplevel(-1)
    history_df = history_df.reset_index().set_index(['bid', 'day', 'rid'])
    return history_df
