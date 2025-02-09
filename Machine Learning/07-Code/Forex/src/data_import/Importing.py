import pandas as pd
import datetime
import numpy as np
import talib
from talib import abstract


def import_df(csv_name):
    df = pd.read_csv(csv_name)
    year = csv_name[csv_name.find("20"):csv_name.find("20") + 4]
    df['minute'] = 0.00
    df['log_open'] = 0.00
    df['log_close'] = 0.00
    df['log_high'] = 0.00
    df['log_low'] = 0.00
    df['minute'] = extract_minute(df)
    df[['log_open', 'log_close', 'log_high', 'log_low']] = \
        df[['date', 'open', 'close', 'high', 'low']].groupby('date') \
            .apply(lambda x: np.log(x / x.loc[0, 'open'])) \
            .reset_index(drop=True)

    df[['volume']] = \
        df[['date', 'count']].groupby('date').apply(lambda x: (x / x.iloc[0,])) \
            .reset_index(drop=True)
    df = df.drop('count', axis=1)
    df["datetime"] = pd.to_datetime((df["date"] + "/" + year + " " + df["time_min"]), format="%m/%d/%Y %H:%M")
    return df


def import_talib(df, talib_func, talib_historical):
    df['periods'] = float(talib_historical)
    talib_names = []
    for name in talib_func:
        if 'timeperiod' in talib.abstract.Function(name).parameters.keys():
            temp = talib.abstract.Function(name)(df, timeperiod=talib_historical)
        else:
            temp = talib.abstract.Function(name)(df)
        if isinstance(temp, pd.Series):
            df[name] = temp
            talib_names += [name]
        else:
            df[[name + t for t in list(temp.columns)]] = temp
            talib_names += [name + t for t in list(temp.columns)]
    return df, talib_names


def apply_talib(df, talib_functions):
    # this implementation does not work with MAVP (MAVP requires column periods)
    talib_names = []
    for indicator in talib_functions:
        talib_output = talib.abstract.Function(indicator['name'])(df, **indicator['parameters'])
        if isinstance(talib_output, pd.Series):
            df[indicator['name']] = talib_output
            talib_names += [indicator['name']]
        else:
            df[[indicator['name'] + t for t in list(talib_output.columns)]] = talib_output
            talib_names += [indicator['name'] + t for t in list(talib_output.columns)]
    return df, talib_names


def extract_minute(df):
    deltas = pd.to_timedelta(df['time_min'] + ':00')
    return deltas / np.timedelta64(1, 'm')


def create_tuples(start_day, end_day, df, space_size, columns_all, columns_single,starting_time):
    timeperiod_talib = max(np.max(df.isnull().sum(axis=0)),starting_time)
    shift_talib = max(space_size, timeperiod_talib) - space_size
    colnames = [name + '_' + str(i - space_size) for name in columns_all for i in
                range(0, space_size + 1)] + columns_single
    type_names = [df[name].dtype.name for name in columns_all for i in range(0, space_size + 1)] + \
                 [df[name].dtype.name for name in columns_single]
    my_types = dict(zip(colnames, type_names))

    from_day = datetime.datetime.strptime(start_day + "/" + str(df.datetime[0].year), '%m/%d/%Y')
    to_day = datetime.datetime.strptime(end_day + "/" + str(df.datetime[0].year), '%m/%d/%Y') + \
             datetime.timedelta(hours=24)
    df = df[(df['datetime'] >= from_day) & (df['datetime'] < to_day)]

    names_loop = [name + '_' + str(0) for name in columns_all]
    col_names_last_iter = np.unique(columns_all + columns_single + ['open', 'close', 'minute']).tolist()
    df_loop = rows_by_day(df=df,
                          cols=col_names_last_iter,
                          skip_start=shift_talib + space_size, skip_end=+1)

    df_in = pd.DataFrame(columns=colnames, index=range(0, df_loop.shape[0]))
    df_out = pd.DataFrame(columns=colnames, index=range(0, df_loop.shape[0]))

    df_in[names_loop + columns_single] = df_loop[columns_all + columns_single].values
    price_info = df_loop[['open', 'close']].reset_index(drop=True)
    minutes = pd.Series(df_loop.minute.values)

    df_loop = rows_by_day(df=df,
                          cols=col_names_last_iter,
                          skip_start=shift_talib + space_size + 1, skip_end=+0)

    df_out[names_loop + columns_single] = df_loop[columns_all + columns_single].values

    for i in range(0, space_size):
        names_loop = [name + '_' + str(i - space_size) for name in columns_all]

        df_loop = rows_by_day(df=df, cols=columns_all, skip_start=shift_talib + i, skip_end=space_size - i + 1)
        df_in[names_loop] = df_loop.values

        df_loop = rows_by_day(df=df, cols=columns_all, skip_start=shift_talib + i + 1, skip_end=space_size - i)
        df_out[names_loop] = df_loop.values

    df_in = df_in.astype(my_types)
    df_out = df_out.astype(my_types)

    return df_in, df_out, price_info, minutes


def rows_by_day(df, cols, skip_start=0, skip_end=0):
    df_filtered = df[["date"] + cols].groupby("date") \
        .apply(lambda x: x.iloc[skip_start:x.count()[0] - skip_end, :]) \
        .reset_index(drop=True) \
        .drop("date", axis=1)

    return df_filtered
