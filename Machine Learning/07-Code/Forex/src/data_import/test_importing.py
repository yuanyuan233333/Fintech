import numpy as np
import pandas as pd

import src.data_import.Importing as importing
import talib


def test_out():
    A = list(zip([1, 2, 4, 6, 5],
                 [3, 4, 6, 6, 7],
                 ['01/01', '01/01', '01/01', '01/01', '01/02'],
                 [10, 10, 10, 10, 10],
                 [20, 20, 20, 20, 20],
                 ['00:21', '00:22', '00:23', '00:24', '00:25']))
    df = pd.DataFrame(A, columns=['a', 'b', 'date', 'open', 'close', 'time_min'])
    start_day = '01/01'
    end_day = '01/01'
    space_size = 1
    columns_all = ['a']
    columns_single = ['b']
    year = '2016'
    df["datetime"] = pd.to_datetime((df["date"] + "/" + year + " " + df["time_min"]), format="%m/%d/%Y %H:%M")
    df['minute'] = [21.0, 22.0, 23.0, 24.0, 25.0]
    out1, out2, prices, minutes = importing.create_tuples(start_day, end_day, df, space_size, columns_all,
                                                          columns_single, 0)

    assert (np.array(out1 == np.array([[1, 2, 4], [2, 4, 6]])).all and np.array(
        out2 == np.array([[2, 4, 6], [4, 6, 6]])).all())
    assert np.array(prices['open'] == np.array([10, 10])).all
    assert np.array(prices['close'] == np.array([20, 20])).all
    assert np.array(minutes == np.array([22, 23])).all
    assert minutes.index.equals(out1.index)
    assert minutes.shape == (2,)


def test_multiple_historical_columns():
    A = list(zip([1, 2, 4, 6, 5],
                 [3, 4, 6, 6, 7],
                 [10, 20, 30, 40, 50],
                 ['01/01', '01/01', '01/01', '01/01', '01/02'],
                 [10, 10, 10, 10, 10],
                 [20, 20, 20, 20, 20],
                 [1, 2, 3, 4, 5]))
    df = pd.DataFrame(A, columns=['a', 'b', 'c', 'date', 'open', 'close', 'minute'])
    df['time_min'] = ['00:01', '00:02', '00:03', '00:04', '00:05']
    year = '2016'
    df["datetime"] = pd.to_datetime((df["date"] + "/" + year + " " + df["time_min"]), format="%m/%d/%Y %H:%M")
    start_day = '01/01'
    end_day = '01/01'
    space_size = 1
    columns_all = ['a', 'b']
    columns_single = ['c']

    out1, out2, _, _ = importing.create_tuples(start_day, end_day, df, space_size, columns_all, columns_single,0)
    assert (np.array(out1 == np.array([[1, 2, 3, 4, 20], [2, 4, 4, 6, 30]])).all and np.array(
        out2 == np.array([[2, 4, 4, 6, 30], [4, 6, 6, 6, 40]])).all())


def test_multiple_single_columns():
    A = list(zip([1, 2, 4, 6, 5],
                 [3, 4, 6, 6, 7],
                 [10, 20, 30, 40, 50],
                 ['01/01', '01/01', '01/01', '01/01', '01/02'],
                 [10, 10, 10, 10, 10],
                 [20, 20, 20, 20, 20],
                 [1, 2, 3, 4, 5]))
    df = pd.DataFrame(A, columns=['a', 'b', 'c', 'date', 'open', 'close', 'minute'])
    df['time_min'] = ['00:01', '00:02', '00:03', '00:04', '00:05']
    year = '2016'
    df["datetime"] = pd.to_datetime((df["date"] + "/" + year + " " + df["time_min"]), format="%m/%d/%Y %H:%M")
    start_day = '01/01'
    end_day = '01/01'
    space_size = 1
    columns_all = ['a']
    columns_single = ['b', 'c']
    out1, out2, _, _ = importing.create_tuples(start_day, end_day, df, space_size, columns_all, columns_single,0)
    assert (np.array(out1 == np.array([[1, 2, 4, 20], [2, 4, 6, 30]])).all and np.array(
        out2 == np.array([[2, 4, 6, 30], [4, 6, 6, 40]])).all())


def test_multiple_days():
    A = list(zip([1, 2, 4, 6, 5, 10],
                 [3, 4, 6, 6, 7, 20],
                 ['01/01', '01/01', '01/01', '01/02', '01/02', '01/02'],
                 [10, 10, 10, 10, 10, 10],
                 [20, 20, 20, 20, 20, 20],
                 [1, 2, 3, 4, 5, 6]))
    df = pd.DataFrame(A, columns=['a', 'b', 'date', 'open', 'close', 'minute'])
    df['time_min'] = ['00:01', '00:02', '00:03', '00:04', '00:05', '00:06']
    year = '2016'
    df["datetime"] = pd.to_datetime((df["date"] + "/" + year + " " + df["time_min"]), format="%m/%d/%Y %H:%M")
    start_day = '01/01'
    end_day = '01/02'
    space_size = 1
    columns_all = ['a']
    columns_single = ['b']
    out1, out2, _, _ = importing.create_tuples(start_day, end_day, df, space_size, columns_all, columns_single,0)
    assert (np.array(out1 == np.array([[1, 2, 4], [6, 5, 7]])).all and np.array(
        out2 == np.array([[2, 4, 6], [5, 10, 20]])).all())


def test_extract_minute():
    df = pd.DataFrame({'date': ['01/04', '01/10'],
                       'time_min': ['08:21', '02:10']})
    minutes = importing.extract_minute(df)
    assert minutes[0] == 8 * 60 + 21
    assert minutes[1] == 2 * 60 + 10


def test_import_talib():
    A = list(zip([1, 2, 4, 6, 5],
                 [3, 4, 6, 6, 7],
                 [10, 20, 30, 40, 50],
                 ['01/01', '01/01', '01/01', '01/01', '01/02'],
                 [10, 10, 10, 10, 10],
                 np.array([20, 20, 20, 20, 20], dtype='f8'),
                 [1, 2, 3, 4, 5]))
    df = pd.DataFrame(A, columns=['a', 'b', 'c', 'date', 'open', 'close', 'minute'])
    df['time_min'] = ['00:01', '00:02', '00:03', '00:04', '00:05']
    year = '2016'
    df["datetime"] = pd.to_datetime((df["date"] + "/" + year + " " + df["time_min"]), format="%m/%d/%Y %H:%M")
    talib_historical = 2
    talib_func = ['SMA']
    df, names = importing.import_talib(df, talib_func, talib_historical)
    assert np.array(df['SMA'][1:-1] == talib.SMA(df['close'], talib_historical)[1:-1]).all()


def test_multidim():
    A = list(zip([1, 2, 4, 6, 5],
                 [3, 4, 6, 6, 7],
                 [10, 20, 30, 40, 50],
                 ['01/01', '01/01', '01/01', '01/01', '01/02'],
                 [10, 10, 10, 10, 10],
                 np.array([20, 20, 20, 20, 20], dtype='f8'),
                 [1, 2, 3, 4, 5]))
    df = pd.DataFrame(A, columns=['a', 'b', 'c', 'date', 'open', 'close', 'minute'])
    df['time_min'] = ['00:01', '00:02', '00:03', '00:04', '00:05']
    year = '2016'
    df["datetime"] = pd.to_datetime((df["date"] + "/" + year + " " + df["time_min"]), format="%m/%d/%Y %H:%M")
    talib_historical = 2
    talib_func = ['BBANDS']
    df, names = importing.import_talib(df, talib_func, talib_historical)
    assert np.array(names == ['BBANDSupperband', 'BBANDSmiddleband', 'BBANDSlowerband']).all()


def test_MAMA():
    A = list(zip([1, 2, 4, 6, 5],
                 [3, 4, 6, 6, 7],
                 [10, 20, 30, 40, 50],
                 ['01/01', '01/01', '01/01', '01/01', '01/02'],
                 [10, 10, 10, 10, 10],
                 np.array([20, 20, 20, 20, 20], dtype='f8'),
                 [1, 2, 3, 4, 5]))
    df = pd.DataFrame(A, columns=['a', 'b', 'c', 'date', 'open', 'close', 'minute'])
    df['time_min'] = ['00:01', '00:02', '00:03', '00:04', '00:05']
    year = '2016'
    df["datetime"] = pd.to_datetime((df["date"] + "/" + year + " " + df["time_min"]), format="%m/%d/%Y %H:%M")
    talib_historical = 2
    talib_func = ['MAMA']
    df, names = importing.import_talib(df, talib_func, talib_historical)
    assert np.array(names == ['MAMAmama', 'MAMAfama']).all()


def test_MAVP():
    A = list(zip([1, 2, 4, 6, 5],
                 [3, 4, 6, 6, 7],
                 [10, 20, 30, 40, 50],
                 ['01/01', '01/01', '01/01', '01/01', '01/02'],
                 [10, 10, 10, 10, 10],
                 np.array([20, 20, 20, 20, 20], dtype='f8'),
                 [1, 2, 3, 4, 5]))
    df = pd.DataFrame(A, columns=['a', 'b', 'c', 'date', 'open', 'close', 'minute'])
    df['time_min'] = ['00:01', '00:02', '00:03', '00:04', '00:05']
    year = '2016'
    df["datetime"] = pd.to_datetime((df["date"] + "/" + year + " " + df["time_min"]), format="%m/%d/%Y %H:%M")
    talib_historical = 2
    talib_func = ['MAVP']
    df, names = importing.import_talib(df, talib_func, talib_historical)
    assert np.array(names == ['MAVP']).all()


def test_apply_talib_single_output():
    df = pd.DataFrame({'close': [1., 2., 3., 4., 5.]})
    period = 2
    talib_functions = [
        {'name': 'MA', 'parameters': {'timeperiod': period}}
    ]
    talib_df, _ = importing.apply_talib(df, talib_functions)
    assert np.array(talib_df['MA'][1:] == talib.MA(df['close'], period)[1:]).all()


def test_apply_talib_multiple_outputs():
    df = pd.DataFrame({'close': [1., 2., 3., 4., 5.]})
    period = 2
    talib_functions = [
        {'name': 'BBANDS', 'parameters': {'timeperiod': period}}
    ]
    talib_df, _ = importing.apply_talib(df, talib_functions)
    actual = talib_df.loc[:, ['BBANDSupperband', 'BBANDSmiddleband', 'BBANDSlowerband']]
    expected = talib.BBANDS(df['close'], period)
    assert np.array(actual.loc[:, 'BBANDSupperband'][1:] == expected[0][1:]).all()
    assert np.array(actual.loc[:, 'BBANDSmiddleband'][1:] == expected[1][1:]).all()
    assert np.array(actual.loc[:, 'BBANDSlowerband'][1:] == expected[2][1:]).all()


def test_apply_talib_single_output_no_parameters():
    df = pd.DataFrame({'close': np.linspace(1, 100, 200)})
    talib_functions = [
        {'name': 'HT_TRENDLINE', 'parameters': {}}
    ]
    talib_df, _ = importing.apply_talib(df, talib_functions)
    not_nan = ~np.isnan(talib_df['HT_TRENDLINE'])
    expected = talib.HT_TRENDLINE(df['close'])
    assert np.array(talib_df['HT_TRENDLINE'][not_nan] == expected[not_nan]).all()


def test_apply_talib_multiple_parameters():
    df = pd.DataFrame({'close': np.linspace(1, 100, 200)})
    talib_functions = [
        {'name': 'MACD', 'parameters': {'fastperiod': 5, 'slowperiod': 10}}
    ]
    talib_df, _ = importing.apply_talib(df, talib_functions)
    actual = talib_df.loc[:, ['MACDmacd', 'MACDmacdsignal', 'MACDmacdhist']]
    not_nan = ~np.isnan(talib_df['MACDmacd'])
    expected = talib.MACD(df['close'], fastperiod=5, slowperiod=10)
    assert np.array(actual.loc[:, 'MACDmacd'][not_nan] == expected[0][not_nan]).all()
    assert np.array(actual.loc[:, 'MACDmacdsignal'][not_nan] == expected[1][not_nan]).all()
    assert np.array(actual.loc[:, 'MACDmacdhist'][not_nan] == expected[2][not_nan]).all()

