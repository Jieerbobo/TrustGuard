import pandas as pd
import datetime


def get_snapshot_index(time_slots, data_path):
    df = pd.read_csv(data_path)
    df.columns = ['source','target','rating','time']
    time_list = list(df['time'])
    time_list.sort()

    ts_begin = time_list[0]
    begin = datetime.datetime.fromtimestamp(ts_begin).strftime('%Y-%m-%d %H:%M:%S')
    ts_finish = time_list[-1]
    finish = datetime.datetime.fromtimestamp(ts_finish).strftime('%Y-%m-%d %H:%M:%S')
    print('Total {} interactions from {} to {}'.format(len(time_list),begin,finish))
    days = (datetime.datetime.fromtimestamp(ts_finish)-datetime.datetime.fromtimestamp(ts_begin)).days
    print('Total {} days'.format(days))

    span = ts_finish-ts_begin  # timestamp gap
    # print('Max: {}, Min: {}, Span: {}'.format(ts_finish,ts_begin,span))

    split_list = []
    for i in range(1,time_slots):
        ts_begin += (span//time_slots)
        split_list.append(ts_begin)
        # print(datetime.datetime.fromtimestamp(ts_begin).strftime('%Y-%m-%d %H:%M:%S'))
    split_list.append(ts_finish)
    # print(datetime.datetime.fromtimestamp(ts_finish).strftime('%Y-%m-%d %H:%M:%S'))
    # print('Split list:',split_list)

    index_list = []
    for span in split_list:
        if span == split_list[-1]:
            index_list.append(len(time_list)-1)
            break
        for j in range(len(time_list)):
            if time_list[j] < span < time_list[j + 1]:
                index_list.append(j)

    for i in range(len(split_list)):
        if i == 0:
            print('In snapshot {}'.format(i),'Time:{}'.format(datetime.datetime.fromtimestamp(split_list[i]).strftime('%Y-%m-%d %H:%M:%S')),'#Edges={}'.format(
            index_list[i]))
        else:
            print('In snapshot {}'.format(i),'Time:{}'.format(datetime.datetime.fromtimestamp(split_list[i]).strftime('%Y-%m-%d %H:%M:%S')),'#Edges={}'.format(
            index_list[i] - index_list[i-1]))
    return index_list
