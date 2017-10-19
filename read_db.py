import pymysql
import datetime
import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False 


list_map = {'0-0':[u'150FI前润滑油压【1LHP 150LP】',u'压力','bar',10,0.1],
            '0-1':[u'150FI后润滑油压【1LHP 151LP】',u'压力','bar',10,0.1],
            '0-2':[u'151FI前润滑油压【1LHP 152P】',u'压力','bar',10,0.1],
            '0-3':[u'151FI后润滑油压【1LHP 153LP】',u'压力','bar',10,0.1],
            '0-4':[u'100FI前燃油压力【1LHP 100LP】',u'压力','bar',6,0.05],
            '0-5':[u'100FI后燃油压力【1LHP 101LP】',u'压力','bar',6,0.05],
            '0-6':[u'001M0A侧助燃空气压力【1LHP 300LP】',u'压力','bar',4,0.05],
            '0-7':[u'001M0B侧助燃空气压力【1LHP 301LP】',u'压力','bar',4,0.05],
            
            '1-0':[u'001M0冷却水入口压力【1LHP 200LP】',u'压力','bar',10,0.1],
            '1-1':[u'001M0冷却水入口温度【1LHP 200LT】',u'温度','℃',120,1],
            '1-2':[u'001M0润滑油A侧入口油温【1LHP 150LT】',u'温度','℃',120,1],
            '1-3':[u'001M0润滑油B侧入口油温【1LHP 152LT】',u'温度','℃',120,1],
            '1-4':[u'001M0冷却水出口温度【1LHP 203LT】',u'温度','℃',120,1],
            '1-5':[u'001M0A侧助燃空气温度【1LHP 300LT】',u'温度','℃',120,1],
            '1-6':[u'001M0B侧助燃空气温度【1LHP 301LT】',u'温度','℃',120,1]
            ###'表盘编号'[标题名称，Y坐标名称，单位，量程最大值，比例]
            }


def read_data_db(camera_num, dial_num, time_start, time_stop):
    conn = pymysql.connect(user='root', passwd='941120', db='bhxz')
    cursor = conn.cursor()
    if time.strftime('%Y-%m-%d',time.localtime(time_start)) == time.strftime('%Y-%m-%d',time.localtime(time_stop)):
        db_name = time.strftime('data%Y%m%d',time.localtime(time_start))
        sql = r'SELECT * FROM %s WHERE time>=%s and time<=%s and name="%s-%s"'%(db_name, time_start, time_stop, camera_num, dial_num)
    elif (time_stop - time_start) <= 24*60*60:
        db_name_0 = time.strftime('data%Y%m%d',time.localtime(time_start))
        db_name_1 = time.strftime('data%Y%m%d',time.localtime(time_stop))
        sql = r'SELECT * FROM {0} WHERE time>={1} and name="{2}-{3}" UNION ALL SELECT * FROM {4} WHERE time<={5} and name="{2}-{3}"'.format(db_name_0, time_start, camera_num, dial_num, db_name_1, time_stop)
    else:
        db_name_0 = time.strftime('data%Y%m%d',time.localtime(time_start))
        db_name_1 = time.strftime('data%Y%m%d',time.localtime(time_stop))
        time_stop = time_start + 24*60*60
        sql = r'SELECT * FROM {0} WHERE time>={1} and name="{2}-{3}" UNION ALL SELECT * FROM {4} WHERE time<={5} and name="{2}-{3}"'.format(db_name_0, time_start, camera_num, dial_num, db_name_1, time_stop)
    try:
        cursor.execute(sql)
        conn.commit()
        result = cursor.fetchall()
    except:
        print ('无所选时间历史数据!!')
        os._exit(0)
    finally:
        conn.close()
    if len(result) == 0:
        print ('无所选时间历史数据!')
        os._exit(0)
    return camera_num, dial_num, result

def plot_data(db_datas):
    data = []
    index = 0
    x_ticks_index = []
    date_time_index = []
    camera_num, dial_num, db_data = db_datas
    for i in db_data:
        one_data = list(map(int,i[2].split()))
        data += (one_data)
        x_ticks_index.append(index)
        index += len(one_data)
        date_time_index.append(time.strftime("%H:%M:%S", time.localtime(i[0])))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.ylim(0, list_map['%s-%s'%(camera_num, dial_num)][3])
    data = np.array(data)*list_map['%s-%s'%(camera_num, dial_num)][4]
    ax.plot(data)
    LEN = len(date_time_index)
    new_x_ticks_index = []
    new_date_time_index = []
    for i in range(10):
        new_x_ticks_index.append(x_ticks_index[int(i*LEN/10)])
        new_date_time_index.append(date_time_index[int(i*LEN/10)])
    ax.set_xticks(new_x_ticks_index)
    labels = ax.set_xticklabels(new_date_time_index, rotation=30, fontsize='small')
    ax.set_title('Camera %s, Dial %s'%(camera_num,dial_num))
    ax.set_title(u'%s'%list_map['%s-%s'%(camera_num, dial_num)][0])
    ax.set_xlabel('Time')
    ax.set_ylabel('%s (%s)'%(list_map['%s-%s'%(camera_num, dial_num)][1],list_map['%s-%s'%(camera_num, dial_num)][2]))
    plt.show()    

def show_data(camera_num, dial_num, time_start, time_stop):
    db_data = read_data_db(camera_num=camera_num, dial_num=dial_num, time_start=time_start, time_stop=time_stop)
    plot_data(db_data)


def last_m_hour(camera_num, dial_num, m):###截至目前为止的前【m】个小时的数据绘图
    time_stop = time.time()
    time_start = time_stop - 3600*m
    show_data(camera_num, dial_num, time_start, time_stop)
    
    
if __name__ == '__main__':
    now_time = time.time()
    last_m_hour(0, 0, 1)
    '''
    start_time = time.mktime(time.strptime('2017-10-17 00:00:00','%Y-%m-%d %H:%M:%S'))
    stop_time = time.mktime(time.strptime('2017-10-18 00:04:00','%Y-%m-%d %H:%M:%S'))
    show_data(camera_num = 0, dial_num = 0, time_start = start_time, time_stop = stop_time)
    '''




