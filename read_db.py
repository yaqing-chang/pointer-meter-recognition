import pymysql
import datetime
import matplotlib
import matplotlib.pyplot as plt
import time
import numpy as np

from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False 


list_map = {'0-0':[u'150FI前润滑油压【LHP 150LP】',u'压力','bar',10,0.1],
            '0-1':[u'150FI后润滑油压【LHP 151LP】',u'压力','bar',10,0.1],
            '0-2':[u'150FI前润滑油压【LHP 152P】',u'压力','bar',10,0.1],
            '0-3':[u'150FI后润滑油压【LHP 153LP】',u'压力','bar',10,0.1],
            '0-4':[u'100FI前燃油压力【LHP 100LP】',u'压力','bar',6,0.05],
            '0-5':[u'100FI后燃油压力【LHP 101LP】',u'压力','bar',6,0.05],
            '0-6':[u'001MOA侧助燃空气压力【LHP 300LP】',u'压力','bar',4,0.05],
            '0-7':[u'001MOB侧助燃空气压力【LHP 301LP】',u'压力','bar',4,0.05]
            }


def read_data_db(camera_num,dial_num,time_start,time_stop):
    conn = pymysql.connect(user='root', passwd='941120', db='bhxz')
    cursor = conn.cursor()
    now_day = time.strftime("%Y%m%d", time.localtime())
    sql = r'SELECT * FROM data%s WHERE time>="%s" and time<="%s" and name="%s-%s"'%(now_day, time_start, time_stop, camera_num, dial_num)
    cursor.execute(sql)
    conn.commit()
    result = cursor.fetchall()
    data = []
    x_ticks_index = []
    date_time_index = []
    index = 0
    for i in result:
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
    conn.close()

def read_data(camera_num=0,dial_num=6,time_start="s",time_stop="s"):
    if time_start== 's' and time_stop == 's':
        time_stop = datetime.datetime.now().strftime("%H:%M:%S")
        time_start = (datetime.datetime.now()+datetime.timedelta(hours=-1)).strftime("%H:%M:%S")
    read_data_db(camera_num=camera_num,dial_num=dial_num,time_start=time_start,time_stop=time_stop)
    
if __name__ == '__main__':
    now_time = time.time()
    start_time = time.mktime(time.strptime('2017-10-18 00:00:00','%Y-%m-%d %H:%M:%S'))
    stop_time = time.mktime(time.strptime('2017-10-18 23:00:00','%Y-%m-%d %H:%M:%S'))
    read_data(time_start = start_time, time_stop = stop_time)
