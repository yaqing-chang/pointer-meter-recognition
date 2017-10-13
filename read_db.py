import pymysql
import datetime
import matplotlib.pyplot as plt
import time

def read_data_db(camera_num,dial_num,time_start,time_stop):
    conn = pymysql.connect(user='root', passwd='941120', db='bhxz')
    cursor = conn.cursor()
    sql = r'SELECT * FROM test WHERE time>="%s" and time<="%s" and name=%s-%s'%(time_start,time_stop,camera_num,dial_num)
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
    #plt.ylim(0,120)
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
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    plt.show()
    conn.close()

def read_data(camera_num=0,dial_num=0,time_start="s",time_stop="s"):
    if time_start== 's' and time_stop == 's':
        time_stop = datetime.datetime.now().strftime("%H:%M:%S")
        time_start = (datetime.datetime.now()+datetime.timedelta(hours=-1)).strftime("%H:%M:%S")
    read_data_db(camera_num=camera_num,dial_num=dial_num,time_start=time_start,time_stop=time_stop)
    
if __name__ == '__main__':
    read_data(time_start=1507361110,time_stop=1507472449)
