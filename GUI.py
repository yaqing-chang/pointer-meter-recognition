import sys    
import tkinter as tk
import datetime
import time
import os
import psutil

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(parentdir,'model')
sys.path.insert(0,model_dir)
from model import Universal_value

class state(Universal_value):
    def __init__(self):
        Universal_value.__init__(self)
        
    def ping(self,camera_num):
        import os
        failed = os.popen("ping %s"%self.ip_address[camera_num]).readlines()[8][30]
        if int(failed) == 0:
            exec("carera_%s_0.config(text='已连接',fg='green',font=('times', 12, 'bold'))"%camera_num)
        else:
            exec("carera_%s_0.config(text='未连接',fg='red',font=('times', 12, 'bold'))"%camera_num)

        
    def connect(self):
        import threading
        for camera_num in range(self.camera_nums):
            threading.Thread(target=self.ping,args = (camera_num,)).start()
        exec('carera_%s_0.after(3000, self.connect)'%camera_num)

def tick():
    global start_time
    time = (datetime.datetime.now()-start_time).seconds
    day = time//(24*60*60)
    hour = time%(24*60*60)//(60*60)
    minute = time%(24*60*60)%(60*60)//60
    second = time%60
    text = '运行时间：%d天  %02d:%02d:%02d'%(day,hour,minute,second)
    clock.config(text=text)
    clock.after(1000, tick)

def update_cpu():
    cpu_use_0.config(text='%s%%'%(psutil.cpu_percent()))
    cpu_use_0.after(2000, update_cpu)
def update_mem():
    mem_use_0.config(text='%.2fGB'%((psutil.virtual_memory().used)/1024/1024/1024))
    mem_use_0.after(2000, update_mem)
    
    
def start0():
    import subprocess
    child = subprocess.Popen(r"python C:\Users\caiwd\Desktop\viewsystem\trainwell\usemodel.py")
    start_button.config(text='正在识别',font=('times',12,'bold'),state='disable')

def start():
    import threading
    threading.Thread(target=start0).start()

def show_result():
    import read_db
    read_db.read_data()

start_time = datetime.datetime.now()
win = tk.Tk()
win.title("测试样机")

root = tk.LabelFrame(win,text='',relief=tk.FLAT)
root.pack(padx=8, pady=8)

root_title = tk.Label(root,text='大亚湾核电基地模拟仪表识别系统',fg='blue',font=("黑体", 24, "bold"))
root_title.grid(row=0, column=0, padx=15, pady=10)
root_main = tk.Label(root)
root_main.grid(row=1, column=0, padx=15, pady=10)

root_cmd = tk.LabelFrame(root_main,text='命令',fg='blue',font=2)
root_cmd.grid(row=1, column=0, padx=5, pady=5)
root_state_0 = tk.LabelFrame(root_main,text='系统状态',fg='blue',font=2)
root_state_0.grid(row=1, column=1, padx=5, pady=5)


start_button = tk.Button(root_cmd, text ='开始识别' ,font=('times', 12, 'bold'),command=start)
start_button.grid(row=0, column=0, padx=15, pady=10)
show = tk.Button(root_cmd, text ='显示曲线' ,font=('times', 12, 'bold'),command=show_result)
show.grid(row=1, column=0, padx=15, pady=10)

root_state = tk.Label(root_state_0)
root_state.grid(row=0, column=0, padx=5, pady=5)
#root_state_1 = tk.Label(root_state_0)
#root_state_1.grid(row=2, column=0, padx=5, pady=5)
root_state_2 = tk.Label(root_state_0)
root_state_2.grid(row=1, column=0, padx=5, pady=5)



carera_0 = tk.Label(root_state,text = '摄像头1:', font=('times', 12, 'bold'))
carera_0.grid(row=1, column=0)
carera_0_0 = tk.Label(root_state,text = '未连接', fg='red', font=('times', 12, 'bold'))
carera_0_0.grid(row=1, column=1, padx=10)

carera_1  = tk.Label(root_state,text = '摄像头2:', font=('times', 12, 'bold'))
carera_1.grid(row=1, column=2,padx=10)
carera_1_0 = tk.Label(root_state,text = '未连接', fg='red',font=('times', 12, 'bold'))
carera_1_0.grid(row=1, column=3)

carera_2  = tk.Label(root_state,text = '摄像头3:', font=('times', 12, 'bold'))
carera_2.grid(row=2, column=0)
carera_2_0 = tk.Label(root_state,text = '未连接', fg='red',font=('times', 12, 'bold'))
carera_2_0.grid(row=2, column=1, padx=10)

carera_3  = tk.Label(root_state,text = '摄像头4:', font=('times', 12, 'bold'))
carera_3.grid(row=2, column=2,padx=10)
carera_3_0 = tk.Label(root_state,text = '未连接', fg='red',font=('times', 12, 'bold'))
carera_3_0.grid(row=2, column=3)

'''
cpu_use  = tk.LabelFrame(root_state_1,text = 'CPU:', font=('times', 12, 'bold'))
cpu_use.grid(row=0, column=0,padx=10)
cpu_use_0 = tk.Label(root_state_1,text = '0%',font=('times', 12, 'bold'))
cpu_use_0.grid(row=0, column=1)
mem_use  = tk.Label(root_state_1,text = '内存:', font=('times', 12, 'bold'))
mem_use.grid(row=0, column=2,padx=10)
mem_use_0 = tk.Label(root_state_1,text = 'GB',font=('times', 12, 'bold'))
mem_use_0.grid(row=0, column=3)
'''


clock = tk.Label(root_state_2,text = '运行时间：00:00:00', font=('times', 12, 'bold'))
clock.grid(row=0, column=0) 
tick()
#update_cpu()
#update_mem()
state = state()
state.connect()
root.mainloop()
