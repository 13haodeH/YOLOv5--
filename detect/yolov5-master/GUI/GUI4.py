import tkinter as tk
from tkinter import *
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename
from moviepy.editor import VideoFileClip
from threading import Thread
from PIL import Image, ImageTk
from itertools import count
from time import sleep

root = tk.Tk()
root.title('视频检测器')
root.geometry('1200x640+200+100')

isplaying = False

###用来显示视频画面的Label组件
pw = tk.PanedWindow(root, orient='horizontal', sashrelief='sunken')
pw.pack(fill='both', expand=1)
pw_1 = tk.PanedWindow(pw, orient='vertical', sashrelief='sunken',width=600)
pw_2 = tk.PanedWindow(pw, orient='vertical', sashrelief='sunken',width=600)
pw.add(pw_1),pw.add(pw_2)
lbVideo = tk.Label(pw_1, bg='white',width=600)
lbVideo.pack(side=LEFT,expand=0,fill='both')

def play_video(video):
    vw = video.w
    vh = video.h
    ###逐帧播放画面
    for frame in video.iter_frames(fps=video.fps*2):
        if not isplaying:
            break
        w = pw_1.winfo_width()
        h = pw_1.winfo_height()
        ###保持原视频的纵横比
        ratio = min(w/vw, h/vh)
        size = (int(vw*ratio),int(vh*ratio))
        frame = Image.fromarray(frame).resize(size)
        frame = ImageTk.PhotoImage(frame)
        lbVideo['image'] = frame
        lbVideo.image = frame
        lbVideo.update()

###创建主菜单
mainMenu = tk.Menu(root)

###创建子菜单
subMenu = tk.Menu(tearoff=0)

def open_video():
    global isplaying
    isplaying = False
    fn = askopenfilename(title='打开视频文件',
                         filetypes=[('视频','*.mp4 *.avi *.MOV')])
    if fn:
        root.title(f'正在检测“{fn}”')
        video = VideoFileClip(fn)
        isplaying = True
        ###播放视频的线程
        t = Thread(target=play_video, args=(video,))
        t.daemon = True
        t.start()

###添加菜单，设置命令
subMenu.add_command(label='打开视频文件', command=open_video)
###把子菜单挂到主菜单
mainMenu.add_cascade(label='文件',menu=subMenu)
###把主菜单放置到窗口上
root['menu'] = mainMenu
##确保子线程关闭
def exiting():
    global isplaying
    isplaying = False
    sleep(0.05)
    root.destroy()
root.protocol('推出窗口', exiting)

root.mainloop()