import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename
from moviepy.editor import VideoFileClip
from threading import Thread
from PIL import Image, ImageTk
from itertools import count

root = tk.Tk()
root.geometry('400x600+100+100')
isplaying = False

def play_video(video):
    vw = video.w
    vh = video.h
    ###逐帧播放画面
    for frame in video.iter_frames(fps=video.fps/2.5):
        if not isplaying:
            break
        w = root.winfo_width()
        h = root.winfo_height()
        ###保持原视频的纵横比
        ratio = min(w/vw, h/vh)
        size = (int(vw*ratio),int(vh*ratio))
        frame = Image.fromarray(frame).resize(size)
        frame = ImageTk.PhotoImage(frame)
        lbVideo['image'] = frame
        lbVideo.image = frame
        lbVideo.update()



def open_video():
    global isplaying
    isplaying = False
    fn = askopenfilename(title='打开视频文件',
                         filetypes=[('视频','*.mp4 *.avi')])
    if fn:
        root.title(f'正在检测“{fn}”')
        video = VideoFileClip(fn)
        isplaying = True
        ###播放视频的线程
        t = Thread(target=play_video, args=(video,))
        t.daemon = True
        t.start()


menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='文件',menu=filemenu),\
filemenu.add_command(label='打开视频文件',command=open_video),filemenu.add_command(label='新建'),filemenu.add_command(label='保存')
root.config(menu=menubar)

##面板
pw = tk.PanedWindow(root, orient='vertical', sashrelief='sunken')
pw.pack(fill='both', expand=1)
pw_1 = tk.PanedWindow(pw, orient='horizontal', sashrelief='sunken')
pw_2 = tk.PanedWindow(pw, orient='horizontal', sashrelief='sunken')
left_frame, right_frame, bottom_frame = ttk.Frame(pw_1, width=500, relief='raised'),\
                                        ttk.Frame(pw_1, height=500, relief='raised'),\
                                        ttk.Frame(pw_2, relief='raised')

pw.add(pw_1), pw.add(pw_2), pw_1.add(left_frame), pw_1.add(right_frame), #pw_2.add(bottom_frame)
lbVideo = tk.Label(left_frame, bg='black')
lbVideo.pack(fill='both', expand=1)


root.mainloop()
