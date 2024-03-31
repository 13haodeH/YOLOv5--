# coding=utf-8
import json
import os
import threading

import msg_box


def thread_runner(func):
    def wrapper(*args, **kwargs):
        threading.Thread(target=func, args=args, kwargs=kwargs).start()

    return wrapper


class Global:
    def __init__(self):
        self.config = dict()

    def init_config(self):
        if not os.path.exists('config'):
            os.mkdir('config')  # make new config folder
            return
        try:
            with open('config/config.json', 'r') as file_settings:
                GLOBAL.config = json.loads(file_settings.read())
        except FileNotFoundError as err_file:
            print('配置文件不存在: ' + str(err_file))

    def record_config(self, _dict):
        # 更新配置
        for k, v in _dict.items():
            self.config[k] = v
        if not os.path.exists('config'):
            os.mkdir('config')
        try:
            # 写入文件
            with open('config/config.json', 'w') as file_config:
                file_config.write(json.dumps(self.config, indent=4))
        except FileNotFoundError as err_file:
            print(err_file)
            msg = msg_box.MsgWarning()
            msg.setText('参数保存失败！')
            msg.exec()


GLOBAL = Global()
