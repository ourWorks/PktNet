# 复制自 https://blog.csdn.net/zhanxiaohai/article/details/123851624

import logging
from datetime import datetime

liveHandlers = [] # 如果不做删除的操作，那么每一次记录的时候，都会写入多份文件

def get_logger(filename):

    for item in liveHandlers:
        logging.getLogger().removeHandler(item)

    # 创建一个文件处理器
    file_handler = logging.FileHandler(filename, mode='w+', encoding='utf-8')

    # 将文件处理器添加到根记录器中
    logging.getLogger().addHandler(file_handler)
    liveHandlers.append(file_handler)

    # 设置日志级别
    logging.getLogger().setLevel(logging.DEBUG)

    return logging.getLogger()
 
# if __name__ == "__main__":
#     log1 = get_logger('logger1')
#     log2 = get_logger('logger2')
 
#     log1.debug("This is a debug log.")
#     log1.info("This is a info log.")
#     log1.warning("This is a warning log.")
#     log1.error("This is a error log.")
#     log1.critical("This is a critical log.")
 
#     log2.debug("This is a debug log.")
#     log2.info("This is a info log.")
#     log2.warning("This is a warning log.")
#     log2.error("This is a error log.")
#     log2.critical("This is a critical log.")