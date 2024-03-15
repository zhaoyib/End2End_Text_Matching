'''
@File    :   logger.py
@Time    :   2024/03/11 11:46:05
@Author  :   YiboZhao 
@Version :   1.0
@Site    :   https://github.com/zhaoyib
'''
import logging

def logger_wrapper(name='JPM model'):
    logging.basicConfig(format='%(asctime)s - [%(levelname)s] -%(name)s->>>    %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(name)
    return logger