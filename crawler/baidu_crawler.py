#!usr/python/env
# -*- coding:utf-8 -*-

import requests
from bs4 import BeautifulSoup
import sys
import time
import random
import json
reload(sys)
sys.setdefaultencoding('utf-8')


def get_list(page):
    position_list = []
    url = 'http://zhaopin.baidu.com/quanzhi?query=%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0'
    html = requests.get(url, timeout=30).text
    soup = BeautifulSoup(html, 'lxml')
    results = soup.findAll('td', {'class': 'zwmc'})
    for result in results:
        detail_url = result.find_all('a')[0]['href']
        position_list.append(detail_url)
    return position_list

def get_position_info(position_list):
    position_info_list = []

    for url in position_list:
        time.sleep(random.randint(1, 5))
        if 'xiaoyuan' in url:
            continue
        else:
            html = requests.get(url, timeout=30).text
            soup = BeautifulSoup(html, 'lxml')
            positon = soup.h1.text
            print positon
            requirement = soup.find_all('div', {'class': 'tab-inner-cont'})[0].text
            print requirement
        position_info_list.append((positon,requirement.replace('\n',' ')))
    return position_info_list

if __name__ == '__main__':

    # po = get_list(1)
    # get_position_info(po)

    # pages = 90
    out_filename = '../data/baidu_jobs.json'
    out_file = open(out_filename,'a')
    #
    # for i in range(pages):
    #     position_list = get_list(i+1)
    #     position_info_list = get_position_info(position_list)
    #     for position_info in position_info_list:
    #         out_file.write(position_info[0]+'\t'+position_info[1]+'\n')
    # out_file.close()

    pages = 38
    for i in range(pages):
        time.sleep(random.randint(1, 5))
        url = 'http://zhaopin.baidu.com/api/quanzhiasync?query=%E7%AE%97%E6%B3%95&sort_type=1&city=%E5%8C%97%E4%BA%AC&detailmode=close&rn=20&pn='+str(i+1)
        json_info = requests.get(url, timeout=30).json()
        out_file.write(json.dumps(json_info)+'\n')
        if i % 5 == 0:
            print 'handle pages %d' % i
    out_file.close()

    '深度学习 '
    '自然语言处理：http://zhaopin.baidu.com/quanzhi?query=%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86'
    '图像处理 http://zhaopin.baidu.com/quanzhi?query=%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86'
    '语音识别 http://zhaopin.baidu.com/quanzhi?query=%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB'
    '机器视觉 http://zhaopin.baidu.com/quanzhi?query=%E6%9C%BA%E5%99%A8%E8%A7%86%E8%A7%89'
    '模式识别 http://zhaopin.baidu.com/quanzhi?query=%E6%A8%A1%E5%BC%8F%E8%AF%86%E5%88%AB'
    '自动驾驶 http://zhaopin.baidu.com/quanzhi?query=%E8%87%AA%E5%8A%A8%E9%A9%BE%E9%A9%B6'
    '算法 http://zhaopin.baidu.com/quanzhi?query=%E7%AE%97%E6%B3%95'