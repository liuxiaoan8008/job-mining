#!usr/python/env
# -*- coding:utf-8 -*-

import requests
from bs4 import BeautifulSoup
import sys
import time
import random
reload(sys)
sys.setdefaultencoding('utf-8')
import json


def get_list(page):
    position_list = []
    url = 'http://sou.zhaopin.com/jobs/searchresult.ashx?jl=%E5%85%A8%E5%9B%BD&kw=%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86&sm=0&isfilter=0&fl=489&isadv=0&sb=1&sg=2e9c93378ef84471a56d71ee0ce1aaf5&p='+str(page)
    html = requests.get(url, timeout=30).text
    soup = BeautifulSoup(html, 'lxml')
    results = soup.findAll('td', {'class': 'zwmc'})
    for result in results:
        detail_url = result.find_all('a')[0]['href']
        position_list.append(detail_url)
        print detail_url
    return position_list

def get_position_info(position_list):
    position_info_list = []

    for url in position_list:
        # time.sleep(random.randint(1, 5))
        pos_info = {}
        if 'xiaoyuan' in url:
            continue
        else:
            html = requests.get(url, timeout=30).text
            soup = BeautifulSoup(html, 'lxml')
            positon = soup.h1.text
            try:
                requirement = soup.find_all('div', {'class': 'tab-inner-cont'})[0].text
                pos_info['positon'] = positon
                pos_info['requirement'] = requirement
                position_info_list.append(pos_info)
            except:
                continue
    return position_info_list

if __name__ == '__main__':

    # po = get_list(1)
    # get_position_info(po)

    pages = 6

    out_filename = '../data/zhilian_job_description.json'
    out_file1 = open(out_filename,'a')

    for i in range(pages):
        position_list = get_list(i+1)
        position_info_list = get_position_info(position_list)
        out_file1.write(json.dumps(position_info_list))
        print 'handle page: %d' % i
    out_file1.close()

    'http://sou.zhaopin.com/jobs/searchresult.ashx?jl=%E5%85%A8%E5%9B%BD&kw=%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0&sb=1&sm=0&p=90&isfilter=0&fl=489&isadv=0&sg=2e9c93378ef84471a56d71ee0ce1aaf5'