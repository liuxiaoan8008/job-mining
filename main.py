#!usr/bin/env python
# -*- coding:utf-8 -*-

from deeptext.run import deep_ner

if __name__=='__main__':
    deep_ner('train','./models/','./data/skill_entity_train_data.txt','./data/skill_entity_valid_data.txt')