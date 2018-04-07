#!usr/bin/env python
# -*- coding:utf-8 -*-

import re
import logging
import sys
import os
import json
import random

from deeptext.models.sequence_labeling.biLSTM_crf_sequence_labeling import BiCrfSequenceLabeling
from deeptext.run import deep_ner




def get_entity(model, input_file,output_file):
    output_f = open(output_file,'w')
    with open(input_file) as f:
        for line in f:
            skill_entitys = set()
            line = unicode(line.strip(),'utf-8')
            sublines = re.split(ur'[！？｡。，,；.]', line)
            for subline in sublines:
                if subline == '':
                    continue
                if len(subline) == 1:
                    continue
                # ner
                sentence = subline.lower().replace(' ','')
                sentence = u'^' + sentence + u'$'
                char_list = [list(sentence)]

                labels = model.predict(char_list)
                labels = labels[0][0].split()

                for i, char in zip(range(len(labels) - 1), char_list[0]):
                    if labels[i] == 'E' and labels[i - 1] == 'O':
                        new_entity = ''
                        new_entity = new_entity + char
                    elif labels[i] == 'E' and labels[i + 1] == 'E':
                        new_entity = new_entity + char
                    elif labels[i] == 'E' and labels[i + 1] == 'O':
                        new_entity = new_entity + char
                        skill_entitys.add(new_entity)
            print '\t'.join(skill_entitys)
            output_f.write('\t'.join(skill_entitys)+'\n')
    output_f.close()

def get_RDF(input_file1,input_file2,output_file):
    job_list = []
    with open(input_file1) as f:
        for line in f:
            line = unicode(line.strip(), 'utf-8')
            line_list = line.split('\t')
            job_list.append((line_list[0],line_list[2]))

    skill_list = []
    with open(input_file2) as f:
        for line in f:
            line = unicode(line.strip(), 'utf-8')
            line_list = line.split('\t')
            skill_list.append(line_list)

    jobs_skills = []
    jobs = {}

    i = 0
    for job,idx in job_list:
        if job not in jobs.keys():
            job_skill = {}
            job_skill['job'] = job
            job_skill['count'] = 1
            job_skill['skills'] = []

            jobs_skills.append(job_skill)
            jobs[job] = i
            i += 1
        else:
            jobs_skills[jobs[job]]['count'] += 1

        # get idx data
        skills = {}
        for skill_idx in range(len(jobs_skills[jobs[job]]['skills'])):
            skills[jobs_skills[jobs[job]]['skills'][skill_idx]['name']] = skill_idx

        for skill in skill_list[int(idx)]:
            if skill not in skills.keys():
                skill_dict = {}
                skill_dict['name'] = skill
                skill_dict['count'] = 1

                jobs_skills[jobs[job]]['skills'].append(skill_dict)
            else:
                jobs_skills[jobs[job]]['skills'][skills[skill]]['count'] += 1

    print jobs
    output_f = open(output_file,'w')
    output_f.write(json.dumps(jobs_skills))
    print 'done.'


def RDF2graph(input_file,output_file):
    RDF_str = unicode(open(input_file).readline().strip(),'utf-8')
    jobs_skills = json.loads(RDF_str)

    graph = {}
    graph['nodes'] = []
    graph['edges'] = []

    skill_names = []

    for i in range(len(jobs_skills)):
        color = '#%02X%02X%02X' % (0, 0, 0)

        x = random.uniform(-100,100)
        y = random.uniform(-100,100)
        key_node = {}
        key_node['attributes'] = {}
        key_node['size'] = jobs_skills[i]['count']/10.
        key_node['color'] = color
        key_node['id'] = jobs_skills[i]['job']
        key_node['label'] = jobs_skills[i]['job']
        key_node['x'] = x
        key_node['y'] = y
        graph['nodes'].append(key_node)
        skill_names.append(jobs_skills[i]['job'])

        for skill in jobs_skills[i]['skills']:
            r = lambda: random.randint(0, 255)
            color = '#%02X%02X%02X' % (r(), r(), r())

            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)

            if skill['name'] not in skill_names:

                key_node = {}
                key_node['attributes'] = {}
                key_node['size'] = skill['count']
                key_node['color'] = color
                key_node['id'] = skill['name']
                key_node['label'] = skill['name']
                key_node['x'] = x
                key_node['y'] = y

                graph['nodes'].append(key_node)
                skill_names.append(skill['name'])
            else:
                key_node['size'] = key_node['size'] + skill['count']/10.

            # add adge
            edge = {}
            edge['attributes'] = {}
            edge['size'] = 1
            edge['sourceID'] = jobs_skills[i]['job']
            edge['targetID'] = skill['name']

            graph['edges'].append(edge)

    with open(output_file,'w') as f:
        f.write('var json = '+ json.dumps(graph))




if __name__=='__main__':

    # FORMAT = '[%(asctime)-15s] %(message)s'
    # logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    #
    # params = {}
    #
    # params["max_document_len"] = 25
    # params["embedding_size"] = 100
    # params["dropout_prob"] = 0.5
    #
    # params["model_name"] = 'model_' + str(params["embedding_size"]) + '_' + str(params["max_document_len"])
    # params["model_dir"] = './models'
    #
    # model = BiCrfSequenceLabeling(params)
    #
    # model.pre_restore()
    # model.restore(model.sess)
    #
    # get_entity(model,'./data/jobDesCleaning2.txt','./data/job_skills.txt')

    # deep_ner('test','./models','./data/skill_entity_train_data.txt','./data/skill_entity_valid_data.txt')
    # get_RDF('./data/title_classfy.txt','./data/job_skills.txt','./data/job_skills.json')
    # RDF2graph('./data/job_skills.json','./data/job_graph.json')
    RDF2graph('./data/job_skills.json', './data/all_jobs_skills.js')

