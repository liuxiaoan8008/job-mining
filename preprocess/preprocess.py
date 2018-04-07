#!usr/bin/env python
# -*- coding:utf-8 -*-

import re

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# cut paragrah to lines
def rebuild_data(input_file,output_file):
    output_f = open(output_file,'w')
    with open(input_file) as f:
        for line in f:
            line = unicode(line.strip(),'utf-8')
            sublines = re.split(ur'[！？｡。，,；.]', line)
            for subline in sublines:
                if subline == '':
                    continue
                if len(subline) == 1:
                    continue
                output_f.write(subline.replace(' ','')+'\n')
    output_f.close()

# extract english skill name entity
def ex_skill_dict(input_file,output_file):
    skill_dict = {}
    with open(input_file) as f:
        for line in f:
            line = unicode(line.strip(), 'utf-8')
            if line == '':
                continue
            line = line.lower()
            results = re.findall(ur'[a-z+]+',line)
            if results:
                for res in results:
                    if res not in skill_dict.keys():
                        skill_dict[res] = 0
                    skill_dict[res] += 1

    print len(skill_dict)

    skill_dict = [(key,skill_dict[key]) for key in skill_dict.keys()]
    skill_dict_sorted = sorted(skill_dict,key=lambda x:x[1], reverse=True)
    print skill_dict_sorted[0]

    with open(output_file,'w') as f:
        for skill in skill_dict_sorted:
            if skill[1] >= 10:
                f.write(skill[0]+'\t'+str(skill[1])+'\n')
    print 'done!'

# easy skill sort
def skill_similarity(input_file,output_file):
    skill_list = []
    with open(input_file) as f:
        for line in f:
            line = unicode(line.strip(), 'utf-8')
            if line.startswith('+'):
                line = line[1:]
            if len(line) == 1 and line is not 'c':
                continue
            skill_list.append(line.split('\t')[0])

    skill_list_sort = sorted(skill_list)
    with open(output_file,'w') as f:
        for skill in skill_list_sort:
            f.write(skill+'\n')

def rebuild_entity(entity,sentence):
    # input unicode
    label = ['O' for i in range(len(sentence))]
    for e in entity:
        try:
            i = sentence.index(e)
            for index_e in range(i,i+len(e)):
                label[index_e] = 'E'
        except:
            continue
    return label

# build ner train data
def auto_label(input_file,output_entity_file,output_noentity_file):
    entitys_file = open('../data/skill_sim.txt')
    entitys = set()
    for line in entitys_file:
        line = unicode(line.strip(), 'utf-8')
        entitys.add(line)
    entitys_file.close()

    output_entity_f = open(output_entity_file, 'w')
    output_noentity_f = open(output_noentity_file, 'w')
    useful_count = 0

    with open(input_file) as f:
        for line in f:
            line = unicode(line.strip(), 'utf-8')
            line = line.lower()

            label = rebuild_entity(entitys, line)
            sentence_str = u' '.join(list(line))
            label_str = u' '.join(label)
            if 'E' in label:
                useful_count += 1
                output_entity_f.write('^ ' + sentence_str + ' $\n')
                output_entity_f.write('O ' + label_str + ' O\n')
            else:
                output_noentity_f.write('^ ' + sentence_str + ' $\n')
                output_noentity_f.write('O ' + label_str + ' O\n')

    print useful_count
    output_entity_f.close()
    output_noentity_f.close()


# easy title sort
def title_sort(input_file,output_file):
    skill_list = []
    with open(input_file) as f:
        for line in f:
            line = unicode(line.strip(), 'utf-8')
            skill_list.append(line)

    skill_list = list(enumerate(skill_list))
    print skill_list[0]
    skill_list_sort = sorted(skill_list,key=lambda x:x[1])
    with open(output_file,'w') as f:
        for skill in skill_list_sort:
            f.write(skill[1]+'\t'+str(skill[0])+'\n')


if __name__=='__main__':
    # rebuild_data('../data/jobDesCleaning2.txt','../data/jobDesClening2_sublines.txt')
    # ex_skill_dict('../data/jobDesClening2_sublines.txt','../data/skill_fre_10.txt')
    # skill_similarity('../data/skill_fre_10.txt','../data/skill_sim.txt')
    # auto_label('../data/jobDesClening2_sublines.txt','../data/jobDes_entity_data.txt','../data/jobDes_noentity_data.txt')
    title_sort('../data/title_result.txt','../data/title_result_sim.txt')