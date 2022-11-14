import json
import os
import re
import numpy as np

# subroutine section
def merg_JsonFiles(filename):
    result = list()
    for f1 in filename:
        with open(f1, 'rb') as infile:
            result.extend(json.load(infile))

    return result

def getChinese(context):
    filtrate = re.compile(u'[^\u4E00-\u9FA5]')  # non-Chinese unicode range
    context = filtrate.sub(r'', context)  # remove all non-Chinese characters

    return context
# end subroutine section

# get the current working directory
cwd = os.getcwd()
train_label_path = cwd+'/train/label'
train_seq_in_path = cwd+'/train/seq.in'
train_seq_out_path = cwd + '/train/seq.out'
intent_label_path = 'intent_label.txt'
slot_label_path = 'slot_label.txt'
dev_label_path = cwd+'/dev/label'
dev_seq_in_path = cwd+'/dev/seq.in'
dev_seq_out_path = cwd + '/dev/seq.out'
test_label_path = cwd+'/test/label'
test_seq_in_path = cwd+'/test/seq.in'
test_seq_out_path = cwd + '/test/seq.out'

train_source_files = [cwd + '/FewJoint/SMP_Final_Origin2_1/train/source.json']

# set up development set and test set ratio
testRatio = 0.1
devRatio = 0.1

# prepare all training/dev/test data
train_data = merg_JsonFiles(train_source_files)

intent_data = list()
slot_data = list()

# calculate test and dev data number
# randomly divide data index
total_data_num = len(train_data)
test_data_num = int(total_data_num * testRatio)
dev_data_num = int(total_data_num * devRatio)
train_data_num = total_data_num - dev_data_num - test_data_num
total_shuffled_index = list(range(total_data_num))
np.random.shuffle(total_shuffled_index)
train_shuffled_index = total_shuffled_index[:train_data_num-1]
test_shuffled_index = total_shuffled_index[train_data_num:train_data_num+test_data_num-1]
dev_shuffled_index = total_shuffled_index[train_data_num+test_data_num:train_data_num+test_data_num+dev_data_num-1]

# convert train json data to text
with open(train_label_path, 'w', encoding='utf-8') as label_f, open(train_seq_in_path, 'w', encoding='utf-8') as seq_in_f, open(train_seq_out_path, 'w', encoding='utf-8') as seq_out_f:

    for key in train_shuffled_index:
        seq_in = train_data[key]['text'].replace(u'\xa0', u' ').replace(u'\x20', u'')

        current_intent_name = train_data[key]['intent']
        label_f.write(current_intent_name)
        label_f.write('\n')
        # check intent label and save for global intent label text
        if current_intent_name not in intent_data:
            # can't find current intent name in temporal intent data, so add to the tail of intent_data
            intent_data.append(current_intent_name)

        # create a default seq_in_list
        seq_in_list = list(seq_in)
        seq_out_list = list(seq_in)
        for indSeqIn in range(len(seq_in_list)):
            if indSeqIn < len(seq_in_list):
                seq_in_list[indSeqIn] = seq_in_list[indSeqIn]+' '
                seq_out_list[indSeqIn] = 'O '
            else:
                seq_out_list[indSeqIn] = 'O'

        seq_in_new = "".join(seq_in_list)
        seq_in_f.write(seq_in_new)
        seq_in_f.write('\n')

        # create seq_out by comparing slots to text
        # find out meaningful slots name in text
        for skey in train_data[key]['slots']:
            current_slot_name = train_data[key]['domain']+'-'+skey
            # check slot label and save for global slot label text
            if current_slot_name not in slot_data:
                # can't find current slot name in temporal slot data, so add to the tail of slot_data
                slot_data.append(current_slot_name)

            for sskey in range(len(train_data[key]['slots'][skey])):
                current_text_pattern = train_data[key]['slots'][skey][sskey]
                try:
                    current_text_index = seq_in.index(current_text_pattern)
                    # replace default slot value to current slot label
                    if sskey > 0:
                        seq_out_list[current_text_index] = 'I-'+current_slot_name+' '
                    else:
                        seq_out_list[current_text_index] = 'B-'+current_slot_name+' '

                    # check slot label and save for global slot label text
                    if seq_out_list[current_text_index] not in slot_data:
                        # can't find current slot name in temporal slot data, so add to the tail of slot_data
                        slot_data.append(seq_out_list[current_text_index])
                except ValueError:
                    print(seq_out_list)

        seq_out = "".join(seq_out_list)
        seq_out_f.write(seq_out)
        seq_out_f.write('\n')

# convert dev json data to text
with open(dev_label_path, 'w', encoding='utf-8') as label_f, open(dev_seq_in_path, 'w', encoding='utf-8') as seq_in_f, open(dev_seq_out_path, 'w', encoding='utf-8') as seq_out_f:

    for key in dev_shuffled_index:
        seq_in = train_data[key]['text'].replace(u'\xa0', u' ').replace(u'\x20', u'')

        current_intent_name = train_data[key]['intent']
        label_f.write(current_intent_name)
        label_f.write('\n')

        # create a default seq_in_list
        seq_in_list = list(seq_in)
        seq_out_list = list(seq_in)
        for indSeqIn in range(len(seq_in_list)):
            if indSeqIn < len(seq_in_list):
                seq_in_list[indSeqIn] = seq_in_list[indSeqIn] + ' '
                seq_out_list[indSeqIn] = 'O '
            else:
                seq_out_list[indSeqIn] = 'O'

        seq_in_new = "".join(seq_in_list)
        seq_in_f.write(seq_in_new)
        seq_in_f.write('\n')

        # create seq_out by comparing slots to text
        # find out meaningful slots name in text
        for skey in train_data[key]['slots']:
            current_slot_name = train_data[key]['domain']+'-'+skey

            for sskey in range(len(train_data[key]['slots'][skey])):
                current_text_pattern = train_data[key]['slots'][skey][sskey]
                try:
                    current_text_index = seq_in.index(current_text_pattern)
                    # replace default slot value to current slot label
                    if sskey > 0:
                        seq_out_list[current_text_index] = 'I-' + current_slot_name + ' '
                    else:
                        seq_out_list[current_text_index] = 'B-' + current_slot_name + ' '

                    # check slot label and save for global slot label text
                    if seq_out_list[current_text_index] not in slot_data:
                        # can't find current slot name in temporal slot data, so add to the tail of slot_data
                        slot_data.append(seq_out_list[current_text_index])
                except ValueError:
                    print(seq_out_list)

        seq_out = "".join(seq_out_list)
        seq_out_f.write(seq_out)
        seq_out_f.write('\n')

# convert test json data to text
with open(test_label_path, 'w', encoding='utf-8') as label_f, open(test_seq_in_path, 'w', encoding='utf-8') as seq_in_f, open(test_seq_out_path, 'w', encoding='utf-8') as seq_out_f:

    for key in test_shuffled_index:
        seq_in = train_data[key]['text'].replace(u'\xa0', u' ').replace(u'\x20', u'')

        current_intent_name = train_data[key]['intent']
        label_f.write(current_intent_name)
        label_f.write('\n')

        # create a default seq_in_list
        seq_in_list = list(seq_in)
        seq_out_list = list(seq_in)
        for indSeqIn in range(len(seq_in_list)):
            if indSeqIn < len(seq_in_list):
                seq_in_list[indSeqIn] = seq_in_list[indSeqIn]+' '
                seq_out_list[indSeqIn] = 'O '
            else:
                seq_out_list[indSeqIn] = 'O'

        seq_in_new = "".join(seq_in_list)
        seq_in_f.write(seq_in_new)
        seq_in_f.write('\n')

        # create seq_out by comparing slots to text
        # find out meaningful slots name in text
        for skey in train_data[key]['slots']:
            current_slot_name = train_data[key]['domain']+'-'+skey

            for sskey in range(len(train_data[key]['slots'][skey])):
                current_text_pattern = train_data[key]['slots'][skey][sskey]
                try:
                    current_text_index = seq_in.index(current_text_pattern)
                    # replace default slot value to current slot label
                    if sskey > 0:
                        seq_out_list[current_text_index] = 'I-' + current_slot_name + ' '
                    else:
                        seq_out_list[current_text_index] = 'B-' + current_slot_name + ' '

                    # check slot label and save for global slot label text
                    if seq_out_list[current_text_index] not in slot_data:
                        # can't find current slot name in temporal slot data, so add to the tail of slot_data
                        slot_data.append(seq_out_list[current_text_index])
                except ValueError:
                    print(seq_out_list)

        seq_out = "".join(seq_out_list)
        seq_out_f.write(seq_out)
        seq_out_f.write('\n')

# write global intent and slot label text
with open(intent_label_path, 'w', encoding='utf-8') as intent_label_f:
    intent_label_f.write('UNK')
    intent_label_f.write('\n')
    for key in range(len(intent_data)):
        intent_name = intent_data[key]
        intent_label_f.write(intent_name)
        intent_label_f.write('\n')

with open(slot_label_path, 'w', encoding='utf-8') as slot_label_f:
    slot_label_f.write('PAD')
    slot_label_f.write('\n')
    slot_label_f.write('UNK')
    slot_label_f.write('\n')
    slot_label_f.write('O')
    slot_label_f.write('\n')
    for key in range(len(slot_data)):
        slot_name = slot_data[key]
        slot_label_f.write(slot_name)
        slot_label_f.write('\n')
