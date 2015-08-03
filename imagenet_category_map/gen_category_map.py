import json

inF = 'synset_words.txt'
f = open(inF, 'r')
line = f.readline()

category_map = dict() #category_map['tench'] = 0
category_map_flipped = dict() #category_map_flipped[0] = 'tench'

idx=0
while line:
    line = line.strip()
    print line
    split_line = line.split(' ')
    synsetID = split_line[0] #n02037110
    nicename = " ".join(split_line[1:]) #oystercatcher, oyster catcher

    category_map[nicename] = idx
    category_map_flipped[idx] = nicename
    idx = idx + 1
    line = f.readline()

with open('category_map.json', 'w') as outfile:
    json.dump(category_map, outfile, sort_keys=True)

with open('category_map_flipped.json', 'w') as outfile:
    json.dump(category_map_flipped, outfile, sort_keys=True)


