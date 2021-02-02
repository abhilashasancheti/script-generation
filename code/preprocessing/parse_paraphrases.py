import os
import random
import json
from xml.dom import minidom
random.seed(42)


files_test = os.listdir('../gold_paraphrase_sets/first_gold_annotation/')
if '.DS_Store' in files_test:
	files_test.remove('.DS_Store')


# files_pilot =  os.listdir('pilot_esd')
# if '.DS_Store' in files_pilot:
# 	files_pilot.remove('.DS_Store')


# test_ind = random.sample(range(0, 9), 5)

# valid_files = []
# for i in range(10):
# 	if i not in test_ind:
# 		valid_files.append(files_test[i])

# test_files = []
# for i in range(10):
# 	if i in test_ind:
# 		test_files.append(files_test[i])


# valid_indices = [j for j,scene in enumerate(files) if scene in valid_files]#test_valid[0:4]
# test_indices = [j for j,scene in enumerate(files) if scene in test_files]#test_valid[4:]
# test_valid = test_indices + valid_indices
# print(valid_files)
# # valid_indices = [j if scene in ['riding on a bus.new.xml', 'taking a bath.new.xml', 'flying in an airplane.new.xml', 'borrowing a book from the library.new.xm', 'baking a cake.new.xml'] for j, scene in enumerate(files)]
# # test_indices = [j if scene in ['going on a train.new.xml','going grocery shopping.new.xml','getting a hair cut.new.xml','planting a tree.new.xml','repairing a flat bicycle tire.new.xml'] for j, scene in enumerate(files)]

# ##########################################################
# # to identify the script ids for which annotations are available
# files_annotation = os.listdir('../gold_paraphrase_sets/first_gold_annotation/')
# if '.DS_Store' in files_annotation:
#     files_annotation.remove('.DS_Store')

# script_references = {}
# for f in files_annotation:
#     script_ids = []
#     xmldoc = minidom.parse('../gold_paraphrase_sets/first_gold_annotation/' + f)
#     scripts = xmldoc.getElementsByTagName('item')
#     for s in scripts:
#         if s.attributes is not None:
#             script_ids.append((s.attributes['source'].value,s.attributes['script'].value))
    
#     xmldoc = minidom.parse('../gold_paraphrase_sets/second_gold_annotation/' + f)
#     scripts = xmldoc.getElementsByTagName('item')
#     for s in scripts:
#         if s.attributes is not None:
#             script_ids.append((s.attributes['source'].value,s.attributes['script'].value))

#     script_ids = list(set(script_ids))
#     script_references[f.split('.')[0].strip()] = script_ids
#     print("file", f, len(script_ids))
# ##############################################################


scenarios = {}


for f in files_test:
	script = f.split('.')[0].strip()
	scenarios[script] = {}
	print(script)
	xmldoc = minidom.parse('../gold_paraphrase_sets/first_gold_annotation/' + f)
	labels = xmldoc.getElementsByTagName('label')
	for l in labels:
		scenarios[script][l.attributes['event'].value] = []
		for item in l.childNodes:
			if item.attributes is not None:
				scenarios[script][l.attributes['event'].value].append(str(item.attributes['original'].value).strip().lower())
	
	xmldoc = minidom.parse('../gold_paraphrase_sets/second_gold_annotation/' + f)
	labels = xmldoc.getElementsByTagName('label')
	for l in labels:
		if l.attributes['event'].value not in scenarios[script]:
			scenarios[script][l.attributes['event'].value] = []
		for item in l.childNodes:
			if item.attributes is not None:
				scenarios[script][l.attributes['event'].value].append(str(item.attributes['original'].value).strip().lower())
	


with open('paraphrase.json', 'w') as f:
	json.dump(scenarios,f)

