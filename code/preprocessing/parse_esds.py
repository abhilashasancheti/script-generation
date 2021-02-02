import os
import random
import json
from xml.dom import minidom
random.seed(42)

files = os.listdir('second_esd')
if '.DS_Store' in files:
	files.remove('.DS_Store')


files_test = os.listdir('../gold_paraphrase_sets/first_gold_annotation/')
if '.DS_Store' in files_test:
	files_test.remove('.DS_Store')


files_pilot =  os.listdir('pilot_esd')
if '.DS_Store' in files_pilot:
	files_pilot.remove('.DS_Store')


test_ind = random.sample(range(0, 9), 5)

valid_files = []
for i in range(10):
	if i not in test_ind:
		valid_files.append(files_test[i])

test_files = []
for i in range(10):
	if i in test_ind:
		test_files.append(files_test[i])


valid_indices = [j for j,scene in enumerate(files) if scene in valid_files]#test_valid[0:4]
test_indices = [j for j,scene in enumerate(files) if scene in test_files]#test_valid[4:]
test_valid = test_indices + valid_indices
print(valid_files)
# valid_indices = [j if scene in ['riding on a bus.new.xml', 'taking a bath.new.xml', 'flying in an airplane.new.xml', 'borrowing a book from the library.new.xm', 'baking a cake.new.xml'] for j, scene in enumerate(files)]
# test_indices = [j if scene in ['going on a train.new.xml','going grocery shopping.new.xml','getting a hair cut.new.xml','planting a tree.new.xml','repairing a flat bicycle tire.new.xml'] for j, scene in enumerate(files)]

##########################################################
# to identify the script ids for which annotations are available
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

# i=0
# with open('valid_1.txt', 'w') as out, open('valid_all_tokens_1.txt', 'w') as out2,  open('valid_tokens_1.txt', 'w') as out3:
# 	for f in files:
# 		if i in test_valid:
# 			script = f.split('.')[0].strip()
# 			xmldoc = minidom.parse('./second_esd/' + f)
# 			scripts = xmldoc.getElementsByTagName('script')
# 			for s in scripts:
# 				out.write("{} ".format("<BOS> here is a sequence of events that happen while "+ script + ":"))
# 				out2.write("{} ".format("<BOS> <SCR> "+ script + " <ESCR>:"))
# 				out3.write("{} ".format("<BOS> <SCR> "+ script + " <ESCR>:"))
# 				for item in s.childNodes:
# 					if item.attributes is not None:
# 						out.write("{} ".format(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip()))
# 						out2.write("{} ".format( '<BEVENT> '+ item.attributes['original'].value.lower().strip() + ' <EEVENT>'))
# 						out3.write("{} ".format(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip()))
# 				out.write("<EOS>\n")
# 				out2.write("<EOS>\n")
# 				out3.write("<EOS>\n")

# 			# ### also read the scripts from the pilot study
# 			script = f.split('.')[0].strip()
# 			f = script+'.pilot.xml'
# 			if f in files_pilot:
# 				xmldoc = minidom.parse('./pilot_esd/' + f)
# 				scripts = xmldoc.getElementsByTagName('script')
# 				for s in scripts:
# 					out.write("{} ".format("<BOS> here is a sequence of events that happen while "+ script + ":"))
# 					out2.write("{} ".format("<BOS> <SCR> "+ script + " <ESCR>:"))
# 					out3.write("{} ".format("<BOS> <SCR> "+ script + " <ESCR>:"))
# 					for item in s.childNodes:
# 						if item.attributes is not None:
# 							out.write("{} ".format(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip()))
# 							out2.write("{} ".format( '<BEVENT> '+ item.attributes['original'].value.lower().strip() + ' <EEVENT>'))
# 							out3.write("{} ".format(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip()))
# 					out.write("<EOS>\n")
# 					out2.write("<EOS>\n")
# 					out3.write("<EOS>\n")

# 		i+=1

# i=0
# with open('valid.txt', 'w') as out, open('valid_all_tokens.txt', 'w') as out2,  open('valid_tokens.txt', 'w') as out3:
# 	for f in files:
# 		if i in valid_indices:
# 			script = f.split('.')[0].strip()
# 			xmldoc = minidom.parse('./second_esd/' + f)
# 			scripts = xmldoc.getElementsByTagName('script')
# 			for s in scripts:
# 				if (s.attributes['source'].value, s.attributes['id'].value) in script_references[script]:
# 					out.write("{} ".format("<BOS> here is a sequence of events that happen while "+ script + ":"))
# 					out2.write("{} ".format("<BOS> <SCR> "+ script + " <ESCR>:"))
# 					out3.write("{} ".format("<BOS> <SCR> "+ script + " <ESCR>:"))
# 					for item in s.childNodes:
# 						if item.attributes is not None:
# 							out.write("{} ".format(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip()))
# 							out2.write("{} ".format( '<BEVENT> '+ item.attributes['original'].value.lower().strip() + ' <EEVENT>'))
# 							out3.write("{} ".format(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip()))
# 					out.write("<EOS>\n")
# 					out2.write("<EOS>\n")
# 					out3.write("<EOS>\n")
			
# 			script = f.split('.')[0].strip()
# 			f = script+'.pilot.xml'
# 			if f in files_pilot:
# 				xmldoc = minidom.parse('./pilot_esd/' + f)
# 				scripts = xmldoc.getElementsByTagName('script')
# 				for s in scripts:
# 					if (s.attributes['source'].value, s.attributes['id'].value) in script_references[script]:
# 						out.write("{} ".format("<BOS> here is a sequence of events that happen while "+ script + ":"))
# 						out2.write("{} ".format("<BOS> <SCR> "+ script + " <ESCR>:"))
# 						out3.write("{} ".format("<BOS> <SCR> "+ script + " <ESCR>:"))
# 						for item in s.childNodes:
# 							if item.attributes is not None:
# 								out.write("{} ".format(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip()))
# 								out2.write("{} ".format( '<BEVENT> '+ item.attributes['original'].value.lower().strip() + ' <EEVENT>'))
# 								out3.write("{} ".format(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip()))
# 						out.write("<EOS>\n")
# 						out2.write("<EOS>\n")
# 						out3.write("<EOS>\n")

# 		i+=1

# i=0
# with open('test.txt', 'w') as out, open('test_tokens.txt', 'w') as out2, open('test_references.txt', 'w') as out3, open('test_tokens_references.txt', 'w') as out4:
# 	for f in files:
# 		if i in test_indices:
# 			script = f.split('.')[0].strip()
# 			xmldoc = minidom.parse('./second_esd/' + f)
# 			scripts = xmldoc.getElementsByTagName('script')
# 			references = []
# 			tokens_references = []
# 			j=0
# 			for s in scripts:
# 				if (s.attributes['source'].value, s.attributes['id'].value) in script_references[script]:
# 					events = []
# 					tokens_events = []
# 					if j==0:
# 						out.write("{} ".format("<BOS> here is a sequence of events that happen while "+ script + ":"))
# 						out2.write("{} ".format("<BOS> <SCR> "+ script + " <ESCR>:"))
# 						out.write("\n")
# 						out2.write("\n")
# 					for item in s.childNodes:
# 						if item.attributes is not None:
# 							events.append(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip())
# 							tokens_events.append('<BEVENT> '+ item.attributes['original'].value.lower().strip() + ' <EEVENT>')
# 					references.append(events)
# 					tokens_references.append(tokens_events)
# 					j+=1

# 			script = f.split('.')[0].strip()
# 			f = script+'.pilot.xml'
# 			if f in files_pilot:
# 				xmldoc = minidom.parse('./pilot_esd/' + f)
# 				scripts = xmldoc.getElementsByTagName('script')
# 				for s in scripts:
# 					if (s.attributes['source'].value, s.attributes['id'].value) in script_references[script]:
# 						events = []
# 						tokens_events = []
# 						for item in s.childNodes:
# 							if item.attributes is not None:
# 								events.append(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip())
# 								tokens_events.append('<BEVENT> '+ item.attributes['original'].value.lower().strip() + ' <EEVENT>')
# 						references.append(events)
# 						tokens_references.append(tokens_events)

					
# 			out3.write("{}\n".format(references))
# 			out4.write("{}\n".format(tokens_references))
# 		i+=1


## valid references files
# i=0
# with open('valid_exp.txt', 'w') as out, open('valid_tokens_exp.txt', 'w') as out2, open('valid_references.txt', 'w') as out3, open('valid_tokens_references.txt', 'w') as out4:
# 	for f in files:
# 		if i in valid_indices:
# 			script = f.split('.')[0].strip()
# 			xmldoc = minidom.parse('./second_esd/' + f)
# 			scripts = xmldoc.getElementsByTagName('script')
# 			references = []
# 			tokens_references = []
# 			j=0
# 			for s in scripts:
# 				if (s.attributes['source'].value, s.attributes['id'].value) in script_references[script]:
# 					events = []
# 					tokens_events = []
# 					if j==0:
# 						out.write("{} ".format("<BOS> here is a sequence of events that happen while "+ script + ":"))
# 						out2.write("{} ".format("<BOS> <SCR> "+ script + " <ESCR>:"))
# 						out.write("\n")
# 						out2.write("\n")
# 					for item in s.childNodes:
# 						if item.attributes is not None:
# 							events.append(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip())
# 							tokens_events.append('<BEVENT> '+ item.attributes['original'].value.lower().strip() + ' <EEVENT>')
# 					references.append(events)
# 					tokens_references.append(tokens_events)
# 					j+=1

# 			script = f.split('.')[0].strip()
# 			f = script+'.pilot.xml'
# 			if f in files_pilot:
# 				xmldoc = minidom.parse('./pilot_esd/' + f)
# 				scripts = xmldoc.getElementsByTagName('script')
# 				for s in scripts:
# 					if (s.attributes['source'].value, s.attributes['id'].value) in script_references[script]:
# 						events = []
# 						tokens_events = []
# 						for item in s.childNodes:
# 							if item.attributes is not None:
# 								events.append(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip())
# 								tokens_events.append('<BEVENT> '+ item.attributes['original'].value.lower().strip() + ' <EEVENT>')
# 						references.append(events)
# 						tokens_references.append(tokens_events)

					
# 			out3.write("{}\n".format(references))
# 			out4.write("{}\n".format(tokens_references))
# 		i+=1


# for f in files:
# 	script = f.split('.')[0].strip()
# 	xmldoc = minidom.parse('./second_esd/' + f)
# 	scripts = xmldoc.getElementsByTagName('script')
# 	print(len(scripts))

def create_k_fold_test_references():
	references = {}
	tokens_references = {}
	for f in files:
		script = f.split('.')[0].strip()
		xmldoc = minidom.parse('./second_esd/' + f)
		scripts = xmldoc.getElementsByTagName('script')
		references[script] = []
		tokens_references[script] = []
		for s in scripts:
			events = []
			tokens_events = []
			for item in s.childNodes:
				if item.attributes is not None:
					events.append(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip())
					tokens_events.append('<BEVENT> '+ item.attributes['original'].value.lower().strip() + ' <EEVENT>')
			references[script].append(events)
			tokens_references[script].append(tokens_events)

		script = f.split('.')[0].strip()
		f = script+'.pilot.xml'
		if f in files_pilot:
			xmldoc = minidom.parse('./pilot_esd/' + f)
			scripts = xmldoc.getElementsByTagName('script')
			for s in scripts:
				events = []
				tokens_events = []
				for item in s.childNodes:
					if item.attributes is not None:
						events.append(str(item.attributes['slot'].value) + '. ' + item.attributes['original'].value.lower().strip())
						tokens_events.append('<BEVENT> '+ item.attributes['original'].value.lower().strip() + ' <EEVENT>')
				references[script].append(events)
				tokens_references[script].append(tokens_events)

	with open('test_k_fold_references.json', 'w') as f, open('test_k_fold_tokens_references.json', 'w') as g:
		json.dump(references,f)
		json.dump(tokens_references,g)


create_k_fold_test_references()
