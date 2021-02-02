from bs4 import BeautifulSoup
import os
import random
from xml.dom import minidom
random.seed(42)

files_annotation = os.listdir('../gold_paraphrase_sets/first_gold_annotation/')
if '.DS_Store' in files_annotation:
    files_annotation.remove('.DS_Store')



for f in files_annotation:
    script_ids = []
    xmldoc = minidom.parse('../gold_paraphrase_sets/first_gold_annotation/' + f)
    scripts = xmldoc.getElementsByTagName('item')
    for s in scripts:
        if s.attributes is not None:
            script_ids.append((s.attributes['source'].value,s.attributes['script'].value))
    
    xmldoc = minidom.parse('../gold_paraphrase_sets/second_gold_annotation/' + f)
    scripts = xmldoc.getElementsByTagName('item')
    for s in scripts:
        if s.attributes is not None:
            script_ids.append((s.attributes['source'].value, s.attributes['script'].value))

    script_ids = list(set(script_ids))
    print("file", f, len(script_ids))

