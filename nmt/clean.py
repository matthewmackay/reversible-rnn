import xml.etree.ElementTree as ET
import glob
import io
import os


for f_xml in glob.iglob('data/en-de/*.xml'):
    print(f_xml)
    f_txt = os.path.splitext(f_xml)[0]
    with io.open(f_txt, mode='w', encoding='utf-8') as fd_txt:
        root = ET.parse(f_xml).getroot()[0]
        for doc in root.findall('doc'):
            for e in doc.findall('seg'):
                fd_txt.write(e.text.strip() + '\n')

xml_tags = ['<url', '<keywords', '<talkid', '<description', '<reviewer', '<translator', '<title', '<speaker']

for f_orig in glob.iglob('data/en-de/train.tags*'):
    print(f_orig)
    f_txt = f_orig.replace('.tags', '')
    with io.open(f_txt, mode='w', encoding='utf-8') as fd_txt, io.open(f_orig, mode='r', encoding='utf-8') as fd_orig:
        for l in fd_orig:
            if not any(tag in l for tag in xml_tags):
                fd_txt.write(l.strip() + '\n')
