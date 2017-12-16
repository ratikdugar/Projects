
# coding: utf-8

# In[1]:


#import libaries
import os
import pprint
import xml.etree.cElementTree as ET
import re
import codecs
import csv
import cerberus
import copy
import sqlite3
import schema
import pandas as pd
from collections import defaultdict
OSM_FILE = "sfo.osm"  # Replace this with your osm file
TEST_FILE = "test-sfo.osm"
SAMPLE_FILE="sample-sfo.osm"



# In[2]:


def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag

    Reference:
    http://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
    """
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


# In[ ]:


k = 100 # Parameter: take every k-th top level element
with open(SAMPLE_FILE, 'wb') as output:
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write('<osm>\n  ')

    # Write every kth top level element
    for i, element in enumerate(get_element(OSM_FILE)):
        if i % k == 0:
            output.write(ET.tostring(element, encoding='utf-8'))

    output.write('</osm>')


# In[3]:


#!/usr/bin/env python

#I iteratively parse through the map dataset and count the unique element types by using count_tag function to get a
#feeling on how much of which data you can expect to have in the map.
def count_tags(filename):
    tags={}
    for event,elem in ET.iterparse(filename):
        if elem.tag not in tags:
            tags[elem.tag]=1
        else:
            tags[elem.tag]+=1
    return tags

def test():
    tags = count_tags(TEST_FILE)
    pprint.pprint(tags)
if __name__ == "__main__":
    test()


# In[4]:


'''checks the "k" value for each "<tag>" and sees if there are any potential problems.'''

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

def key_type(element, keys):
    '''Finds the count of each of four tag categories in a dictionary:
  "lower", for tags that contain only lowercase letters and are valid,
  "lower_colon", for otherwise valid tags with a colon in their names,
  "problemchars", for tags with problematic characters, and
  "other", for other tags that do not fall into the other three categories.'''
    
    if element.tag == "tag":
        kay = element.attrib['k']
        if lower.search(kay):
            keys['lower'] += 1
        elif lower_colon.search(kay):
            keys['lower_colon'] += 1
        elif problemchars.search(kay):
            keys['problemchars'] += 1
        else:
            keys['other'] += 1
    return keys

def process_map(filename):
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for event, element in ET.iterparse(filename):
        keys = key_type(element, keys)

    return keys

def test():
    keys = process_map(TEST_FILE)
    pprint.pprint(keys)

if __name__ == "__main__":
    test()


# In[5]:


'''The function process_map should return a set of unique user IDs ("uid") 
and we use that to print how many unique users contributed to the dataset'''
def process_map(filename):
    users = set()
    for event, element in ET.iterparse(filename):
        if 'user' in element.attrib:
            users.add(element.attrib['user'])
            
    return users

unique_users = process_map(TEST_FILE)
pprint.pprint(len(unique_users))


# In[6]:


# Checks if it is a street name and returns the element where the value of the k attribute is "addr:street"  
def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)


# In[7]:


expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons", "Cove", "Alley", "Park", "Way", "Walk" "Circle", "Highway", 
            "Plaza", "Path", "Center", "Mission"]

#Finds all the different types of street not in expected and adds it to a dictionary
def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)
            
#Iteratively finds all the elements that are street addresses
def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types


st_types = audit(TEST_FILE)
pprint.pprint(dict(st_types))


# In[8]:



mapping = { "Ave": "Avenue",
            "Ave.": "Avenue",
            "avenue": "Avenue",
            "ave": "Avenue",
            "Blvd": "Boulevard",
            "Blvd.": "Boulevard",
            "Blvd,": "Boulevard",
            "Boulavard": "Boulevard",
            "Boulvard": "Boulevard",
            "Ct": "Court",
            "Ct.": "Court",
            "Dr": "Drive",
            "Dr.": "Drive",
            "E": "East",
            "E." : "East",
            "Hwy": "Highway",
            "Ln": "Lane",
            "Ln.": "Lane",
            "N.": "North",
            "N": "North",
            "Pl": "Place",
            "Plz": "Plaza",
            "Rd": "Road",
            "Rd.": "Road",
            "S": "South",
            "S.": "South",
            "St": "Street",
            "St.": "Street",
            "st": "Street",
            "street": "Street",
            "square": "Square",
            "parkway": "Parkway",
            "PKWY": "Parkway",
            "W.": "West",
            "W": "West"
            }


# In[9]:


# return the updated names
def update_name(name, mapping):
    words = name.split()
    for w in range(len(words)):
        if words[w] in mapping:
            if words[w-1].lower() not in ['suite', 'ste.', 'ste', 'avenue', 'ave'] or words[0] in mapping: 
            # For example, don't update 'Avenue E' to 'Avenue East'
                words[w] = mapping[words[w]]

    name = " ".join(words)
    return name


# In[10]:


# print the updated names
for st_type, ways in st_types.iteritems():
    for name in ways:
        better_name = update_name(name, mapping)
        print name, "=>", better_name


# In[11]:


# Checks if it is a zipcode and returns the element where the value of the k attribute is "addr:postcode"  or "postal_code"                                      
def is_zipcode(elem):
    return (elem.attrib['k'] == "addr:postcode" or elem.attrib['k'] == "postal_code")
zipcode_re = re.compile(r'^94\d\d\d$')


# In[12]:


#loops over the file and finds all the elements that are zipcodes
def audit_zip(osmfile):
    osm_file = open(osmfile, "r")
    invalid_zipcodes = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_zipcode(tag):
                    audit_zipcode(invalid_zipcodes,tag.attrib['v'])

    return invalid_zipcodes

#Finds if a zipcode is in the desirable format or not and based on that adds it to a set
def audit_zipcode(invalid_zipcodes, zipcode):
    firsttwo = zipcode[0:2]
    
    if firsttwo != '94' or not firsttwo.isdigit() or len(zipcode)>5:
        invalid_zipcodes[firsttwo].add(zipcode)
        
sf_zipcode = audit_zip(TEST_FILE)
pprint.pprint(dict(sf_zipcode))


# In[13]:


# Cleans zipcode formatting
def update_zipcode(postcode):
    try:
        postcode = re.compile(r'94\d\d\d').search(postcode).group()
    except AttributeError:
        postcode = 'None'
    return postcode


# In[14]:


#prints old zipcode and the new zipcode after the update
for event, elem in ET.iterparse(TEST_FILE, events=("start",)):
    if elem.tag == "node" or elem.tag == "way":
        for tag in elem.iter("tag"):
            if is_zipcode(tag):
                if zipcode_re.match(tag.attrib['v']) == None:
                    print tag.attrib['v'], "=>", update_zipcode(tag.attrib['v'])


# In[15]:


#combines all the update functions        
def clean(value, tag, mapping_street):
    if is_street_name(tag):
        value = update_name(value, mapping)
    elif is_zipcode(tag):
        value = update_zipcode(value)
    return value


# In[16]:


#prints all of the changes made together
counter = 0
for element in get_element(TEST_FILE):
    if element.tag == 'node':
        for tag in element.iter('tag'):
            if is_street_name(tag) or is_zipcode(tag):
                print tag.attrib['v'], '=>', clean(tag.attrib['v'], tag, mapping)
                counter += 1


# In[ ]:




