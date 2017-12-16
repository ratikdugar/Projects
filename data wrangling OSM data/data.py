
# coding: utf-8

# In[6]:


"""
This program parses the elements of the OSM XML file, transforms them from 
document format to tabular format, thus making it possible to write to csv files. 
The csv files can then be imported to a SQL database as tables.
The process for this transformation is as follows:
- Use iterparse to iteratively step through each top level element in the XML
- Shape each element into several data structures using a custom function
- Utilize a schema and validation library to ensure the transformed data is in the correct format
- Write each data structure to the appropriate .csv files
"""
import pprint
import csv
import codecs
import re
import xml.etree.cElementTree as ET
import cerberus
import schema
from clean import clean


OSM_FILE = "sfo.osm"
SAMPLE_FILE = "sample-sfo.osm"
TEST_FILE = "test-sfo.osm"

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

NODES_PATH = "nodes1.csv"
NODE_TAGS_PATH = "node_tags1.csv"
WAYS_PATH = "ways1.csv"
WAY_NODES_PATH = "way_nodes1.csv"
WAY_TAGS_PATH = "way_tags1.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
schema = {
    'node': {
        'type': 'dict',
        'schema': {
            'id': {'required': True, 'type': 'integer', 'coerce': int},
            'lat': {'required': True, 'type': 'float', 'coerce': float},
            'lon': {'required': True, 'type': 'float', 'coerce': float},
            'user': {'required': True, 'type': 'string'},
            'uid': {'required': True, 'type': 'integer', 'coerce': int},
            'version': {'required': True, 'type': 'string'},
            'changeset': {'required': True, 'type': 'integer', 'coerce': int},
            'timestamp': {'required': True, 'type': 'string'}
        }
    },
    'node_tags': {
        'type': 'list',
        'schema': {
            'type': 'dict',
            'schema': {
                'id': {'required': True, 'type': 'integer', 'coerce': int},
                'key': {'required': True, 'type': 'string'},
                'value': {'required': True, 'type': 'string'},
                'type': {'required': True, 'type': 'string'}
            }
        }
    },
    'way': {
        'type': 'dict',
        'schema': {
            'id': {'required': True, 'type': 'integer', 'coerce': int},
            'user': {'required': True, 'type': 'string'},
            'uid': {'required': True, 'type': 'integer', 'coerce': int},
            'version': {'required': True, 'type': 'string'},
            'changeset': {'required': True, 'type': 'integer', 'coerce': int},
            'timestamp': {'required': True, 'type': 'string'}
        }
    },
    'way_nodes': {
        'type': 'list',
        'schema': {
            'type': 'dict',
            'schema': {
                'id': {'required': True, 'type': 'integer', 'coerce': int},
                'node_id': {'required': True, 'type': 'integer', 'coerce': int},
                'position': {'required': True, 'type': 'integer', 'coerce': int}
            }
        }
    },
    'way_tags': {
        'type': 'list',
        'schema': {
            'type': 'dict',
            'schema': {
                'id': {'required': True, 'type': 'integer', 'coerce': int},
                'key': {'required': True, 'type': 'string'},
                'value': {'required': True, 'type': 'string'},
                'type': {'required': True, 'type': 'string'}
            }
        }
    }
}
# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']


def shape_element(element, node_attr_fields = NODE_FIELDS, way_attr_fields = WAY_FIELDS,
                  problem_chars = PROBLEMCHARS, default_tag_type = 'regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  # Handle secondary tags the same way for both node and way elements

    if element.tag == 'node':
        for attribute in NODE_FIELDS:
            # if the key exists
            if element.attrib.get(attribute):
                node_attribs[attribute] = element.attrib[attribute]
            else:
                # an empty return statement will: a) exit the function, b) with a return value of `None`
                return
        
        for child in element:
            node_tag = {}
            # 'tag' elements
            if child.tag == 'tag':
                #parse tag elements
                if PROBLEMCHARS.match(child.attrib['k']):
                    continue
                    
                # relevant section for 'addr:street' attributes:    
                elif LOWER_COLON.match(child.attrib['k']):
                    node_tag['type'] = child.attrib['k'].split(':',1)[0]
                    node_tag['key'] = child.attrib['k'].split(':',1)[1]
                    node_tag['id'] = element.attrib['id']
                else:
                    node_tag['type'] = 'regular'
                    node_tag['key'] = child.attrib['k']
                    node_tag['id'] = element.attrib['id']
                if child.attrib['k'] in ['addr:street', 'addr:postcode']:
                    if clean(child.attrib['v'], child, mapping):
                        node_tag['value'] = clean(child.attrib['v'], child, mapping)
                    else:
                        # if the cleaning function doesn't return anything, move to the next loop
                        continue
                else:
                    node_tag['value'] = child.attrib['v']
                tags.append(node_tag)
                
        return {'node': node_attribs, 'node_tags': tags}
        
    elif element.tag == 'way':
        for attribute in WAY_FIELDS:
            # if the key exists
            if element.attrib.get(attribute):
                way_attribs[attribute] = element.attrib[attribute]
            else:
                # an empty return statement will: a) exit the function, b) with a return value of `None`
                return
        
        position = 0
        for child in element:
            way_tag = {}
            way_node = {}
            
            if child.tag == 'tag':
                if PROBLEMCHARS.match(child.attrib['k']):
                    continue
                    
                elif LOWER_COLON.match(child.attrib['k']):
                    way_tag['type'] = child.attrib['k'].split(':',1)[0]
                    way_tag['key'] = child.attrib['k'].split(':',1)[1]
                    way_tag['id'] = element.attrib['id']
                
                else:
                    way_tag['type'] = 'regular'
                    way_tag['key'] = child.attrib['k']
                    way_tag['id'] = element.attrib['id']
                if child.attrib['k'] in ['addr:street', 'addr:postcode']:
                    if clean(child.attrib['v'], child, mapping):
                        way_tag['value'] = clean(child.attrib['v'], child, mapping)
                    else:
                        # if the cleaning function doesn't return anything, move to the next loop
                        continue
                else:
                    way_tag['value'] = child.attrib['v']
                tags.append(way_tag)
                    
            elif child.tag == 'nd':
                way_node['id'] = element.attrib['id']
                way_node['node_id'] = child.attrib['ref']
                way_node['position'] = position
                position += 1
                way_nodes.append(way_node)

    if element.tag == 'node':
        return {'node': node_attribs, 'node_tags': tags}
    elif element.tag == 'way':
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}


# In[7]:




# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


def validate_element(element, validator, schema=schema):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)
        
        raise Exception(message_string.format(field, error_string))


class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)
            

# ================================================== #
#               Main Function                        #
# ================================================== #

def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file,          codecs.open(NODE_TAGS_PATH, 'w') as node_tags_file,          codecs.open(WAYS_PATH, 'w') as ways_file,          codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file,          codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(node_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])

if __name__ == '__main__':
# Note: Validation is ~ 10X slower. For the project consider using a small
# sample of the map when validating.
    process_map(OSM_FILE, validate =False)



# In[ ]:




