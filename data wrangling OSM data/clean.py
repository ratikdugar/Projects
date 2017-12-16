
# coding: utf-8

# In[8]:


"""
Updates tag attribute values to proper formats and then returns those values when called from the shape_element function.
"""

import re

#combines all the update functions        
def clean(value, tag, mapping):
    if is_street_name(tag):
        value = update_name(value, mapping)
    elif is_zipcode(tag):
        value = update_zipcode(value)
    return value

#returns the updated zipcodes
def update_zipcode(postcode):
    try:
        postcode = re.compile(r'94\d\d\d').search(postcode).group()
    except AttributeError:
        postcode = 'None'
    return postcode

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

# Checks if it is a zipcode and returns the element where the value of the k attribute is "addr:postcode"  or "postal_code"                                      
def is_zipcode(elem):
    return (elem.attrib['k'] == "addr:postcode" or elem.attrib['k'] == "postal_code")

# Checks if it is a street name and returns the element where the value of the k attribute is "addr:street"             
def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


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

