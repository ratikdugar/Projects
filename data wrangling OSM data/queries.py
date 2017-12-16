
# coding: utf-8

# In[1]:


import sqlite3
import pprint
sqlite_file = 'openstreetmap.db'
db = sqlite3.connect(sqlite_file)
c = db.cursor()


# In[8]:


#Query finds the count of each different postal code in the db and arranges them by count
c.execute("SELECT tags.value, COUNT(*) as numcount FROM (SELECT * FROM node_tags UNION ALL     SELECT * FROM way_tags) tags WHERE tags.key='postcode' GROUP BY tags.value ORDER BY numcount DESC;")
pprint.pprint(c.fetchall())


# In[14]:


#Query finds all the cities in the db and arranges them by count
c.execute("SELECT tags.value, COUNT(*) as numcount FROM (SELECT * FROM node_tags UNION ALL       SELECT * FROM way_tags) tags WHERE tags.key= 'city' GROUP BY tags.value ORDER BY numcount DESC;")
pprint.pprint(c.fetchall())


# In[15]:


#Finds the number of nodes in the db
c.execute("SELECT COUNT(*) FROM nodes;")
pprint.pprint(c.fetchall())


# In[16]:


#Finds the number of ways in the db
c.execute("SELECT COUNT(*) FROM ways;")
pprint.pprint(c.fetchall())


# In[17]:


#Finds the number of node tags
c.execute("SELECT COUNT(*) FROM node_tags;")
pprint.pprint(c.fetchall())


# In[18]:


#Finds the number of way tags
c.execute("SELECT COUNT(*) FROM way_tags;")
pprint.pprint(c.fetchall())


# In[19]:


#Finds the number of way nodes
c.execute("SELECT COUNT(*) FROM way_nodes;")
pprint.pprint(c.fetchall())


# In[23]:


#Finds the number of unique users 
c.execute("SELECT COUNT(DISTINCT(allusers.uid)) FROM (SELECT uid FROM nodes UNION ALL SELECT uid FROM ways) as allusers ;")
pprint.pprint(c.fetchall())


# In[4]:


#Finds the 10 most prolific users
c.execute("SELECT allusers.user, COUNT(*) as num FROM (SELECT user,uid FROM nodes UNION ALL SELECT user,uid FROM ways) as allusers GROUP BY allusers.uid ORDER BY num DESC LIMIT 10;")      
pprint.pprint(c.fetchall())


# In[17]:


#Finds the different amenities and arranges them by count
c.execute("SELECT value,count(*) as numcount FROM node_tags WHERE key='amenity' GROUP BY value ORDER BY numcount DESC ;")
pprint.pprint(c.fetchall())


# In[9]:


#Finds the location of all the charging stations 
c.execute("SELECT node_tags.value,node_tags.id,nodes.lat,nodes.lon FROM nodes JOIN node_tags ON nodes.id=node_tags.id WHERE node_tags.value='charging_station';") 
pprint.pprint(c.fetchall())


# In[16]:


#Finds the 20 most popular restaurant cuisines
c.execute("SELECT node_tags.value, COUNT(*) as num            FROM node_tags                JOIN (SELECT DISTINCT(id) FROM node_tags WHERE value = 'restaurant') AS distinctones                 ON node_tags.id = distinctones.id            WHERE node_tags.key = 'cuisine'           GROUP BY node_tags.value           ORDER BY num DESC LIMIT 20;")
pprint.pprint(c.fetchall())


# In[26]:


#Finds the location of all the places you can possibly stay at
c.execute("SELECT node_tags.value,nodes.lat,nodes.lon FROM nodes JOIN node_tags ON nodes.id=node_tags.id WHERE node_tags.key='tourism' and node_tags.value='camp_site' or node_tags.value='hotel' or node_tags.value='hostel' or node_tags.value='motel' or node_tags.value='inn' ORDER BY node_tags.value;") 
pprint.pprint(c.fetchall())


# In[ ]:




