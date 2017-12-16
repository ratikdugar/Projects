
# coding: utf-8

# In[2]:


import sqlite3
import csv
from pprint import pprint
import pandas as pd


# In[ ]:


#creates a database named openstreetmap
sqlite_file = 'openstreetmap.db'
db = sqlite3.connect(sqlite_file)
c = db.cursor()
#creates tables within the database having the same schema we had earlier
c.execute('''CREATE TABLE nodes(id INTEGER PRIMARY KEY NOT NULL, lat REAL, lon REAL, user TEXT, uid INTEGER, version INTEGER, changeset INTEGER, timestamp TEXT)''')
c.execute('''CREATE TABLE node_tags(id INTEGER, key TEXT, value TEXT, type TEXT, FOREIGN KEY (id) REFERENCES nodes (id))''')
c.execute('''CREATE TABLE ways(id INTEGER PRIMARY KEY NOT NULL, user TEXT, uid INTEGER, version TEXT, changeset INTEGER, timestamp TEXT)''')
c.execute('''CREATE TABLE way_tags(id INTEGER NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, type TEXT, FOREIGN KEY (id) REFERENCES ways(id))''')
c.execute('''CREATE TABLE way_nodes(id INTEGER NOT NULL, node_id INTEGER NOT NULL, position INTEGER NOT NULL, FOREIGN KEY (id) REFERENCES ways (id), FOREIGN KEY (node_id) REFERENCES nodes (id))''')
db.commit()


# In[5]:


#reads the different csv files and then writes them into the tables created above
df_nodes = pd.read_csv('nodes1.csv',encoding = "utf-8")
df_nodes.to_sql('nodes', db, if_exists='append', index=False)
df_nodes_tags = pd.read_csv('node_tags1.csv',encoding = "utf-8")
df_nodes_tags.to_sql('node_tags', db, if_exists='append', index=False)
df_ways = pd.read_csv('ways1.csv',encoding = "utf-8")
df_ways.to_sql('ways', db, if_exists='append', index=False)
df_ways_tags = pd.read_csv('way_tags1.csv',encoding = "utf-8")
df_ways_tags.to_sql('way_tags', db, if_exists='append', index=False)
df_ways_nodes = pd.read_csv('way_nodes1.csv',encoding = "utf-8")
df_ways_nodes.to_sql('way_nodes', db, if_exists='append', index=False)


# In[ ]:




