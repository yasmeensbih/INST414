import pandas as pd 
import re
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv("C:\Users\yasme\OneDrive\INST414\tweeter_threat.csv")


print(df.head())

print(df.info())


#drop rows with missing userID or tweet content 

df.dropna(subset= ['user_id', 'tweet_content'], inplace= True)


#remove duplicate tweets or records

df.drop_duplicates(subset='tweet_content', inplace=True)


#convert the timestamp column to datetime format 

df['timestamp'] = pd.to_datetime(df['timestamp'])


def clean_tweet(text):
    
    text = text.lower() 
    text = re.sub(r'http\ST', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


#Apply the cleaning function to the tweet content 

df['cleaned_tweet'] = df['tweet_content'].apply(clean_tweet)


G = nx.DiGraph()


for index, row in df.iterrows():
    user = row['user_id']
    mentioned_users = re.findall(r'@(\w+)', row['cleaned_tweet']) 
    
    for mentioned_user in mentioned_users:
        G.add_edge(user, mentioned_user)
        

print(nx.info(G))


degree_centrality = nx.degree_centrality(G)

betweenness_centrality = nx.betweenness_centrality(G)


important_nodes_degree = sorted(degree_centrality.items(), key = lambda x: x[1], reverse = True)


important_nodes_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse = True )


print(important_nodes_degree[:3])

print(important_nodes_betweenness[:3])



#visualize the network and drawing it

plt.figure(figsize = (12,12))
nx.draw(G, with_labels = True, node_size = 50, font_size = 10, node_color = 'blue')
plt.title("Twitter User Interaction Network")
plt.show()


#save the cleaned data to a new CSV file 

df.new_csv('cleaned_data.csv', index = False)
