import csv
import itertools
import copy
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from networkx.readwrite import edgelist
from operator import itemgetter
from networkx.algorithms import community
from pylab import *
import scipy as sp


from google.colab import drive
drive.mount('/content/drive')

#MAKE THE GRAPH
G=nx.read_adjlist(r'/content/drive/MyDrive/3.csv',delimiter=',')

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
                       node_color = values, node_size = 500)
#NODES OF GRAHP					   
len(list(G.nodes))

nx.draw(G,with_labels=True)
show()

#DEGREES OF NODES
degrees = [G.degree(n) for n in G.nodes()]
plt.hist(degrees)

#LOG LOG PLOT
degree_sequence = degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
plt.loglog(degree_sequence,marker='*')
plt.show()

#DENSITY
nx.classes.function.density(G)

#RECIPROCITY
nx.reciprocity(G)

#TRANSITIVITY
nx.transitivity(G)

#NUMBER OF TRIANGLES IN EACH NODE
sum(nx.triangles(G).values())

#AVERAGE CLUSTERING COEFFICIENT OF THE GRAPH
nx.average_clustering(G) #Compute the average clustering coefficient for the graph G.
#Global clustering coefficient


nx.algorithms.components.number_connected_components(G)

nx.clustering(G) #Local CC for nodes

#Statistics of the GRAPH

def net_prop_dict(G):
    prop_dict = {}

    prop_dict['no_of_nodes'] = nx.number_of_nodes(G)
    prop_dict['no_of_edges'] = nx.number_of_edges(G)
    if nx.is_connected(G):
        prop_dict['average_shortest_path_length'] = nx.average_shortest_path_length(G)
        prop_dict['diameter'] = nx.diameter(G)
    prop_dict['transitivity'] = nx.transitivity(G)
    prop_dict['average_clustering'] = nx.average_clustering(G)  #Global CC (or) CC for graph 
    prop_dict['edge_density'] = nx.classes.function.density(G)
    prop_dict['average_degree'] = np.array([d for n, d in G.degree()]).sum()/nx.number_of_nodes(G)
    prop_dict['total_triangles'] = np.array(list(nx.triangles(G).values())).sum()
    prop_dict['number_connected_components'] = nx.algorithms.components.number_connected_components(G)
    return prop_dict

prop_dict_G = net_prop_dict(G)

prop_dict_G

#Size of Giant Connected Component
giantC = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])
print("Size of Giant Connected Component (in nodes): ",len(giantC.nodes))



#SHOW THE STATISTICS OF GCC
net_prop_dict(giantC)

#MAKE THE GRAPH ONLY WITH THE NODES THAT HAVE 2 EDGES
k_core = nx.k_core(G, k=2)

nx.draw(k_core)

net_prop_dict(k_core)

#AVERAGE DEGREE OF NEIGHBORS NODES
nx.average_neighbor_degree(G)

vk = dict(G.degree())
vk = list(vk.values())
print('Degree', vk)


#FIND AND PLOT DEGREE DISTRIBUTION P(K)
def degree_distribution(G):
    vk = dict(G.degree())
    vk = list(vk.values()) # we get only the degree values
    maxk = np.max(vk)
    mink = np.min(min)
    kvalues= arange(0,maxk+1) # possible values of k
    Pk = np.zeros(maxk+1) # P(k)
    for k in vk:
        Pk[k] = Pk[k] + 1
    Pk = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one
    return kvalues,Pk

ks, Pk = degree_distribution(G)

plt.figure()
plt.loglog(ks,Pk,'bo',basex=10,basey=10)
plt.xlabel("k", fontsize=20)
plt.ylabel("P(k)", fontsize=20)
plt.title("Degree distribution", fontsize=20)
plt.grid(True)
plt.savefig('degree_dist.eps') #save the figure into a file
plt.show(True)





def momment_of_degree_distribution(G,m):
    k,Pk = degree_distribution(G)
    M = sum((k**m)*Pk)
    return M
    
    
    
def momment_of_degree_distribution2(G,m):
    M = 0
    for i in G.nodes:
        M = M + G.degree(i)**m
    M = M/N
    return M
    
#First moment of the degree distribution   
k1 = momment_of_degree_distribution(G,1)
print("Mean degree = ", mean(vk))
print("First moment of the degree distribution = ", k1)


k2 = momment_of_degree_distribution(G,2)
print("Second moment of the degree distribution = ", k2)

variance = momment_of_degree_distribution(G,2) - momment_of_degree_distribution(G,1)**2
print("Variance of the degree = ", variance)

#Shannon Entropy

def shannon_entropy(G):
    k,Pk = degree_distribution(G)
    H = 0
    for p in Pk:
        if(p > 0):
            H = H - p*math.log(p, 2)
    return H

H = shannon_entropy(G)
print("Shannon Entropy = ", "%3.4f"%H)

#Clustering of all nodes

vcc = []
for i in G.nodes():
    vcc.append(nx.clustering(G, i))
vcc= np.array(vcc)
print('Clustering of all nodes:', vcc)

#PLOT OF CLUSTERING COEFFICIENT

plt.figure()
plt.hist(vcc, bins  = 10, normed=True)
plt.title("Distribution of the clustering coefficient")
plt.ylabel("P(cc)")
plt.xlabel("Clustering coefficient (cc)")
plt.grid(True)
plt.savefig('clustering.eps') #save the figure into a file
plt.show()


#Average Degree of Giant Connected Component
degrees = list()
for node in giantC.nodes:
    degrees.append(giantC.degree[node])
formatted_float = "{:.2f}".format(statistics.mean(degrees))
print("Average Degree of Giant Connected Component = ",formatted_float)

#degreeDistribution in gcc
degree_sequence=sorted([degree for node, degree in giantC.degree()], reverse=False)
degreeCount = collections.Counter(degree_sequence)
degree_list=list()
for x in degreeCount:
    degree_list.append(x)
    row = [x,degreeCount[x]]
    f = open('degreeDistribution.txt','a',newline='')
    writer = csv.writer(f)
    writer.writerow(row)
f.close()

assortativity = giantC.degree
for deg in degree_list:
    nodes_degree=list()
    total_sum = 0;
    total_count = 0;
    for n in giantC.nodes:
        if assortativity[n] == deg:
            nodes_degree.append(n)
    for i in nodes_degree:
        sum1=0;
        count=0;
        for k in giantC.neighbors(i):
            sum1+=giantC.degree(k)
            count+=1
        total_sum+=(sum1/count)
        total_count+=1
    row = [deg,total_sum/total_count]
    f = open('assortativity.txt','a',newline='')
    writer = csv.writer(f)
    writer.writerow(row)
f.close()

#Average clustering
degrees = giantC.degree
for d in degree_list:
    nodes_degree=list()
    for no in giantC.nodes:
        if degrees[no] == d:
            nodes_degree.append(no)
    nodes_degree_clustering = nx.clustering(giantC,nodes_degree)
    clustering_degree=list()
    sum=0
    count=0
    for x in nodes_degree:
        sum+=(nodes_degree_clustering[x])
        count+=1
    row = [d,str(sum/count)]
    f = open('clustering.txt','a',newline='')
    writer = csv.writer(f)
    writer.writerow(row)
f.close()


#NETWORK EFFICIENCY
E = nx.global_efficiency(G)
print('Network efficiency', E)


#BETWEENNESS CENTRALITY OF ALL GRAPH AND GCC
b=nx.betweenness_centrality(G)
print(b)

b=nx.betweenness_centrality(G0)
print(b)

#AVERAGE CLUSTERING AMONG GCC
print(nx.average_clustering(G0))

#AVERAGE DEGREE OF GCC NEIGHBORS
nx.average_neighbor_degree(G0)

