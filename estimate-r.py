import csv
from csv import reader
import networkx as nx

correlationList=list()
correlationList={0.5,0.6,0.7}

for correlation in correlationList:
    with open("data.txt") as data:
        for line in data:
            rc = line.split()
            value = (float(rc[2]))
            if value > correlation:
                row = [rc[0],rc[1],value]
                f = open('dataCorellation'+str(correlation)+'.txt', 'a',newline='')
                writer = csv.writer(f)
                writer.writerow(row)

    graph = nx.Graph()
    with open('dataCorellation'+str(correlation)+'.txt', 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            graph.add_edge(row[0],row[1])

    size_of_graph=graph.number_of_nodes()
    Gcc = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])
    print(str(correlation)+" correlation => Size of Giant Connected Component ",(len(Gcc)/size_of_graph))


correlationList=list()
correlationList={0.701,0.702,0.703,0.704,0.705,0.706,0.707,0.708,0.709} 

for correlation in correlationList:
    with open("dataCorellation0.7.txt") as data:
        for line in data:
            rc = line.split(',')
            value = (float(rc[2]))
            if value > correlation:
                row = [rc[0],rc[1],value]
                f = open('dataCorellation'+str(correlation)+'.txt', 'a',newline='')
                writer = csv.writer(f)
                writer.writerow(row)

    graph = nx.Graph()
    with open('dataCorellation'+str(correlation)+'.txt', 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            graph.add_edge(row[0],row[1])

    size_of_graph=graph.number_of_nodes()
    Gcc = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])
    print(str(correlation)+" correlation => Size of Giant Connected Component ",(len(Gcc)/size_of_graph))




