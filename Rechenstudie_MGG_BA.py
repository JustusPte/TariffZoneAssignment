import networkx as nx
import operator
import random
import matplotlib.pyplot as plt
from gurobipy import *
from tqdm import tqdm
import data_management_BA as dm
import time
import pickle
import pandas as pd
import csv

#TODO def computeGS():

#original MGG algorithm
def MGG(G,commodities):
    """"
    original MGG algorithm
    param:
    G should be a graph instance
    commodities should be triples (P_i,w_i,c_i)i \in [1,k]

    returns: cut set S
    """

    cut_set = set()
    

    intersection_commodity_cut_set = dict()
    for i in commodities:
        intersection_commodity_cut_set[i] = 0

    max_edge = ()
    
    greedyScores = dict()

    for edge in G.edges:
        greedyScores[edge] = 0

    for i in commodities:
        for edge in i[0]:
            greedyScores[edge] += i[1]

    #determin edge with max GS and update everything
    max_edge = max((greedyScores.items()), key=operator.itemgetter(1))[0]



    while greedyScores[max_edge] > 0:

        #determine edge with max GS and update everything
        cut_set.add(max_edge)
        greedyScores.pop(max_edge)
        

        #update GS for all edges that lie on commodities, affected by max_edge
        for i in commodities:
            if max_edge in i[0]:
                intersection_commodity_cut_set[i] += 1 
                if intersection_commodity_cut_set[i] == i[2] + 1:
                    for edge in i[0]:
                        if greedyScores.get(edge):
                            greedyScores[edge] += (i[1]*(i[2]+1))
                
                elif intersection_commodity_cut_set[i] == i[2]:
                    for edge in i[0]:
                        if greedyScores.get(edge):
                            greedyScores[edge] -= (i[1]*(i[2]+2))
    
        max_edge = max((greedyScores.items()), key=operator.itemgetter(1))[0]
        
    return cut_set


#compute optimum for given cutset on TZw\D instance
def computeOptimum(G,commodities,s):
    """
    param:
    G should be a graph instance
    commodities should be triples (P_i,w_i,c_i)i \in [1,k]
    s cut set

    returns: optimum
    """

    #compute optimum for given cutset on TZw\D instance
    optimum = 0
    for i in commodities:
        cut_edges = len(set(i[0]).intersection(s))
        if cut_edges <= i[2]:
            optimum += (i[1] * (cut_edges + 1))
    return optimum


#updated MGG algorithm
def MGG_updated(G, commodities):
    """"
    param:
    G should be a graph instance
    commodities should be triples (P_i,w_i,c_i)i \in [1,k]

    returns:
    List of profits(j) ,here j is number of edges in cut set
    List of cut_sets of each iteration
    List of greedyScores of each iteration
    List of intersection_commodity_cut_set of each iteration (counts number of cuts on each commodities path)
    
    The last two are returned to run MGG_RC, where MGG_udated left off.
    """
    cut_set = set()
    cut_sets = [cut_set]

    
    intersection_commodity_cut_set = dict()
    intersection_commodity_cut_set_iter = [intersection_commodity_cut_set]
    
    for i in commodities:
        intersection_commodity_cut_set[i] = 0

    max_edge = ()
    profits = [computeOptimum(G,commodities, cut_set)]

    
    greedyScores = dict()
    greedyScores_iter = [greedyScores]

    for edge in G.edges:
        greedyScores[edge] = 0

    for i in commodities:
        for edge in i[0]:
            greedyScores[edge] += i[1]



    for j in range(0,len(G.edges)):

        #determine edge with max GS and update everything
        filtered_greedyScores = {key: value for key, value in greedyScores.items() if key not in cut_set}
        max_edge = max(filtered_greedyScores, key=filtered_greedyScores.get)
        cut_set.add(max_edge)
 
        #update GS for all edges that lie on commodities, affected by max_edge
        for i in commodities:
            if max_edge in i[0]:
                intersection_commodity_cut_set[i] += 1 
                if intersection_commodity_cut_set[i] == i[2]+1:
                    for edge in i[0]:
                        if greedyScores.get(edge):
                            greedyScores[edge] += (i[1]*(i[2]+1))
                
                elif intersection_commodity_cut_set[i] == i[2]:
                    for edge in i[0]:
                        if greedyScores.get(edge):
                            greedyScores[edge] -= (i[1]*(i[2]+2))
        
        #store iteration
        current_profit = computeOptimum(G,commodities,cut_set)
        profits.append(current_profit)
        greedyScores_iter.append(greedyScores)
        intersection_commodity_cut_set_iter.append(intersection_commodity_cut_set)
        cut_sets.append(cut_set.copy())
        assert len(greedyScores_iter) == len(profits) == len(cut_sets) == len(intersection_commodity_cut_set_iter), "Lengths do not match!"
    return profits, cut_sets, greedyScores_iter, intersection_commodity_cut_set_iter

#MGG_IP algorithm
def Greedy_IP_combined(G,commodities):
    """
    param:
    G should be a graph instance
    commodities should be triples (P_i,w_i,c_i)i \in [1,k]
    returns: IP model, cut set
    """
    
    profits, cut_sets, _, _ = MGG_updated(G,commodities)
    max_index  = profits.index(max(profits))
    cut_set = cut_sets[max_index]
    filtered_commodities = [
    com for com in commodities
    if len(set(com[0]).intersection(cut_set)) <= com[2]
]
    

    

    model = Model("FZ_w\D")
    model.Params.LogToConsole = 0
    #disable output of optimization information on console
    #model.Params.LogToConsole = 0

    anz_commodities = len(filtered_commodities)
    #variables
    y = {e: model.addVar(vtype= GRB.INTEGER,name=f'y_{e}') for e in G.edges}
    

    u = {i: model.addVar(vtype= GRB.INTEGER,name=f'u_{i}') for i in range(anz_commodities)}
    d = {i: model.addVar(vtype= GRB.INTEGER,name=f'd_{i}') for i in range(anz_commodities)}

    #constraints

    for e in G.edges:
        model.addConstr(y[e] == [0,1])
    
    for i in range(anz_commodities):
        model.addConstr(u[i] == filtered_commodities[i][2]+1)
        model.addConstr(d[i] == filtered_commodities[i][1])

        model.addConstr(quicksum(y[e] for e in filtered_commodities[i][0]) <= (u[i])-1)
        
    

    #objective
    model.setObjective(quicksum(((quicksum(y[e] for e in filtered_commodities[i][0])+1) * filtered_commodities[i][1])  for i in range(anz_commodities)),GRB.MAXIMIZE)

    model.update()

    model.optimize()
    

    cut_set = set()
    for e in G.edges:
        if y[e].X == 1:
            cut_set.add(e)
    return model, cut_set

#MGG_Remove_Commodity algorithm
def MGG_Remove_Commodity(G, commodities, cut_set = set(), greedyScores = dict(), intersection_commodity_cut_set = dict()):
    """"
    param:
    G should be a graph instance
    commodities should be triples (P_i,w_i,c_i)i \in [1,k]
    List of greedyScores of each iteration
    List of intersection_commodity_cut_set of each iteration (counts number of cuts on each commodities path)
    
    The last two are returned to run MGG_RC, where MGG_udated left off.

    returns:
    List of profits(j) ,here j is number of edges in cut set
    List of cut sets

    """
    cut_set = cut_set
    cut_sets = []
    profits = [computeOptimum(G,commodities,cut_set)]
    commodities = commodities.copy()
    intersection_commodity_cut_set = intersection_commodity_cut_set
    size = len(G.edges)
    full_commodtities = []
 

    if not(intersection_commodity_cut_set):
        for i in commodities:
            intersection_commodity_cut_set[i] = 0

    for i in commodities:
        if intersection_commodity_cut_set[i] >= i[2]:
            full_commodtities.append(i) 
    #iteration_configuration = []

    if not greedyScores:
        
        for edge in G.edges:
            greedyScores[edge] = 0

        for i in commodities:
            for edge in i[0]:
                greedyScores[edge] += i[1]

    

    #iterate while there are commodities left for removal as well as uncut edges left
    while commodities and len(cut_set) < size:
       
        

        filtered_greedyScores = {key: value for key, value in greedyScores.items() if key not in cut_set}
        
        

        max_edge = max(filtered_greedyScores, key=filtered_greedyScores.get)
        cut_sets.append(cut_set)
        profits.append(computeOptimum(G,commodities,cut_set))
        #determine edge with max GS and update everything
        
        if greedyScores[max_edge] > 0:
            cut_set.add(max_edge)
            

            #update GS for all edges that lie on commodities, affected by max_edge
            for i in commodities:
                if max_edge in i[0]:
                    intersection_commodity_cut_set[i] += 1 
                    if intersection_commodity_cut_set[i] == i[2] + 1:
                        for edge in i[0]:
                            if greedyScores.get(edge):
                                greedyScores[edge] += (i[1]*(i[2]+1))
                    
                    elif intersection_commodity_cut_set[i] == i[2]:
                        full_commodtities.append(i)
                        for edge in i[0]:
                            if greedyScores.get(edge):
                                greedyScores[edge] -= (i[1]*(i[2]+2))
        
        elif not full_commodtities:
            break
        else:
            

            #find commodity with minimal profit of all full commoditites
            min_commodity = min(full_commodtities, key= lambda x: (x[1]*(x[2]+1)))

            #delete that minimal commodity
            full_commodtities.remove(min_commodity)
            commodities.remove(min_commodity)

            #remove all edges of that commodity from the cut_set
            cut_set = cut_set.difference(min_commodity[0])

            #update GreedyScores
            
            #fallunterscheidung min_commodity am limit oder über limit gewesen 
            
            if intersection_commodity_cut_set[min_commodity] == min_commodity[2]:
                for edge in min_commodity[0]:
                    greedyScores[edge] += (min_commodity[1]*(min_commodity[2]+1)) 


            for i in commodities:
                #cut aus i und min_commodity bestimmen und greedy Score von jeder Kante auf i anpassen, falls i nun am limit oder unterm limit ist
                status_commodity_before = intersection_commodity_cut_set[i]
                intersection_commodity_cut_set[i] -= len(set(i[0]).intersection(set(min_commodity[0])))
                if intersection_commodity_cut_set[i] == i[2] and status_commodity_before > i[2]:
                    for edge in i[0]:
                        greedyScores[edge] -= (i[1]*(i[2]+1))
                elif intersection_commodity_cut_set[i] < i[2] and status_commodity_before > i[2]:
                    full_commodtities.remove(i)
                    for edge in i[0]:
                        greedyScores[edge] += i[1]
                elif intersection_commodity_cut_set[i] < i[2] and status_commodity_before == i[2]:
                    full_commodtities.remove(i)
                    for edge in i[0]:
                        greedyScores[edge] += (i[1]*(i[2]+2))

        
        
    return profits, cut_sets

#LP set up and solved for TZw/D instance
def solve_LP(G, commodities, M):
    """
    param:
    G Graph instance
    commodities should be triples (P_i,w_i,c_i)i \in [1,k]
    M offset

    returns: model, cut_set
    """


    model = Model("FZ_w\D")
    model.Params.LogToConsole = 0
    #disable output of optimization information on console
    #model.Params.LogToConsole = 0
    anz_commodities = len(commodities)
    #variables

    y = {e: model.addVar(name=f'y_{e}') for e in G.edges}
    x = {i: model.addVar(name=f'x_{i}') for i in range(anz_commodities)}
    sp = {i: model.addVar(name=f'sp_{i}') for i in range(anz_commodities)} #s^+
    sm = {i: model.addVar(name=f'sm_{i}') for i in range(anz_commodities)} #s^-
    z = {i: model.addVar(name=f'z_{i}') for i in range(anz_commodities)}

    u = {i: model.addVar(name=f'u_{i}') for i in range(anz_commodities)}
    d = {i: model.addVar(name=f'd_{i}') for i in range(anz_commodities)}

    #constraints

    for e in G.edges:
        model.addConstr(y[e] == [0,1])
    
    for i in range(anz_commodities):
        model.addConstr(u[i] == commodities[i][2]+1)
        model.addConstr(d[i] == commodities[i][1])

        model.addConstr(z[i] == [0,1])
        model.addConstr(sp[i] >= 0)
        model.addConstr(sm[i] >= 0)
        model.addConstr(x[i]-(M*z[i]) <= u[i]-1)
        model.addConstr(x[i] == quicksum(y[e] for e in commodities[i][0]))
        model.addConstr(sp[i]-sm[i]+x[i]+1 == u[i])
        
    

    #objective
    model.setObjective(quicksum(commodities[i][1]*((x[i]+1)-((commodities[i][2]+1)*z[i])-sm[i]) for i in range(anz_commodities)),GRB.MAXIMIZE)

    model.update()

    model.optimize()
    
    cut_set = []
    for e in G.edges:
        if y[e].X == 1:
            cut_set.append(e)
    return model, cut_set
    
    
#IP is set up and solved for TZw/D instance
def solve_IP(G, commodities, M):
    """
    param:
    G Graph instance
    commodities should be triples (P_i,w_i,c_i)i \in [1,k]
    M offset

    returns: model, cut_set
    """

    

    model = Model("FZ_w\D")

    #disable output of optimization information on console
    model.Params.LogToConsole = 0
    anz_commodities = len(commodities)
    #variables

    y = {e: model.addVar(vtype=GRB.INTEGER, name=f'y_{e}') for e in G.edges}
    x = {i: model.addVar(vtype=GRB.INTEGER, name=f'x_{i}') for i in range(anz_commodities)}
    sp = {i: model.addVar(vtype=GRB.INTEGER, name=f'sp_{i}') for i in range(anz_commodities)} #s^+
    sm = {i: model.addVar(vtype=GRB.INTEGER, name=f'sm_{i}') for i in range(anz_commodities)} #s^-
    z = {i: model.addVar(vtype=GRB.INTEGER, name=f'z_{i}') for i in range(anz_commodities)}

    u = {i: model.addVar(vtype=GRB.INTEGER, name=f'u_{i}') for i in range(anz_commodities)}
    d = {i: model.addVar(vtype=GRB.INTEGER, name=f'd_{i}') for i in range(anz_commodities)}

    #constraints

    for e in G.edges:
        model.addConstr(y[e] == [0,1])
    
    for i in range(anz_commodities):
        model.addConstr(u[i] == commodities[i][2]+1)
        model.addConstr(d[i] == commodities[i][1])

        model.addConstr(z[i] == [0,1])
        model.addConstr(sp[i] >= 0)
        model.addConstr(sm[i] >= 0)
        model.addConstr(x[i]-(M*z[i]) <= u[i]-1)
        model.addConstr(x[i] == quicksum(y[e] for e in commodities[i][0]))
        model.addConstr(sp[i]-sm[i]+x[i]+1 == u[i])
        
    

    #objective
    model.setObjective(quicksum(commodities[i][1]*((x[i]+1)-((commodities[i][2]+1)*z[i])-sm[i]) for i in range(anz_commodities)),GRB.MAXIMIZE)

    model.update()

    model.optimize()
    
    cut_set = []
    for e in G.edges:
        #decision_variables.append(f"{y[e].VarName} = {y[e].X}")
        if y[e].X == 1:
            cut_set.append(e)
    return model, cut_set
    
#generates the commodities for the TZw/D instance (old version where commodities were not uniformly distributed on graph)
def commodityGenerator(G, anz_commodities_choice, upperbound_capacity_choice):
    '''
    param:
    G should be graph instance
    anz_commodities_choice: list of possible choices for number of commodities
    upperbound_capacity_choice: list of possible choices for upperbound for capacities

    returns: list of commodities
    '''
    size = len(list(G.nodes))

    anz_commodities = round(random.choice(anz_commodities_choice) * size)
   

    commodities = []
    upperbound_demand = random.choice([2,3,4,5,8,10,50,100,500,1000])
    for i in range(anz_commodities):
        start = random.choice(list(G.nodes))
        #end = list(G.nodes)[-1]
        end = random.choice(list(G.nodes))
        while end == start:
            end = random.choice(list(G.nodes))
        if start > end:
            path = tuple(nx.shortest_path(G,end,start))
        else:
            path = tuple(nx.shortest_path(G,start,end))
        path_edges = tuple(list(zip(path, path[1:])))

        demand = random.randint(1,upperbound_demand)
       
        
        upper_end_capacity = round(random.choice(upperbound_capacity_choice) * len(path_edges))
        if upper_end_capacity < 1:
            upper_end_capacity= 1
        capacity = random.randint(1,upper_end_capacity) #zufällig zwichen 0 und min(upperbound, Pfadlänge)
        commodities.append((path_edges,demand,capacity))
    
    return commodities

#runs MGG_u and tracks time
def MGG_updated_with_timer(G, commodities):
    """
    param:
    G should be a graph instance
    commodities should be triples (P_i,w_i,c_i)i \in [1,k]

    returns: 
    optimal revenue
    optimal cut_set
    time taken
    Greedy Scores of iteration with optimal revenue
    Intersection_commodity_cut_set of iteration with optimal revenue
    """
    start_time = time.perf_counter()  # Start timing MGG
    profits,cutsets,Greedy_scores_iter,Intersection_commodity_cut_set_iter = MGG_updated(G, commodities)  # Call the MGG_updated function
    revenue = max(profits)
    max_index = profits.index(revenue)
    cut_set_opt = cutsets[max_index]

    end_time = time.perf_counter()  # Stop timing MGG
    Greedy_scores_iter_opt = Greedy_scores_iter[max_index]
    Intersection_commodity_cut_set_iter_opt = Intersection_commodity_cut_set_iter[max_index]
    mgg_time = end_time - start_time  # Calculate time taken
    return revenue,cut_set_opt, mgg_time,Greedy_scores_iter_opt,Intersection_commodity_cut_set_iter_opt  # Return both the result and the time taken

#runs MGG_RC and tracks time
def MGG_Remove_Commodity_with_timer(g,commodities,cut_set,Greedy_scores_iter_opt,Intersection_commodity_cut_set_iter_opt,mgg_time):
    """
    param:
    G should be a graph instance
    commodities should be triples (P_i,w_i,c_i)i \in [1,k]
    optimal cut_set (given by MGG_updated)
    Greedy Scores of iteration with optimal revenue (given by MGG_updated)
    Intersection_commodity_cut_set of iteration with optimal revenue (given by MGG_updated)
    time taken by MGG_updated

    returns:
    optimal revenue
    time taken
    """
    start_time = time.perf_counter()  # Start timing MGG
    MGG_Remove_Commodity_revenues,_ = MGG_Remove_Commodity(g,commodities,cut_set,Greedy_scores_iter_opt,Intersection_commodity_cut_set_iter_opt)  # Call the MGG_Remove_Commodity function
    
    

    end_time = time.perf_counter()  # Stop timing MGG
    mgg3_time = end_time - start_time +mgg_time  # Calculate time taken
    return MGG_Remove_Commodity_revenues, mgg3_time  # Return both the result and the time taken

#runs MGG_IP and tracks time
def Greedy_IP_combined_with_timer(G, commodities):
    """
    param:
    G should be a graph instance
    commodities should be triples (P_i,w_i,c_i)i \in [1,k]

    returns:
    IP model
    cut set
    time taken
    """

    start_time = time.perf_counter()  # Start timing MGG
    model, s = Greedy_IP_combined(G, commodities)  # Call the original MGG function
    end_time = time.perf_counter()  # Stop timing MGG
    
    mgg_time = end_time - start_time  # Calculate time taken
    return model, s, mgg_time  # Return both the result and the time taken

#runs IP and tracks time
def solve_IP_with_timer(G, commodities, M):
    """
    param:
    G should be a graph instance
    commodities should be triples (P_i,w_i,c_i)i \in [1,k]
    M offset

    returns:
    IP model
    cut set
    time taken
    """

    start_time = time.perf_counter()  # Start timing IP solver
    model, cut_set = solve_IP(G, commodities, M)  # Call the original solve_IP function
    end_time = time.perf_counter()  # Stop timing IP solver
    
    ip_time = end_time - start_time  # Calculate time taken
    return model, cut_set, ip_time  # Return both the result and the time taken

#commodites are generated randomly
def commodityGenerator_advanced(G,anz_commodities_choice,upperbound_capacity_choice):
    '''
    param:
    G should be graph instance
    anz_commodities_choice: list of possible choices for number of commodities
    upperbound_capacity_choice: list of possible choices for upperbound for capacities

    returns: list of commodities
    '''
    size = len(list(G.nodes))

    anz_commodities = round(random.choice(anz_commodities_choice) * size)
    #upperbound_capacity = round(random.choice(upperbound_capacity) * size)
    #if upperbound_capacity == 0:
     #   upperbound_capacity += 1

    commodities = []
    upperbound_demand = random.choice([2,3,4,5,8,10,50,100,500,1000])
    for i in range(anz_commodities):
        
        coinflip = random.choice([True,False])
        if coinflip:
            path_length = random.randint(1,int(1/2 * size))
        else:
            path_length = random.randint(int(1/2*size),size-1)

        start = random.choice(list(G.nodes)[:(size-path_length)])
        
        path = tuple(range(start,start+path_length))
        path_edges = tuple(list(zip(path, path[1:])))

        demand = random.randint(1,upperbound_demand)
       
        
        upper_end_capacity = round(random.choice(upperbound_capacity_choice) * len(path_edges))
        if upper_end_capacity < 1:
            upper_end_capacity= 1
        capacity = random.randint(1,upper_end_capacity) #zufällig zwichen 0 und min(upperbound, Pfadlänge)
        commodities.append((path_edges,demand,capacity))
    
    return commodities


#runs alls MGG variants and the MIP on number instances of fixed size solver and tracks trime 
def fixedSize_MGG_IP_comparison(anz_nodes,num_iterations):
    '''
    param:
    anz_nodes: Amount of nodes in the graphs, that will be generated
    num_iterations: number of graphs generated
    '''
    IP_values = []
    MGG_values = []
    xvalues = []
    mgg_1_time_total = 0
    ip_time_total = 0
    Greedy_IP_combined_time_total = 0
    mgg_3_time_total = 0
    max_IP = 1
    max_MGG_1 = 1
    max_MGG_3 = 1
    max_MGG_IP = 1

    for i in range(num_iterations):
        g = nx.path_graph(anz_nodes)
        commodities = commodityGenerator_advanced(g,[1],[1])
        model, cut_set_ip, ip_time = solve_IP_with_timer(g,commodities, len(g.nodes) -1)
        ip_time_total += ip_time
        IP_value = model.ObjVal
        IP_values.append(IP_value)

        MGG_updated_revenue, cut_set, mgg_time,Greedy_scores_iter_opt,Intersection_commodity_cut_set_iter_opt = MGG_updated_with_timer(g,commodities)
        mgg_1_time_total += mgg_time
        MGG_value = computeOptimum(g,commodities,cut_set)
        #print(MGG_value)
        MGG_values.append(MGG_value)
        xvalues.append(IP_value/MGG_value)

        model_greedy_ip, cut_set_greedy_Ip, Greedy_IP_combined_time = Greedy_IP_combined_with_timer(g,commodities)
        Greedy_IP_combined_time_total += Greedy_IP_combined_time
        Greedy_IP_combined_revenue = model_greedy_ip.ObjVal

        MGG_Remove_Commodity_revenues,mgg3_time = MGG_Remove_Commodity_with_timer(g,commodities,cut_set,Greedy_scores_iter_opt,Intersection_commodity_cut_set_iter_opt,mgg_time)
        MGG_Remove_Commodity_revenue = max(MGG_Remove_Commodity_revenues)
        mgg_3_time_total += mgg3_time
        if IP_value / MGG_updated_revenue > max_IP / max_MGG_1:
                max_MGG_1 = MGG_updated_revenue
                max_IP = IP_value
                max_MGG_3 = MGG_Remove_Commodity_revenue
                max_MGG_IP = Greedy_IP_combined_revenue

   
    dm.store_results_IP_Greedy_variants(anz_nodes,commodities,max_IP,max_MGG_1,max_MGG_3,max_MGG_IP)
        
    return ip_time_total,mgg_1_time_total,Greedy_IP_combined_time_total,mgg_3_time_total

#runs a whole testset of instances and stores results
def run_instance_set(starting_instance_size = 2, ending_instance_size = 10, num_instances = 10,runtimes_file_name = "runtime.csv", pickle_file_name = "results_instance_size_"):
    """
    param:
    starting_instance_size: smallest instance size (int)
    ending_instance_size: largest instance size (int)
    num_instances: number of instances for each instance size (int)
    runtimes_file_name: name of the file to store the runtimes (str)
    pickle_file_name: name of the file to store the results
    """
    
    with open(runtimes_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Instance Size', 'IP Time', 'MGG_1 Time', 'Greedy IP Combined Time', 'MGG_3 Time'])
    for i in tqdm(range(starting_instance_size,ending_instance_size+1)):
        
        ip_time_total,mgg_1_time_total,Greedy_IP_combined_time_total,mgg_3_time_total = fixedSize_MGG_IP_comparison(i,num_instances)
        
        filename = pickle_file_name + str(i) +  ".pkl"
        dm.save_results_to_pickle(filename)
        with open(runtimes_file_name, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([i, ip_time_total, mgg_1_time_total, Greedy_IP_combined_time_total, mgg_3_time_total]) 


