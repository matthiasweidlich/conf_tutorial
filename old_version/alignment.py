import numpy
import variables as v
import json
import heapq
import tqdm
import graphviz as gr
import xml.etree.ElementTree as et

class Alignment:
    def __init__(self):
        self.alignment_move = []
        self.move_in_log = []
        self.move_in_model = []
        self.fitness = []

class Astar (Alignment):

    def __init__(self):
        Alignment.__init__(self)
        self.solutions = []

    # Creating Incidence matrix
    def Incidence_Matrix(self, net):
        # Creating an empty matrix
        incidnet_matrix = numpy.zeros((len(net.places.keys()), len(net.transitions.keys())), dtype=int)

        for keyT in net.transitions.keys():
            for keyP in net.places.keys():
                if (net.transitions[keyT][0], net.places[keyP]) in net.edges:
                    col_index = list(net.transitions.values()).index(net.transitions[keyT])
                    row_index = list(net.places.values()).index(net.places[keyP])
                    incidnet_matrix[row_index][col_index] = 1
                elif (net.places[keyP], net.transitions[keyT][0]) in net.edges:
                    col_index = list(net.transitions.values()).index(net.transitions[keyT])
                    row_index = list(net.places.values()).index(net.places[keyP])
                    incidnet_matrix[row_index][col_index] = -1

        return incidnet_matrix

    # A-star algorithm
    def Astar_Exe(self, net, total_trace, index_place_start, index_place_end, no_of_solutions=1):
        # Reading the model
        #total_trace, net, index_place_end, index_place_start, Incident_matrix_net = self.__Reading_IO(model_path, log_path)
        Incident_matrix_net = self.Incidence_Matrix(net)

        #All the solutions, i.e. alignments
        v.solutions = dict()
        for key in tqdm.tqdm(total_trace):
            trace = total_trace[key]

            # Creating Synchronous products
            elements_tot, places, index_place_start_log, index_place_end_log, Incident_matrix = self.__Synchronous_Product(net, trace)

            # Creating initial marking vector
            init_mark_vector = list(numpy.repeat(0, len(places)))
            init_mark_vector[index_place_start] = 1
            init_mark_vector[index_place_start_log] = 1

            # Creating Final marking vector
            final_mark_vector = list(numpy.repeat(0, len(places)))
            final_mark_vector[index_place_end] = 1
            final_mark_vector[index_place_end_log] = 1
            final_marking = list(net.places.values())[index_place_end]

            # Storing the specification of SP net
            v.Incident_matrix = Incident_matrix
            v.init_mark_vector = init_mark_vector
            v.final_mark_vector = final_mark_vector

            # ---Initializing variables
            v.closed_list = closed_list = []
            v.open_list = open_list = []
            self.solutions=[]
            self.fitness=[]
            self.move_in_log=[]
            self.move_in_model=[]
            self.alignment_move=[]

            # Creating initial node in the search space
            current_node = Node()
            current_node.marking_vector = numpy.array(init_mark_vector[:])
            current_node.observed_trace_remain = trace
            open_list.append([current_node.total_cost, current_node])
            current_node.Heuristic_to_Final(elements_tot)

            #Iterating until open_list has an element
            while len(open_list) > 0:
                # Sorting openlist
                heapq._heapify_max(open_list)

                current_node = heapq.heappop(v.open_list)
                closed_list.append(current_node)
                current_node = current_node[1]

                # Investigating the current node
                current_node.Investigate(Incident_matrix, elements_tot, places, final_marking)

                # Check weather the node if a solution or no:
                if (numpy.array_equal(current_node.marking_vector, final_mark_vector)):
                    self.solutions.append(current_node)
                    self.__Fitness()
                    self.__Move_alignment()
                    self.__Move_in_log()
                    self.__Move_in_model()
#                    v.solutions[key]=[{'alignment': self.alignment_move, 'fitness': self.fitness}]
                    v.solutions[key] = self.alignment_move
                    if len(self.solutions) >= no_of_solutions:
                        break
        return v.solutions

    class __elements:
        def __init__(self, id, type, index):
            self.place_pre = []
            self.place_post = []
            self.trans_id = id
            self.trans_type = type  # This can be synchronous, model or log
            self.index = index  # Index of elements in the trace or model

    def __Synchronous_Product(self, net, trace):
        '''
        :param net: is an object from Petri net class
        :param trace: an input trace, for which sych product will be created
        :return: sync product
        '''
        # net.edges is like [(5, -3),(-2, 5),(6, -4),..]
        # net.transition is like {'t4': [-1],'t5': [-2],'t6': [-3],....}
        elements_tot = []
        t_model = list(net.transitions.keys())
        for i in range(len(t_model)):
            e = self.__elements(t_model[i], "model", i)
            for pair in net.edges:
                if net.transitions[t_model[i]][0] == pair[0]:
                    e.place_post.append(pair[1])
                elif net.transitions[t_model[i]][0] == pair[1]:
                    e.place_pre.append(pair[0])
            elements_tot.append(e)

        # Creating trace net
        counter = 0
        for i in range(len(trace)):
            e = self.__elements(trace[i], "log", i)
            e.place_pre.append("PL" + str(counter))
            counter += 1
            e.place_post.append("PL" + str(counter))
            elements_tot.append(e)

        final_log_place = "PL" + str(counter)
        initial_log_place = "PL" + str(0)

        # Creating synchronous transitions
        for i in range(len(trace)):
            e = self.__elements(trace[i], "Sync", i)
            for node in elements_tot:
                if trace[i] == node.trans_id:
                    if (node.trans_type == 'log' and node.index != i) or (node.trans_type == 'Sync'):
                        pass
                    else:
                        e.place_pre = e.place_pre + node.place_pre
                        e.place_post = e.place_post + node.place_post
            elements_tot.append(e)

        # Creating incidence matrix of synchronous product
        places = list(net.places.values())
        # places=set()
        for e in elements_tot:
            for p in e.place_pre:
                if p not in places:
                    places.append(p)
            for p in e.place_post:
                if p not in places:
                    places.append(p)
        # places = list(places)

        index_place_start_log = places.index(initial_log_place)
        index_place_end_log = places.index(final_log_place)

        # Initializing and filling in the matrix
        incidence_matrix = numpy.zeros((len(places), len(elements_tot),), dtype=int)
        for i in range(len(elements_tot)):
            for p in elements_tot[i].place_pre:
                row_ind = places.index(p)
                incidence_matrix[row_ind][i] = -1
            for p in elements_tot[i].place_post:
                row_ind = places.index(p)
                incidence_matrix[row_ind][i] = 1

        #Storing variables as a Private ones (using "__")
        self.__elements_tot=elements_tot
        self.__places=places
        self.__incidence_matrix=incidence_matrix
        # Plotting synchronous product model
        self.Drawing_Model()

        return elements_tot, places, index_place_start_log, index_place_end_log, incidence_matrix

    def Drawing_Model(self):
        places=self.__places
        incident_matrix=self.__incidence_matrix
        elements_tot=self.__elements_tot
        # path=V.destination_path
        # Now attempting to create a graph

        dot = gr.Digraph()

        for i in range(len(places)):
            dot.node(str(places[i]), shape="circle")

        # Here during the initializing the transition nodes of the graph, if we encountered the transition which is appeared in the net_moves,
        # we highlight it and also assign it a moves number.
        # Similarly transitions which are asynch moves or Skipped, will receive another colors
        for e in elements_tot:
            index = str(e.index)
            if e.trans_type == 'model':
                dot.node(str(e.trans_id + "-" + e.trans_type + index), shape="rect", style='filled', color="red")
            elif e.trans_type == 'log':
                dot.node(str(e.trans_id + "-" + e.trans_type + index), shape="rect", style='filled', color="green")
            elif e.trans_type == 'Sync':
                dot.node(str(e.trans_id + "-" + e.trans_type + index), shape="rect", style='filled', color="blue")

        for i in range(len(incident_matrix)):
            for j in range(len(incident_matrix[0])):
                index = str(elements_tot[j].index)
                if incident_matrix[i][j] == 1:
                    dot.edge(str(elements_tot[j].trans_id + "-" + elements_tot[j].trans_type + index), str(places[i]))
                elif incident_matrix[i][j] == -1:
                    dot.edge(str(places[i]), str(elements_tot[j].trans_id + "-" + elements_tot[j].trans_type + index))
        f = open("./sync_product.dot", "w")
        #dot.render("sync_product.png")
        f.write(dot.source)

    def __Fitness(self):
        for sol in self.solutions:
            u = sol.alignment_Up_to
            self.fitness.append(round(len([e for e in u if ((e[0] != '-' and e[1] != '-') or 'tau' in e[1])]) / len(u), 3))
    def __Move_alignment(self):
        for sol in self.solutions:
            u=sol.alignment_Up_to
            self.alignment_move.append([e for e in u if ('tau' not in e[1])])
    def __Move_in_model(self):
        for sol in self.solutions:
            u=sol.alignment_Up_to
            self.move_in_model.append([e[1] for e in u if (e[1] != '-' and 'tau' not in e[1])])
    def __Move_in_log(self):
        for sol in self.solutions:
            u=sol.alignment_Up_to
            self.move_in_log .append([e[0] for e in u if e[0] != '-'])

class Node():
    def __init__(self):
        #Astar.__init__(self)

        # Shows a marking node
        self.marking_vector = []
        self.active_transition = []
        self.parent_node = ''
        self.observed_trace_remain = []
        self.alignment_Up_to = []  # It is of the form ((t1,t1),(t2,-),(-,t3))
        self.cost_from_init_marking = 1 * sum(
                    [1 if ((x[0] != '-' and x[0] != '-') or ('tau' in x[1])) else 0 for x in
                     self.alignment_Up_to]) / float(max(len(self.alignment_Up_to), 1))

        self.cost_to_final_marking = 1000
        self.total_cost = self.cost_to_final_marking + self.cost_from_init_marking

    # Methods for comparison two objects (Only python 3.7)
    def __lt__(self, other):
        return self.total_cost < other.total_cost

    def Find_active_transition(self, incidence_matrix, elements_tot):
        # Finding active transitions
        # Looping over transitions of the model to see which one is active given the marking of that node
        for i in range(0, incidence_matrix.shape[1]):
            if numpy.all((incidence_matrix[:, i] + self.marking_vector) >= 0):
                if i not in self.active_transition:
                    self.active_transition.append(i)


                    ##################################################################

    # This function calls other functions to investigate the current node
    def Investigate(self, incidence_matrix, elements_tot, places, final_marking):
        '''

        :param incidence_matrix: Incidence matrix of the Petri net (numpy index)
        :param trans: A list of transitions of the model
        :param places: A list of places of the model
        :param final_marking: Final marking of the model, like "p8'
        :param distance_matrix: Distance matrix computed from Floyed algorithm
        :param vertices: Names of vertices of the distance matrix
        :return:
        '''

        # Finding active transitions
        self.Find_active_transition(incidence_matrix, elements_tot)

        # Heuristic evaluation of active transitions
        for i in self.active_transition:
            # Computing the distance of an active transition to the final marking (a place in case of WF net)
            self.Heuristic_to_Final(elements_tot)

            # ------------------------Movements-------------------
            # --Synchronous
            if elements_tot[i].trans_type == "Sync":

                # Creating a new node
                node_child = Node()

                # Updating the current marking for that node by firing trans[i]
                node_child.marking_vector = self.marking_vector + incidence_matrix[:, i]
                node_child.parent_node = self

                # Updating the remaining observed trace
                node_child.observed_trace_remain = self.observed_trace_remain[1:]
                node_child.alignment_Up_to = self.alignment_Up_to + [
                    (self.observed_trace_remain[0], elements_tot[i].trans_id)]
                # node_child.cost_to_final_marking = d
                node_child.Heuristic_to_Final(elements_tot)
                node_child.cost_from_init_marking = 1 * sum(
                    [1 if ((x[0] != '-' and x[0] != '-') or ('tau' in x[1])) else 0 for x in
                     node_child.alignment_Up_to]) / float(
                    len(node_child.alignment_Up_to))

                # Updating active transitions for the child node
                node_child.total_cost = node_child.cost_to_final_marking + node_child.cost_from_init_marking

                self.Add_node(v.open_list, v.closed_list, node_child, ",synch,")

            # Move in model
            elif elements_tot[i].trans_type == "model":
                if numpy.all(self.marking_vector + incidence_matrix[:, i] >= 0):

                    # Creating a new node
                    node_child = Node()
                    # Updating the current marking for that node by firing trans[i]
                    node_child.marking_vector = self.marking_vector + incidence_matrix[:, i]
                    node_child.parent_node = self

                    # Updating the remaining observed trace
                    node_child.observed_trace_remain = self.observed_trace_remain[:]
                    node_child.alignment_Up_to = self.alignment_Up_to + [('-', elements_tot[i].trans_id)]
                    # node_child.cost_to_final_marking = d
                    node_child.Heuristic_to_Final(elements_tot)
                    node_child.cost_from_init_marking = 1 * sum(
                        [1 if ((x[0] != '-' and x[0] != '-') or ('tau' in x[1])) else 0 for x in
                         node_child.alignment_Up_to]) / float(
                        len(node_child.alignment_Up_to))

                    # Updating active transitions for the child node
                    node_child.total_cost = node_child.cost_to_final_marking + node_child.cost_from_init_marking

                    # Checking whether it is in the closed list
                    self.Add_node(v.open_list, v.closed_list, node_child, ",move in model,")

            # Move in log
            elif elements_tot[i].trans_type == "log":
                # Creating a new node
                node_child = Node()
                # Updating the current marking for that node
                node_child.marking_vector = self.marking_vector + incidence_matrix[:, i]
                node_child.parent_node = self

                # Updating the remaining observed trace
                node_child.observed_trace_remain = self.observed_trace_remain[1:]
                node_child.alignment_Up_to = self.alignment_Up_to + [(self.observed_trace_remain[0], '-')]
                # node_child.cost_to_final_marking = self.cost_to_final_marking
                node_child.Heuristic_to_Final(elements_tot)
                node_child.cost_from_init_marking = 1 * sum(
                    [1 if ((x[0] != '-' and x[0] != '-') or ('tau' in x[1])) else 0 for x in
                     node_child.alignment_Up_to]) / float(
                    len(node_child.alignment_Up_to))

                # Updating active transitions for the child node
                node_child.total_cost = node_child.cost_to_final_marking + node_child.cost_from_init_marking

                # Checking whether it is in the closed list
                self.Add_node(v.open_list, v.closed_list, node_child, ',move in log,')

    # Heuristic that estimates from the current node to the final marking how many transitions must be fired!
    def Heuristic_to_Final(self, elements_tot):
        b = numpy.array(v.final_mark_vector) - numpy.array(self.marking_vector)
        x = numpy.linalg.lstsq(v.Incident_matrix, b, rcond=None)[0]
        x[x > 0] = 1
        x[x <= 0] = 0

        for e in elements_tot:
            if 'tau' in e.trans_id:
                x[e.index] = 0

        if numpy.sum(x) > 0:
            self.cost_to_final_marking = 1 / numpy.sum(x)
        else:
            self.cost_to_final_marking = 1000

    # Deciding whether to add a now node to the open list
    def Add_node(self, open_list, closed_list, node_child, id):
        # Checking whether it is in the closed list
        ind = [k for k in range(len(closed_list)) if
               numpy.array_equal(node_child.marking_vector, closed_list[k][1].marking_vector)]
        if len(ind) > 0:
            pass
        # checking whether it is in the open list (update if we found it)
        else:
            # ind is a index list like [12,34,10]
            ind = [k for k in range(len(open_list)) if
                   numpy.array_equal(node_child.marking_vector, open_list[k][1].marking_vector)]
            if ind:
                for k in ind:
                    if open_list[k][1].cost_from_init_marking < node_child.cost_from_init_marking:
                        open_list[k] = [node_child.total_cost, node_child]
                    else:
                        continue
            else:
                open_list.append([node_child.total_cost, node_child])





