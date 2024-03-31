from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
import random
import time
import pygraphviz as pgv
from IPython.display import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image as img
from PIL import ImageTk as imgtk 
from scipy.optimize import linprog
import numpy as np


def normalize_transitions(transitions):
    for state, actions in transitions.items():
        for action, dests in actions.items():
            total_weight = sum(weight for _, weight in dests)
            normalized_transitions = [(dest, weight / total_weight) for dest, weight in dests]
            transitions[state][action] = normalized_transitions
    return transitions

#####################################################################################################################################################

def simulate_markov_chain(transitions, initial_state, target_states, actions,steps, simulations=10000):
    if(actions != {'None'}):
        print("Only MP are allowed !")
        return
    successful_simulations = 0
    
    for _ in range(simulations):
        current_state = initial_state
        for _ in range(steps):
            if current_state in target_states:
                successful_simulations += 1
                break 
            if current_state not in transitions or not transitions[current_state]:
                break 
            actions = transitions[current_state].get(None, [])
            if not actions:  
                break
            next_states, probabilities = zip(*actions)
            current_state = random.choices(next_states, weights=probabilities)[0]

    return successful_simulations / simulations


def simulate_expected_reward(transitions, recomp, initial_state, actions,steps, simulations=10000):
    if(actions != {'None'}):
        print("Only MP are allowed !")
        return
    total_reward = 0
    
    for _ in range(simulations):
        current_state = initial_state
        path_reward = 0 
        
        for _ in range(steps):
            path_reward += recomp.get(current_state, 0) 
            
            if current_state not in transitions or not transitions[current_state]:
                break  
            
            actions = transitions[current_state].get(None, [])
            if not actions:
                break 
            
            next_states, probabilities = zip(*actions)
            current_state = random.choices(next_states, weights=probabilities)[0] 
        
        total_reward += path_reward 
    
    expected_reward = total_reward / simulations  
    return expected_reward

def sprt_markov_chain(transitions, initial_state, target_states,actions, steps,theta, epsilon, alpha=0.05, beta=0.05):
    # H0: p >= gamma_0 ,  H1: p < gamma_1
    if(actions != {'None'}):
        print("Only MP are allowed !")
        return
    
    A = ((1 - beta) / alpha)
    B = (beta / (1 - alpha))
    gamma_1 = theta - epsilon
    gamma_0 = theta + epsilon

    Rm = 1
    m = 0
    dm = 0
    while True:
        m += 1

        reached = simulate_markov_chain(transitions, initial_state, target_states, actions, steps,simulations=1)
        
        if reached :
            dm += 1
            Rm *= ((gamma_1)**dm)/((gamma_0)**dm)
        else:
            Rm *= ((1-gamma_1)**(m-dm))/((1-gamma_0)**(m-dm))

        if Rm >= A:
            return "Accept H1 : p < gamma_1", m
        elif Rm <= B:
            return "Accept H0 : p >= gamma_0", m
        

######################################################################################################################################################
def build_transition_matrix_and_vector(transitions, terminal_states, states):

    effective_states = [state for state in states if state not in terminal_states and transitions.get(state, {})]
    state_index = {state: idx for idx, state in enumerate(effective_states)}
    n_states = len(effective_states)
    
    A = np.zeros((n_states, n_states))
    b = np.zeros(n_states)

    for state, actions in transitions.items():
        if state in terminal_states: continue
        i = state_index.get(state, None)
        if i is None: continue  
        
        for action, dests in actions.items():
            total_weight = sum(weight for _, weight in dests)
            for dest, weight in dests:
                if dest in terminal_states:
               
                    b[i] += weight / total_weight
                else:
                    if(dest in effective_states):
                        j = state_index.get(dest)
                        if j is not None:
                            
                            A[i, j] += weight / total_weight
    
    return A, b, effective_states

#Si n_steps = -1, on calcule la probabilité d'arriver pour n'importe quel nombre de transitions (résoudre x = Ax + b)
def model_checking_mc(transitions, terminal_states, n_steps, states,actions):
    
    if any(terminal_state not in states for terminal_state in terminal_states):
        print("Terminal state(s) not found")
        return
    
    if(actions != {'None'}):
        print("Only MP are allowed !")
        return
        
    A, b, effective_states = build_transition_matrix_and_vector(transitions, terminal_states, states)
    print(f"States S? : {effective_states}")
    print(f"A: {A}")
    print(f"b: {b}")

    
    if n_steps == -1:
        I = np.eye(len(effective_states))
        y = np.linalg.solve(I - A, b)
    else:
        y = np.zeros(len(effective_states))
        for _ in range(n_steps):
            y = A.dot(y) + b


    state_to_y_mapping = {state: y[idx] for idx, state in enumerate(effective_states)}
    

    return state_to_y_mapping
########################################################################################################################################################
def build_matrices_mdp(transitions, terminal_states, states, actions):
    effective_states = [state for state in states if state not in terminal_states and transitions.get(state, {})]
    state_idx = {state: i for i, state in enumerate(effective_states)}
    n_states = len(effective_states)
    
    action_rows_per_state = {state: len(transitions.get(state, {})) for state in effective_states}
    total_action_rows = sum(action_rows_per_state.values())
    n_rows = total_action_rows + 2 * n_states
    A = np.zeros((n_rows, n_states))
    I = np.zeros((n_rows, n_states))
    b = np.zeros(n_rows)
    
    num_lines = 0
    
    for i in range(n_states):
        for j in range(num_lines,num_lines+list(action_rows_per_state.values())[i]+2):
            I[j, i] = 1
        num_lines += list(action_rows_per_state.values())[i]+2

    row = 0
    for state in effective_states:
        for action, dests in transitions.get(state, {}).items():
            
            for dest, prob in dests:
                
                    if dest not in terminal_states:
                        if(dest in effective_states):
                            A[row, state_idx[dest]] += prob
                    else:
                        b[row] += prob
            row += 1
        row +=1
    
        A[row, state_idx[state]] = 2
        b[row] = -1
        row += 1 
    
    return A,I,b,effective_states

def model_checking_mdp(transitions, terminal_states, states, actions):
    if any(terminal_state not in states for terminal_state in terminal_states):
        print("Terminal state(s) not found")
        return
    if(actions == {'None'}):
        print("Only MDP are allowed !")
        return
    A,I,b,states = build_matrices_mdp(transitions, terminal_states, states, actions)
    print(f"States S? : {states}")
    print(f"A: {A}")
    print(f"b: {b}")
    
    dict = {}
    
    c = np.ones(A.shape[1])
    res = linprog(c, A_ub=-(I-A), b_ub=-b)
    for i in range(len(states)):
        dict[states[i]]=res.x[i]
    if res.success:
        return dict
        
    else:
        raise ValueError("Failed to solve MDP: " + res.message)

#########################################################################################################################################################

def visualize_markov_chain_with_pygraphviz(transitions, path):
    G = pgv.AGraph(strict=False, directed=True) 

    for source, actions in transitions.items():
        if(actions=={}):
            G.add_node(source, label=f"{source}")
        for action, dests in actions.items():
            action_label = f"{action}, " if action is not None else ""
            for dest, weight in dests:
                G.add_edge(source, dest, label=action_label + f"p={weight}")


    G.graph_attr['rankdir'] = 'LR'
    G.node_attr['shape'] = 'circle'    
    G.node_attr['color'] = 'lightblue'
    G.node_attr['style'] = 'filled'
    if(len(path)>1):
        G.add_edge(path[len(path)-2], path[len(path)-1], color='red')
      

    output_path = 'markov_chain_visualization.png'
    G.draw(output_path, prog='dot', format='png')

    return Image(output_path)




class GraphWalker:

            
    def __init__(self, path, transitions):
        self.path = path
        self.transitions = transitions
        self.window = None
        self.initiate_path()
        
    
    def display_plot(self, image_path):
        image = img.open(image_path)
        photo = imgtk.PhotoImage(image)

        plot_label = tk.Label(self.window, image=photo)
        plot_label.image = photo 
        plot_label.pack()


    def create_window(self):
        button_style = {
            'background': 'lightblue',
            'foreground': 'black',
            'font': ('Helvetica', 15),
            'borderwidth': 5,
            'relief': 'raised',
            'padx': 10,
            'pady': 5
        }



        if self.window:
            self.window.destroy() 
        
        self.window = tk.Tk()  
        self.window.title("Markov Chain Walker")
        

        
        self.display_plot('markov_chain_visualization.png')

        path_str = " -> ".join(self.path)
        tk.Label(self.window, text=f"Path: {path_str}", font=('Helvetica', 12), wraplength=400).pack()
        tk.Label(self.window, text=f"Current state: {self.current_state}").pack()

        for action in self.possible_actions:
            action_text = "Next" if action is None else action
            action_button = tk.Button(self.window, text=action_text, command=lambda a=action: self.choose_action(a),**button_style)
            action_button.pack()
        if(self.possible_actions==[]):
            messagebox.showinfo("End", f"Current State : {self.current_state}, No valid transitions from the current state. Exiting.")

        self.window.mainloop() 

    def choose_action(self, action):

        current_state = self.path[-1]
        action_destinations = self.transitions[current_state][action]
        weights = [dest[1] for dest in action_destinations]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        choice = random.choices(action_destinations, weights=probabilities)[0]
        
        self.path.append(choice[0])
        self.initiate_path()
    
    def initiate_path(self):    
        visualize_markov_chain_with_pygraphviz(self.transitions, self.path)
        self.current_state = self.path[-1]
        try:
            self.possible_actions = list(self.transitions[self.current_state].keys())
        except KeyError:
            if self.window:
                self.window.destroy() 
        
            self.window = tk.Tk()  
            self.window.title("Markov Chain Walker")
        

        
            self.display_plot('markov_chain_visualization.png')

            path_str = " -> ".join(self.path)
            tk.Label(self.window, text=f"Path: {path_str}", font=('Helvetica', 12), wraplength=400).pack()
            tk.Label(self.window, text=f"Current state: {self.current_state}").pack()
            messagebox.showinfo("End", f"Current State : {self.current_state}, No valid transitions from the current state. Exiting.")
            return
        self.create_window()


class gramPrintListener(gramListener):
    def __init__(self):
        self.transitions = {}
        self.actions = set()
        self.states = []
        self.recomp = {}

    def enterDefstate(self, ctx):
        self.states.append(str(ctx.ID()))
        if(str(ctx.INT())!='None'):
            self.recomp[str(ctx.ID())]=int(str(ctx.INT()))
        else:
            self.recomp[str(ctx.ID())]=0
        self.current_state = self.states[0]


    def enterDefactions(self, ctx):
        actions = [str(x) for x in ctx.ID()]
        self.actions.update(actions)
        

    def enterTransnoact(self, ctx):
        source = str(ctx.ID(0))
        if source in self.transitions and any(action is not None for action in self.transitions[source]):
            raise ValueError(f"Conflict detected: Attempting to add a no-action transition for state '{source}' that already has an action-based transition.")
        destinations = [(str(ctx.ID(i)), int(str(ctx.INT(i - 1)))) for i in range(1, len(ctx.ID()), 1)]
        self.transitions.setdefault(source, {}).setdefault(None, []).extend(destinations)

    def enterTransact(self, ctx):
        source = str(ctx.ID(0))
        action = str(ctx.ID(1))
        if source in self.transitions and None in self.transitions[source]:
            raise ValueError(f"Conflict detected: Attempting to add an action-based transition ('{action}') for state '{source}' that already has a no-action transition.")
        # self.actions.add(action)
        destinations = [(str(ctx.ID(i)), int(str(ctx.INT((i - 2))))) for i in range(2, len(ctx.ID()), 1)]
        self.transitions.setdefault(source, {}).setdefault(action, []).extend(destinations)


    
def main():
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        input_stream = FileStream(file_path)
    else:
        print("No input file provided.")
        return  

    lexer = gramLexer(input_stream) #Filestream("ex.mdp")
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)


    for state, actions in printer.transitions.items():
        
        for action in actions.keys():
            if action != None and action not in printer.actions:
                print(f"Warning: Action '{action}' in transitions for state '{state}' is not defined in actions.")

  

    mentioned_states = set()
    mentioned_states.update(printer.transitions.keys())
    for actions in printer.transitions.values():
        for destinations in actions.values():
            mentioned_states.update([state for state, weight in destinations])
    undefined_states = mentioned_states - set(printer.states)

    if undefined_states:
        print("Watning: The following states are mentioned in transitions but not defined in states:", undefined_states)


    for state in printer.states:
        if state not in printer.transitions:
            printer.transitions[state] = {}

    dict = normalize_transitions(printer.transitions)
    ######## for MC ###############
    #print(simulate_markov_chain(printer.transitions,printer.current_state,['S2','S3'],printer.actions,10,10000))
    #print(simulate_expected_reward(printer.transitions,printer.recomp,printer.current_state,printer.actions,10,1000))
    #print(sprt_markov_chain(printer.transitions,printer.current_state,['S1'],printer.actions,10,theta=0.5,epsilon=0.1,alpha=0.05,beta=0.05))
    #print(f"Res : {model_checking_mc(printer.transitions,['S5'],n_steps=-1,states=printer.states,actions=printer.actions)}")
    ###### for MDP ################
    #print(model_checking_mdp(printer.transitions, ['S2'], printer.states, printer.actions))
    
    
    gwalker = GraphWalker([printer.current_state], dict)

    

if __name__ == '__main__':
    main()



    

