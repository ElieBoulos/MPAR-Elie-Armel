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


def normalize_transitions(transitions):
    for state, actions in transitions.items():
        for action, dests in actions.items():
            total_weight = sum(weight for _, weight in dests)
            normalized_transitions = [(dest, weight / total_weight) for dest, weight in dests]
            transitions[state][action] = normalized_transitions
    return transitions

def simulate_markov_chain(transitions, initial_state, target_states, steps, simulations=10000):

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


def select_initial_state(transitions,states):
    button_style = {
            'background': 'lightblue',
            'foreground': 'black',
            'font': ('Helvetica', 15),
            'borderwidth': 5,
            'relief': 'raised',
            'padx': 10,
            'pady': 5
        }
    root = tk.Tk()
    root.title("Select Initial State")
    
    
    visualize_markov_chain_with_pygraphviz(transitions, [])
    image = img.open('markov_chain_visualization.png')
    photo = imgtk.PhotoImage(image)
    image_label = tk.Label(root, image=photo)
    image_label.image = photo
    image_label.pack()

    selected_state = tk.StringVar(root)
    selected_state.set(states[0])  

    tk.Label(root, text=f"Select initial state :").pack()

    tk.OptionMenu(root, selected_state, *states).pack()

    def on_submit():
        global initial_state
        initial_state = selected_state.get()
        root.destroy()

    submit_button = tk.Button(root, text="Submit", command=on_submit,**button_style)
    submit_button.pack()

    root.mainloop()

    return initial_state






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

    def enterDefstates(self, ctx):
        self.current_state = str(ctx.ID(0))
        self.states = [str(x) for x in ctx.ID()]
        



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
    print(simulate_markov_chain(printer.transitions,printer.current_state,['S2'],10,10000))
    # initial_state = select_initial_state(dict,printer.states)
    gwalker = GraphWalker([printer.current_state], dict)

    

if __name__ == '__main__':
    main()
    

