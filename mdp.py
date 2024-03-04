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




def visualize_markov_chain_with_pygraphviz(transitions, path):
    G = pgv.AGraph(strict=False, directed=True) 

    for source, actions in transitions.items():
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


def select_initial_state(states):
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
    root.geometry("200x100")

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
        print("States: %s" % str([str(x) for x in ctx.ID()]))

    def enterDefactions(self, ctx):
        actions = [str(x) for x in ctx.ID()]
        self.actions.update(actions)
        print("Actions: %s" % actions)

    def enterTransnoact(self, ctx):
        source = str(ctx.ID(0))
        destinations = [(str(ctx.ID(i)), int(str(ctx.INT(i - 1)))) for i in range(1, len(ctx.ID()), 1)]
        self.transitions.setdefault(source, {}).setdefault(None, []).extend(destinations)

    def enterTransact(self, ctx):
        source = str(ctx.ID(0))
        action = str(ctx.ID(1))
        self.actions.add(action)
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
    
    initial_state = select_initial_state(printer.states)
    print(f"Selected initial state: {initial_state}")

    gwalker = GraphWalker([initial_state], printer.transitions)
    
    

if __name__ == '__main__':
    main()
    

