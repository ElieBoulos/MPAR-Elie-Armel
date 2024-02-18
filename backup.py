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


plt.ion()

def visualize_markov_chain_with_pygraphviz(transitions, path_with_actions):
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


    if path_with_actions:
        for i in range(len(path_with_actions)):
            source, action, dest = path_with_actions[i]

            G.get_edge(source, dest).attr['color'] = 'red'
            G.get_edge(source, dest).attr['penwidth'] = 2


    output_path = 'markov_chain_visualization.png'
    G.draw(output_path, prog='dot', format='png')


    return Image(output_path)

def display_image_in_plot(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off') 
    plt.show()




class GraphWalker:
    def __init__(self):
        self.path = []
    def choosePath(self, start_state, transitions):
        current_state = start_state
        path = []
        action =''
        while True:
            try:
                possible_actions = list(transitions[current_state].keys())
            except KeyError:
                print(f"No valid transitions from state '{current_state}'. Exiting.")
                break
            
            action = possible_actions[0]
            if(len(possible_actions)>1):
                print(f"Current state: {current_state}")
                print("Available actions: " + ", ".join([str(action) for action in possible_actions]))
            
                action = input("Choose an action (or type 'exit' to quit): ")
                if action.lower() == 'exit':
                    break
            
                if action not in possible_actions:
                    print("Invalid action selected. Please try again.")
                    continue

                action_destinations = transitions[current_state].get(action, [])
                if not action_destinations:
                    print("No destinations available for this action. Please try again.")
                    continue
            action_destinations = transitions[current_state][action]

            weights = [dest[1] for dest in action_destinations]
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            choice = random.choices(action_destinations, weights=probabilities)[0]
            
            self.path.append((current_state, action, choice[0]))  
            current_state = choice[0]
            
            visualize_markov_chain_with_pygraphviz(transitions, self.path)
            display_image_in_plot('markov_chain_visualization.png')
            input("Next ?")
        
class gramPrintListener(gramListener):
    def __init__(self):
        self.transitions = {}
        self.actions = set()

    def enterDefstates(self, ctx):
        self.current_state = str(ctx.ID(0))
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
    gwalker = GraphWalker()
    gwalker.choosePath('S0', printer.transitions)

if __name__ == '__main__':
    main()

