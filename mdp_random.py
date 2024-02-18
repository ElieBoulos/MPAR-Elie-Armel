from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
import random
import time
from graphviz import Digraph
import matplotlib.pyplot as plt

import pygraphviz as pgv
from IPython.display import Image
import matplotlib.image as mpimg

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

    def choosePath(self, start_state, transitions, actions):
        current_state = start_state
        path = []
        count = 0
        while current_state in transitions and count <= 6:
            possible_actions = list(transitions[current_state].keys())
            if not possible_actions:
                break

            action = random.choice(possible_actions)
            action_destinations = transitions[current_state][action]

            weights = [dest[1] for dest in action_destinations]
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            choice = random.choices(action_destinations, weights=probabilities)[0]
            count += 1
            path.append((current_state, action, choice[0]))  
            current_state = choice[0]

        self.path = path

        
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
        print([str(ctx.ID(i)) for i in range(len(ctx.ID()))])
        print([str(ctx.INT(i)) for i in range(len(ctx.INT()))])
        source = str(ctx.ID(0))
        action = str(ctx.ID(1))
        self.actions.add(action)
        destinations = [(str(ctx.ID(i)), int(str(ctx.INT((i - 2))))) for i in range(2, len(ctx.ID()), 1)]
        self.transitions.setdefault(source, {}).setdefault(action, []).extend(destinations)

    
def main():
    lexer = gramLexer(StdinStream()) #Filestream("ex.mdp")
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)
    gwalker = GraphWalker()
    gwalker.choosePath('S0',printer.transitions,printer.actions)

    visualize_markov_chain_with_pygraphviz(printer.transitions, gwalker.path)
    display_image_in_plot('markov_chain_visualization.png')
if __name__ == '__main__':
    main()

