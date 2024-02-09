from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
import random
import time
from graphviz import Digraph



def visualize_markov_chain_with_path(transitions, path):
    dot = Digraph(comment='Markov Chain with Selected Path')

    for source, dests in transitions.items():
        dot.node(source, source)  
        for dest, weight in dests:
            dot.node(dest, dest)  
            dot.edge(source, dest, label=str(weight))


    for i, (source, destination) in enumerate(path):
        if i == 0:  
            dot.node(source, source, style='filled', fillcolor='lightblue')
        
        dot.edge(source, destination, color='red', penwidth='2.0')
        dot.node(destination, destination, style='filled', fillcolor='lightblue')

    
    dot.render('markov_chain_path', view=True)

        
class gramPrintListener(gramListener):

    def __init__(self):

        self.transitions = {}
        self.current_state = None  
        self.path = []
      
        
    def enterDefstates(self, ctx):
        self.current_state = str(ctx.ID(0))
        print("States: %s" % str([str(x) for x in ctx.ID()]))

    # def enterDefactions(self, ctx):
    #     print("Actions: %s" % str([str(x) for x in ctx.ID()]))

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        print("Transition from " + dep + 
              " with action "+ act + 
              " and targets " + str(ids) + " with weights " + str(weights))
        
    # def enterTransnoact(self, ctx):
    #     ids = [str(x) for x in ctx.ID()]
    #     dep = ids.pop(0)
    #     weights = [int(str(x)) for x in ctx.INT()]
    #     print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))

    def enterTransnoact(self, ctx):
        source = str(ctx.ID(0))
        destinations = [(str(ctx.ID(i)), int(str(ctx.INT(i -1)))) for i in range(1, len(ctx.ID()), 1)]
        print(destinations)
        if source not in self.transitions:
            self.transitions[source] = destinations
        else:
            self.transitions[source].extend(destinations)

        if source == self.current_state:  
            weights = [dest[1] for dest in destinations]
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            random.seed(time.time())
            choice = random.choices(destinations, weights=probabilities)[0]
            self.current_state = choice[0]  
            self.path.append((source, choice[0])) 
    

def main():
    lexer = gramLexer(StdinStream()) #Filestream("ex.mdp")
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)
    visualize_markov_chain_with_path(printer.transitions, printer.path)
if __name__ == '__main__':
    main()
