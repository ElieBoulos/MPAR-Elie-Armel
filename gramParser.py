# Generated from gram.g4 by ANTLR 4.13.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,13,99,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,5,1,26,8,1,10,1,12,
        1,29,9,1,1,1,1,1,1,2,1,2,4,2,35,8,2,11,2,12,2,36,1,2,1,2,5,2,41,
        8,2,10,2,12,2,44,9,2,1,2,1,2,1,3,1,3,5,3,50,8,3,10,3,12,3,53,9,3,
        1,4,1,4,1,4,3,4,58,8,4,1,5,1,5,3,5,62,8,5,1,6,1,6,1,6,1,6,1,6,1,
        6,1,6,1,6,1,6,1,6,1,6,1,6,5,6,76,8,6,10,6,12,6,79,9,6,1,6,1,6,1,
        7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,5,7,92,8,7,10,7,12,7,95,9,7,1,
        7,1,7,1,7,0,0,8,0,2,4,6,8,10,12,14,0,0,98,0,16,1,0,0,0,2,21,1,0,
        0,0,4,32,1,0,0,0,6,47,1,0,0,0,8,54,1,0,0,0,10,61,1,0,0,0,12,63,1,
        0,0,0,14,82,1,0,0,0,16,17,3,2,1,0,17,18,3,4,2,0,18,19,3,6,3,0,19,
        20,5,0,0,1,20,1,1,0,0,0,21,22,5,1,0,0,22,27,3,8,4,0,23,24,5,7,0,
        0,24,26,3,8,4,0,25,23,1,0,0,0,26,29,1,0,0,0,27,25,1,0,0,0,27,28,
        1,0,0,0,28,30,1,0,0,0,29,27,1,0,0,0,30,31,5,6,0,0,31,3,1,0,0,0,32,
        34,5,2,0,0,33,35,5,12,0,0,34,33,1,0,0,0,35,36,1,0,0,0,36,34,1,0,
        0,0,36,37,1,0,0,0,37,42,1,0,0,0,38,39,5,7,0,0,39,41,5,12,0,0,40,
        38,1,0,0,0,41,44,1,0,0,0,42,40,1,0,0,0,42,43,1,0,0,0,43,45,1,0,0,
        0,44,42,1,0,0,0,45,46,5,6,0,0,46,5,1,0,0,0,47,51,3,10,5,0,48,50,
        3,10,5,0,49,48,1,0,0,0,50,53,1,0,0,0,51,49,1,0,0,0,51,52,1,0,0,0,
        52,7,1,0,0,0,53,51,1,0,0,0,54,57,5,12,0,0,55,56,5,4,0,0,56,58,5,
        11,0,0,57,55,1,0,0,0,57,58,1,0,0,0,58,9,1,0,0,0,59,62,3,12,6,0,60,
        62,3,14,7,0,61,59,1,0,0,0,61,60,1,0,0,0,62,11,1,0,0,0,63,64,5,12,
        0,0,64,65,5,9,0,0,65,66,5,12,0,0,66,67,5,10,0,0,67,68,5,5,0,0,68,
        69,5,11,0,0,69,70,5,4,0,0,70,77,5,12,0,0,71,72,5,8,0,0,72,73,5,11,
        0,0,73,74,5,4,0,0,74,76,5,12,0,0,75,71,1,0,0,0,76,79,1,0,0,0,77,
        75,1,0,0,0,77,78,1,0,0,0,78,80,1,0,0,0,79,77,1,0,0,0,80,81,5,6,0,
        0,81,13,1,0,0,0,82,83,5,12,0,0,83,84,5,5,0,0,84,85,5,11,0,0,85,86,
        5,4,0,0,86,93,5,12,0,0,87,88,5,8,0,0,88,89,5,11,0,0,89,90,5,4,0,
        0,90,92,5,12,0,0,91,87,1,0,0,0,92,95,1,0,0,0,93,91,1,0,0,0,93,94,
        1,0,0,0,94,96,1,0,0,0,95,93,1,0,0,0,96,97,5,6,0,0,97,15,1,0,0,0,
        8,27,36,42,51,57,61,77,93
    ]

class gramParser ( Parser ):

    grammarFileName = "gram.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'States'", "'Actions'", "'transition'", 
                     "':'", "'->'", "';'", "','", "'+'", "'['", "']'" ]

    symbolicNames = [ "<INVALID>", "STATES", "ACTIONS", "TRANSITION", "DPOINT", 
                      "FLECHE", "SEMI", "VIRG", "PLUS", "LCROCH", "RCROCH", 
                      "INT", "ID", "WS" ]

    RULE_program = 0
    RULE_defstates = 1
    RULE_defactions = 2
    RULE_transitions = 3
    RULE_defstate = 4
    RULE_trans = 5
    RULE_transact = 6
    RULE_transnoact = 7

    ruleNames =  [ "program", "defstates", "defactions", "transitions", 
                   "defstate", "trans", "transact", "transnoact" ]

    EOF = Token.EOF
    STATES=1
    ACTIONS=2
    TRANSITION=3
    DPOINT=4
    FLECHE=5
    SEMI=6
    VIRG=7
    PLUS=8
    LCROCH=9
    RCROCH=10
    INT=11
    ID=12
    WS=13

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ProgramContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def defstates(self):
            return self.getTypedRuleContext(gramParser.DefstatesContext,0)


        def defactions(self):
            return self.getTypedRuleContext(gramParser.DefactionsContext,0)


        def transitions(self):
            return self.getTypedRuleContext(gramParser.TransitionsContext,0)


        def EOF(self):
            return self.getToken(gramParser.EOF, 0)

        def getRuleIndex(self):
            return gramParser.RULE_program

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProgram" ):
                listener.enterProgram(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProgram" ):
                listener.exitProgram(self)




    def program(self):

        localctx = gramParser.ProgramContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_program)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 16
            self.defstates()
            self.state = 17
            self.defactions()
            self.state = 18
            self.transitions()
            self.state = 19
            self.match(gramParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DefstatesContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STATES(self):
            return self.getToken(gramParser.STATES, 0)

        def defstate(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(gramParser.DefstateContext)
            else:
                return self.getTypedRuleContext(gramParser.DefstateContext,i)


        def SEMI(self):
            return self.getToken(gramParser.SEMI, 0)

        def VIRG(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.VIRG)
            else:
                return self.getToken(gramParser.VIRG, i)

        def getRuleIndex(self):
            return gramParser.RULE_defstates

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDefstates" ):
                listener.enterDefstates(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDefstates" ):
                listener.exitDefstates(self)




    def defstates(self):

        localctx = gramParser.DefstatesContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_defstates)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 21
            self.match(gramParser.STATES)
            self.state = 22
            self.defstate()
            self.state = 27
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==7:
                self.state = 23
                self.match(gramParser.VIRG)
                self.state = 24
                self.defstate()
                self.state = 29
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 30
            self.match(gramParser.SEMI)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DefactionsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ACTIONS(self):
            return self.getToken(gramParser.ACTIONS, 0)

        def SEMI(self):
            return self.getToken(gramParser.SEMI, 0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.ID)
            else:
                return self.getToken(gramParser.ID, i)

        def VIRG(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.VIRG)
            else:
                return self.getToken(gramParser.VIRG, i)

        def getRuleIndex(self):
            return gramParser.RULE_defactions

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDefactions" ):
                listener.enterDefactions(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDefactions" ):
                listener.exitDefactions(self)




    def defactions(self):

        localctx = gramParser.DefactionsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_defactions)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 32
            self.match(gramParser.ACTIONS)
            self.state = 34 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 33
                self.match(gramParser.ID)
                self.state = 36 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==12):
                    break

            self.state = 42
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==7:
                self.state = 38
                self.match(gramParser.VIRG)
                self.state = 39
                self.match(gramParser.ID)
                self.state = 44
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 45
            self.match(gramParser.SEMI)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TransitionsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def trans(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(gramParser.TransContext)
            else:
                return self.getTypedRuleContext(gramParser.TransContext,i)


        def getRuleIndex(self):
            return gramParser.RULE_transitions

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTransitions" ):
                listener.enterTransitions(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTransitions" ):
                listener.exitTransitions(self)




    def transitions(self):

        localctx = gramParser.TransitionsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_transitions)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 47
            self.trans()
            self.state = 51
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==12:
                self.state = 48
                self.trans()
                self.state = 53
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DefstateContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(gramParser.ID, 0)

        def DPOINT(self):
            return self.getToken(gramParser.DPOINT, 0)

        def INT(self):
            return self.getToken(gramParser.INT, 0)

        def getRuleIndex(self):
            return gramParser.RULE_defstate

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDefstate" ):
                listener.enterDefstate(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDefstate" ):
                listener.exitDefstate(self)




    def defstate(self):

        localctx = gramParser.DefstateContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_defstate)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 54
            self.match(gramParser.ID)
            self.state = 57
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==4:
                self.state = 55
                self.match(gramParser.DPOINT)
                self.state = 56
                self.match(gramParser.INT)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TransContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def transact(self):
            return self.getTypedRuleContext(gramParser.TransactContext,0)


        def transnoact(self):
            return self.getTypedRuleContext(gramParser.TransnoactContext,0)


        def getRuleIndex(self):
            return gramParser.RULE_trans

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTrans" ):
                listener.enterTrans(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTrans" ):
                listener.exitTrans(self)




    def trans(self):

        localctx = gramParser.TransContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_trans)
        try:
            self.state = 61
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 59
                self.transact()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 60
                self.transnoact()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TransactContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.ID)
            else:
                return self.getToken(gramParser.ID, i)

        def LCROCH(self):
            return self.getToken(gramParser.LCROCH, 0)

        def RCROCH(self):
            return self.getToken(gramParser.RCROCH, 0)

        def FLECHE(self):
            return self.getToken(gramParser.FLECHE, 0)

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.INT)
            else:
                return self.getToken(gramParser.INT, i)

        def DPOINT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.DPOINT)
            else:
                return self.getToken(gramParser.DPOINT, i)

        def SEMI(self):
            return self.getToken(gramParser.SEMI, 0)

        def PLUS(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.PLUS)
            else:
                return self.getToken(gramParser.PLUS, i)

        def getRuleIndex(self):
            return gramParser.RULE_transact

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTransact" ):
                listener.enterTransact(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTransact" ):
                listener.exitTransact(self)




    def transact(self):

        localctx = gramParser.TransactContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_transact)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 63
            self.match(gramParser.ID)
            self.state = 64
            self.match(gramParser.LCROCH)
            self.state = 65
            self.match(gramParser.ID)
            self.state = 66
            self.match(gramParser.RCROCH)
            self.state = 67
            self.match(gramParser.FLECHE)
            self.state = 68
            self.match(gramParser.INT)
            self.state = 69
            self.match(gramParser.DPOINT)
            self.state = 70
            self.match(gramParser.ID)
            self.state = 77
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==8:
                self.state = 71
                self.match(gramParser.PLUS)
                self.state = 72
                self.match(gramParser.INT)
                self.state = 73
                self.match(gramParser.DPOINT)
                self.state = 74
                self.match(gramParser.ID)
                self.state = 79
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 80
            self.match(gramParser.SEMI)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TransnoactContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.ID)
            else:
                return self.getToken(gramParser.ID, i)

        def FLECHE(self):
            return self.getToken(gramParser.FLECHE, 0)

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.INT)
            else:
                return self.getToken(gramParser.INT, i)

        def DPOINT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.DPOINT)
            else:
                return self.getToken(gramParser.DPOINT, i)

        def SEMI(self):
            return self.getToken(gramParser.SEMI, 0)

        def PLUS(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.PLUS)
            else:
                return self.getToken(gramParser.PLUS, i)

        def getRuleIndex(self):
            return gramParser.RULE_transnoact

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTransnoact" ):
                listener.enterTransnoact(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTransnoact" ):
                listener.exitTransnoact(self)




    def transnoact(self):

        localctx = gramParser.TransnoactContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_transnoact)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 82
            self.match(gramParser.ID)
            self.state = 83
            self.match(gramParser.FLECHE)
            self.state = 84
            self.match(gramParser.INT)
            self.state = 85
            self.match(gramParser.DPOINT)
            self.state = 86
            self.match(gramParser.ID)
            self.state = 93
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==8:
                self.state = 87
                self.match(gramParser.PLUS)
                self.state = 88
                self.match(gramParser.INT)
                self.state = 89
                self.match(gramParser.DPOINT)
                self.state = 90
                self.match(gramParser.ID)
                self.state = 95
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 96
            self.match(gramParser.SEMI)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





