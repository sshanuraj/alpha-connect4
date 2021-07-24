import numpy as np 
import os
from c4Node import Node 
from c4Agent import c4Agent
from c4Grid import c4Grid
import pickle

RED = 2
YELLOW = 1

class DataPack:
    def __init__(self, agent, root):
        self.agent=agent
        self.root=root

class c4:
    def __init__(self, yroot, yAgent, rroot, rAgent, grid):
        self.yroot = yroot
        self.rroot = rroot
        self.grid = grid
        if len(self.yroot.children) == 0:
            self.yroot.populateNode(RED)
        if len(self.rroot.children) == 0:
            self.rroot.populateNode(RED)
        self.rAgent = rAgent
        self.yAgent = yAgent
        self.actions = ["0", "1", "2", "3", "4", "5", "6"]

    def play(self, n_games, n_iterations):
        red_wins=0
        yellow_wins=0
        draw=0
        f=open("red.obj", "wb")
        g=open("yellow.obj", "wb")
        for i in range(n_games):
            win = False
            print("-----  GAME %s  -----\n"%(str(i+1)))
            actions = []
            for j in range(42):
                if j%2 == 0:  #red move
                    action = self.rAgent.getBestMove(actions, n_iterations, self.rroot, self.grid)
                    actions.append(action)
                    self.grid.makeMove(RED, action)
                    self.grid.displayGrid()
                    if self.grid.checkWin():
                        print("RED WINS\n")
                        win = True
                        red_wins+=1
                        break
                else: #yellow move
                    action = self.yAgent.getBestMove(actions, n_iterations, self.yroot, self.grid)
                    actions.append(action)
                    self.grid.makeMove(YELLOW, action)
                    self.grid.displayGrid()
                    if self.grid.checkWin():
                        print("YELLOW WINS\n")
                        win = True
                        yellow_wins+=1
                        break   
            if not win:
                print("DRAW\n")
                draw+=1
            print("-----  GAME %s ENDS  -----\n"%(str(i+1)))        
            self.yAgent.train(actions, self.yroot, 501)
            self.rAgent.train(actions, self.rroot, 501)
            self.grid.resetGrid()   
        print("Red Wins: %d, Yellow wins: %d, Draws: %d"%(red_wins, yellow_wins, draw))
        red_data=DataPack(self.rAgent, self.rroot)
        yellow_data=DataPack(self.yAgent, self.yroot)
        pickle.dump(red_data, f)
        pickle.dump(yellow_data, g)
        f.close()
        g.close()

    def playAgainstRed(self, n_games, n_iterations):
            for i in range(n_games):
                win = False
                print("-----  GAME %s  -----\n"%(str(i+1)))
                
                actions = []
                for j in range(42):
                    if j%2 == 0:  #red move
                        action = self.rAgent.getBestMove(actions, n_iterations, self.rroot, self.grid)
                        actions.append(action)
                        self.grid.makeMove(RED, action)
                        self.grid.displayGrid()
                        if self.grid.checkWin():
                            print("RED WINS\n")
                            win = True
                            break
                    else: #yellow move
                        while True:
                            action = input("Enter move:")
                            if action in self.actions:
                                action = int(action)
                                break
                            else:
                                print("---Enter a number between 0 and 6---")

                        actions.append(action)
                        self.grid.makeMove(YELLOW, action)
                        self.grid.displayGrid()
                        if self.grid.checkWin():
                            print("YELLOW WINS\n")
                            win = True
                            break   
                if not win:
                    print("DRAW\n")
                print("-----  GAME %s ENDS  -----\n"%(str(i+1)))        
                self.grid.resetGrid()

    def playAgainstYellow(self, n_games, n_iterations):
            for i in range(n_games):
                win = False
                print("-----  GAME %s  -----\n"%(str(i+1)))
                actions = []
                for j in range(42):
                    if j%2 == 0:  #red move
                        while True:
                            action = input("Enter move:")
                            if action in self.actions:
                                action = int(action)
                                break
                            else:
                                print("---Enter a number between 0 and 6---")

                        actions.append(action)
                        self.grid.makeMove(RED, action)
                        self.grid.displayGrid()
                        if self.grid.checkWin():
                            print("RED WINS\n")
                            win = True
                            break
                        
                    else: #yellow move
                        action = self.rAgent.getBestMove(actions, n_iterations, self.rroot, self.grid)
                        actions.append(action)
                        self.grid.makeMove(RED, action)
                        self.grid.displayGrid()
                        if self.grid.checkWin():
                            print("YELLOW WINS\n")
                            win = True
                            break
                if not win:
                    print("DRAW\n")
                print("-----  GAME %s ENDS  -----\n"%(str(i+1)))        
                self.grid.resetGrid()


rgrid = c4Grid()
ygrid = c4Grid()
main_grid = c4Grid()

yAgent, rAgent, yroot, rroot = None, None, None, None

if not (os.path.isfile("red.obj") and os.path.isfile("yellow.obj")):
    yroot = Node(0, 0, None, rgrid.grid, rgrid.cols, rgrid.moveCnt, 0, 0, 0)
    rroot = Node(0, 0, None, ygrid.grid, ygrid.cols, ygrid.moveCnt, 0, 0, 0)
    rAgent=c4Agent(RED)
    yAgent=c4Agent(YELLOW)
else:
    f=open("red.obj", "rb")
    g=open("yellow.obj", "rb")
    red_data=pickle.load(f)
    yellow_data=pickle.load(g)
    rroot=red_data.root
    rAgent=red_data.agent
    yroot=yellow_data.root
    yAgent=yellow_data.agent
    f.close()
    g.close()

c4 = c4(yroot, yAgent, rroot, rAgent, main_grid)
c4.play(10, 10000)
c4.playAgainstRed(1, 1000)

