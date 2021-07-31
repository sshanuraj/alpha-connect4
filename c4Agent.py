import numpy as np 
import random as rd 
import torch
from log import Logger
import torch.nn as nn
from c4Grid import c4Grid

IN_LEN=43
OUT_LEN=7
NUM_IMG=1
DEVICE=torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
DTYPE=torch.float
LR=1e-3
NUM_ITERATIONS=401
INF = 1000000
RED = 2
YELLOW = 1
DRAW = -1
MAX_MOVES = 42
FINAL_LOSS = -100000
TERMINAL_LOSS = -10000
ROLLOUT_LOSS = -10000
TERMINAL_WIN = 100
ROLLOUT_WIN = 10
TERMINAL_DRAW = 1
ROLLOUT_DRAW = 0

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.denseN=nn.Sequential(
                nn.Linear(IN_LEN, 100),
                nn.Tanh(),
                nn.Linear(100, 50),
                nn.Sigmoid(),
                nn.Linear(50, OUT_LEN),
                nn.Softmax(dim=1)
                )
    def forward(self, x):
        return self.denseN(x)

class c4Agent:
    def __init__(self, color):
        self.color = color
        self.nnObj = NeuralNet().to(device=DEVICE, dtype=DTYPE)
        self.data_target_map={}
        self.logger=Logger("c4Log.txt")

    def addDataTarget(self, data, target):
        data=tuple(data[0])
        self.data_target_map[data]=target

    def make_target(self, node):
        arr=[]
        for i in node.children:
            if i!=None:
                arr.append(i.n/node.n)
            else:
                arr.append(0)
        return arr

    def train(self, actions, root, iterations):
        node=root
        color=RED
        target=[]
        data=[]
        for action in actions:
            temp_target=self.make_target(node)
            temp_data=self.gridToNNState(node.state, color).numpy()
            """
            target.append(temp_target)
            data.append(temp_data[0])
            """
            self.addDataTarget(temp_data, temp_target)
            if len(node.children) > 0:
                node = node.children[action]
            else:
                node=None
            color = self.switchColor(color)
            """
            if not node: #check for when playing against human
                prev_node.populateNode(color)
                node = prev_node.children[action]
            """
        for dataPoint in self.data_target_map.keys():
            data.append(list(dataPoint))
            target.append(self.data_target_map[dataPoint])

        target=torch.tensor(target, device=DEVICE, dtype=DTYPE)
        data=torch.tensor(data, device=DEVICE, dtype=DTYPE)
        loss_fn=nn.MSELoss()
        optimizer=torch.optim.Adam(self.nnObj.parameters(), lr=LR)

        for i in range(iterations):
            res=self.nnObj.forward(data)
            loss=loss_fn(res, target)
            if i%100==0:
                print(res.shape, target.shape)
                print("Loss at %d: %f"%(i, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def getReward(self, winColor):
        if winColor == DRAW:
            return ROLLOUT_DRAW

        if self.color == winColor:
            return ROLLOUT_WIN #for win
        return ROLLOUT_LOSS #for loss

    def gridToNNState(self, state, colorToMove):
        nnState=[]
        temp_inp=[]
        dict1={0:0, 1:0.5, 2:1}
        for i in range(6):
            for j in range(7):
                temp_inp.append(dict1[state[i][j]])
        temp_inp.append(colorToMove)
        nnState.append(temp_inp)
        nnState=torch.tensor(nnState, device=DEVICE, dtype=DTYPE)
        return nnState

    def getArgMax(self, out, cols):
        max_ind=-1
        max_val=-10000
        for i in range(7):
            if cols[i]>=0 and out[i].item()>max_val:
                max_val=out[i].item()
                max_ind=i
        return max_ind
    
    def getNNOutput(self, state, color):
        return self.nnObj.forward(self.gridToNNState(state, color))

    def makeRandomVirtualMove(self, state, cols, color):
        action = -1
        ret=self.nnObj.forward(self.gridToNNState(state, color))
      
        action=self.getArgMax(ret[0], cols)
        state[cols[action]][action] = color
        x = cols[action]
        y = action
        cols[action] -= 1

        return state, cols, x, y

    def switchColor(self, color):
        if color == RED:
            return YELLOW
        return RED

    def rollout(self, vgrid, vcols, moveCnt, colorToMove):
        grid = c4Grid()

        while True:
            vgrid, vcols, x, y = self.makeRandomVirtualMove(vgrid, vcols, colorToMove)
            
            moveCnt += 1
            if moveCnt == 42:
                return 0 #draw reward

            if grid.checkWinVirtual(vgrid, x, y):
                return self.getReward(colorToMove) #return win 

            colorToMove = self.switchColor(colorToMove)

    def getRewardTerminal(self, winColor):
        if winColor == DRAW:
            return TERMINAL_DRAW

        if self.color == winColor:
            return TERMINAL_WIN #for win
        return TERMINAL_LOSS #for loss


    def getBestMove(self, actions, n_iterations, root, grid):
        next_node = None
        action = 0
        count = 0 
        node = root
        prev_node = root
        color = YELLOW

        for action in actions:
            prev_node = node

            if len(node.children) > 0:
                node = node.children[action]
            else:
                node = None
            color = self.switchColor(color)

            if not node: #check for when playing against human
                prev_node.populateNode(color)
                node = prev_node.children[action]

        if node.checkLeaf():
            node.populateNode(self.color)

        curr = node
        change = False

        print(self.getNNOutput(node.state, self.color)) 
        
        while count < n_iterations:
            if not change: #to reset curr to the initial node
                #self.logger.log("LOG", "-------Running iteration %d-------"%(count+1))
                curr = node
            if curr.checkLeaf():
                if curr.n == 0:
                    if curr.isTerminal:
                        reward = self.getRewardTerminal(curr.winColor)
                        curr.backpropagate(reward)
                        #self.logger.log("LOG", "Got reward: %d, in leaf+terminal+unvisited node"%(reward))
                        count += 1
                        change = False
                        #self.logger.log("LOG", "-------Ending iteration %d-------"%(count+1))
                        continue
                    else:
                        vgrid = curr.state.copy()
                        vcols = curr.cols.copy()
                        colorToMove = YELLOW if curr.moveCnt%2 == 1 else RED
                        reward = self.rollout(vgrid, vcols, curr.moveCnt, colorToMove)
                        curr.backpropagate(reward)
                        #self.logger.log("LOG", "Got reward: %d, in leaf+unvisited node"%(reward))
                        
                        count += 1
                        change = False
                        #self.logger.log("LOG", "-------Ending iteration %d-------"%(count))
                        continue
                else:
                    colorToMove = YELLOW if curr.moveCnt%2 == 1 else RED

                    if curr.isTerminal:
                        reward = self.getRewardTerminal(curr.winColor)
                        curr.backpropagate(reward)
                        #self.logger.log("LOG", "Got reward: %d, in leaf+terminal node"%(reward))
                        count += 1
                        change = False
                        #self.logger.log("LOG", "-------Ending iteration %d-------"%(count))
                        continue

                    curr.populateNode(colorToMove)


                    curr, _, _ = curr.getMaxUcbNode(root.n)

                    if curr.isTerminal:
                        reward = self.getRewardTerminal(curr.winColor)
                        curr.backpropagate(reward)
                        #self.logger.log("LOG", "Got reward: %d, in leaf_expanded+terminal node"%(reward))
                        count += 1
                        change = False
                        #self.logger.log("LOG", "-------Ending iteration %d-------"%(count))
                        continue

                    vgrid = curr.state.copy()
                    vcols = curr.cols.copy()

                    colorToMove = YELLOW if curr.moveCnt%2 == 1 else RED
                    reward = self.rollout(vgrid, vcols, curr.moveCnt, colorToMove)
                    curr.backpropagate(reward)
                    #self.logger.log("LOG", "Got reward: %d, in expanded+leaf node"%(reward))
                    count += 1
                    change = False
                    #self.logger.log("LOG", "-------Ending iteration %d-------"%(count))
                    continue

            else:
                change = True
                #self.logger.log("LOG", "Already visited, choosing max_ucb node")
                curr, _ , _= curr.getMaxUcbNode(root.n)

        next_node, action, ucbs = node.getMaxUcbNode(root.n)
        print(ucbs)
        return action

    def feedFinalReward(self, actions, root, res):
        node = root
        prev_node = root
        color = YELLOW

        for action in actions:
            prev_node = node

            if len(node.children) > 0:
                node = node.children[action]
            else:
                node = None
            color = self.switchColor(color)

            if not node: #check for when playing against human
                prev_node.populateNode(color)
                node = prev_node.children[action]

        if res=="LOSS":
            node.backpropagate(FINAL_LOSS)
        elif res=="WIN":
            node.backpropagate(TERMINAL_WIN)
        else:
            node.backpropagate(TERMINAL_DRAW)

