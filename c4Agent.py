import numpy as np 
import random as rd 
import torch
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

pieces={'.':0, 'P':1, 'p':2, 'N':3, 'n':4, 'B':5, 'b':6, 'R':7, 'r':8, 'Q':9, 'q':10, 'K':11, 'k':12}

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

    def addDataTarget(self, data, target):
        data=tuple(data[0])
        self.data_target_map[data]=target

    def make_target(self, node):
        arr=[]
        total=0
        for i in node.children:
            if i!=None:
                arr.append(i.n/node.n)
                total+=i.n
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
            return 0

        if self.color == winColor:
            return 1 #for win
        return -1 #for loss

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
            return 2

        if self.color == winColor:
            return 10 #for win
        return -100000 #for loss


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

        while count < n_iterations:
            if not change: #to reset curr to the initial node
                curr = node
            if curr.checkLeaf():
                # print("in leaf node")
                if curr.n == 0:
                    #start rollout
                    if curr.isTerminal:
                        # print("is terminal in leaf")
                        reward = self.getRewardTerminal(curr.winColor)
                        # print("Backpropagate reward")
                        curr.backpropagate(reward)
                        
                        count += 1
                        change = False
                        continue
                    else:
                        # print("rollout in first visit")
                        vgrid = curr.state.copy()
                        vcols = curr.cols.copy()
                        colorToMove = YELLOW if curr.moveCnt%2 == 1 else RED
                        
                        reward = self.rollout(vgrid, vcols, curr.moveCnt, colorToMove)
                        # print("Backpropagate reward")
                        curr.backpropagate(reward)
                        
                        count += 1
                        change = False
                        continue
                else:
                    #get node
                    colorToMove = YELLOW if curr.moveCnt%2 == 1 else RED
                    # print("Expansion in visited node")

                    if curr.isTerminal:
                        # print("is terminal node ")
                        reward = self.getRewardTerminal(curr.winColor)
                        # print("Backpropagate reward")
                        curr.backpropagate(reward)
                        
                        count += 1
                        change = False
                        continue

                    curr.populateNode(colorToMove)


                    curr, _, _ = curr.getMaxUcbNode(root.n)

                    if curr.isTerminal:
                        # print("is terminal node after expansion")
                        reward = self.getRewardTerminal(curr.winColor)
                        # print("Backpropagate reward")
                        curr.backpropagate(reward)
                        
                        count += 1
                        change = False
                        continue

                    vgrid = curr.state.copy()
                    vcols = curr.cols.copy()

                    colorToMove = YELLOW if curr.moveCnt%2 == 1 else RED

                    # print("Rollout in through expanded node")
                    reward = self.rollout(vgrid, vcols, curr.moveCnt, colorToMove)
                    # print("Backpropagate reward")
                    curr.backpropagate(reward)
                    
                    count += 1
                    change = False
                    continue

            else:
                change = True
                curr, _ , _= curr.getMaxUcbNode(root.n)

        next_node, action, ucbs = node.getMaxUcbNode(root.n)
        
        # print("sending action %s and next node"%(str(action)))
        # print("Total iterations", root.n)
        print(ucbs)
        return action

