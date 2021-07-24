import numpy as np
import random as rd

class c4Grid:
    def __init__(self):
        self.grid = np.zeros((6, 7))
        self.cols = [5, 5, 5, 5, 5, 5, 5] #keep track of where the moves are being made 
        self.moveCnt = 0
        self.lastMove = [-1, -1]
        
    def displayGrid(self):
        dict1 = {0:"-", 1:"Y", 2:"R"}
        for i in range(6):
            for j in range(7):
                print(dict1[self.grid[i][j]], end = " ")
            print()

    def makeMove(self, player, colNo):
        self.grid[self.cols[colNo]][colNo] = player
        self.lastMove = [self.cols[colNo], colNo]
        self.cols[colNo] -= 1
        self.moveCnt += 1


    def inBoundary(self, x, y):
        if x<0 or x>5:
            return False
        if y<0 or y>6:
            return False
        return True

    def checkVirtualDraw(self, state):
        if 0 in state:
            return False
        return True

    def checkDown(self, grid, lastMove):
        x = lastMove[0]
        y = lastMove[1]
        color = grid[x][y] 
        count = 0
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                return False
            x += 1
            count += 1
            if count == 4:
                return True
        return False

    def checkForWinningMove(self, color):
        bcopy = self.grid.copy()
        for i in range(7):
            if self.cols[i] >= 0:
                bcopy[self.cols[i]][i] = color
                if self.checkWinVirtual(bcopy, self.cols[i], i):
                    return i
                bcopy[self.cols[i]][i] = 0
        return -1

    def checkHorizontal(self, grid, lastMove):
        x = lastMove[0]
        y = lastMove[1]
        color = grid[x][y]
        count = 0
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            y += 1
            count += 1
            if count == 4:
                return True

        x = lastMove[0]
        y = lastMove[1] - 1

        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            y -= 1
            count += 1
            if count == 4:
                return True

        if count >= 4:
            return True
        return False

    def checkDiag(self, grid, lastMove):
        x = lastMove[0]
        y = lastMove[1]
        color = grid[x][y]
        count = 0
        # upright 
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            x -= 1
            y += 1
            count += 1
            if count == 4:
                return True

        #downleft
        x = lastMove[0] + 1
        y = lastMove[1] - 1
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            x += 1
            y -= 1
            count += 1
            if count == 4:
                return True

        #upleft
        x = lastMove[0]
        y = lastMove[1]
        count = 0
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            x -= 1
            y -= 1
            count += 1
            if count == 4:
                return True

        #down right
        x = lastMove[0] + 1
        y = lastMove[1] + 1
        while True:
            if not self.inBoundary(x, y) or color != grid[x][y]:
                break
            x += 1
            y += 1
            count += 1
            if count == 4:
                return True
        return False

    def isMovePossible(self, col):
        if self.cols[col] == -1:
            return False
        return True

    def checkWin(self):
        if self.checkDown(self.grid, self.lastMove):
            return True
        if self.checkDiag(self.grid, self.lastMove):
            return True
        if self.checkHorizontal(self.grid, self.lastMove):
            return True
        return False

    def checkWinVirtual(self, grid, x, y):
        lm = [x, y]
        if self.checkDown(grid, lm):
            return True
        if self.checkDiag(grid, lm):
            return True
        if self.checkHorizontal(grid, lm):
            return True
        return False

    def resetGrid(self):
        self.grid = np.zeros((6, 7))
        self.cols = [5, 5, 5, 5, 5, 5, 5]
        self.moveCnt = 0
        self.lastMove = [-1, -1]

