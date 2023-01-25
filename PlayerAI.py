from random import randint
from BaseAI import BaseAI
import math
from itertools import chain
import time

(PLAYER_TURN, COMPUTER_TURN) = (0, 1)
max_depth = 100
INF = math.inf
[UP, DOWN, LEFT, RIGHT] = range(4)
dr = [0,0,1,-1]
dc = [1,-1,0,0]

order1 =  [[32768, 256, 128, 1],
        [ 16384, 512, 64, 2],
        [ 8192, 1024, 32, 4],
        [ 4096, 2048, 16, 8]]

order7 =  [[32768, 16384, 8192, 4096],
            [  256,   512, 1024, 2048],
            [  128,    64,   32,   16],
            [    1,     2,    4,    8]]

s_time = 0
timeLimit = 0.85
timeout = 'no'

'''
comment
코드는 기본적으로 minimax algorithm(with alpha-beta prunning)을 사용.
이 때 탐색순서를 maxValue(player) func에서는 heuristic eval의 값이 큰 순서부터 탐색하였고
minValue(computer) func에서는 heuristic eval의 값이 작은 순서부터 탐색.

사용한 heuristic evaluation function
heuristic2 * 0.2 + heuristic4 + heuristic5

heuristic2
 - row와 column마다 merge가 가능한 cell들을 찾아 그 cell들의 값에 weight를 주어 모두 더한 값 -> weighted merge value
   여기서 weight는 order1을 사용.

heuristic4
 - cell들의 숫자들을 merge가 쉬운 순서대로 만들기 위하여 사용.
   grid에 있는 cell들의 값에 weight를 주어 모두 더한 값 -> weighted board value
   여기서 weight는 order1을 사용.

heuristic5
 - order1의 순서에 맞지 않는 cell들을 보정하기위하여 사용(더 먼저 merge할 수 있게 value를 추가) / (3,0), (0,1), (3,2)부분은 제외
 - order1의 순서대로 merge가 되면 코너부분 (3,0), (3,1), (0,1), (0,2) 등의 cell들이 순서가 안맞게는데 그 부분을 보정하기위해 사용.
   특히 (3,1)의 cell이 옆쪽의 cell과 weight의 값이 많이 차이나기때문에 더 보정하기 어렵다고 생각하여 먼저, (3,1)의 cell을 보정.
   다른 부분의 코너 cell들도 보정해봤지만 오히려 결과 안좋은 영향을 미쳐 현재는 (3,1)의 cell만 보정.

위에 사용한 heuristic eval function만으로는 1초안에 움직일만큼 depth를 줄일 수 없다고 생각하여
직접적으로 시간제한(0.85)를 걸어 탐색.

'''

class PlayerAI(BaseAI):

    def maxValue(self, grid, turn, depth, alpha, beta, mergeVal):
        v = -INF
        
        moves = grid.getAvailableMoves()
        lr_val, ud_val = self.heuristic2(grid)
        
        max_grid_val = -INF
        val_list = []
        # search the larger grid value
        for move in moves:
            gridCopy = grid.clone()
            gridCopy.move(move)
            lr, ud = self.heuristic2(gridCopy)
            val = (lr+ud)*0.2 + self.heuristic4(grid) + self.heuristic5(grid)*1
            val_list.append((val, move))
        
        val_list.sort(reverse=True)

        for _, move in val_list:
            gridCopy = grid.clone()
            gridCopy.move(move)
            new_mergeVal = ud_val if move <= 1 else lr_val

            v = max(v, self.value(gridCopy, turn, depth + 1, alpha, beta, new_mergeVal))
            if v >= beta:
                return v
            alpha = max(alpha, v)

        return v

    def minValue(self, grid, turn, depth, alpha, beta, mergeVal):
        v = INF
        newTiles = [2, 4]
        cells = grid.getAvailableCells()

        min_grid_val = INF
        val_list = []
        # search the smaller grid value
        for r,c in cells:
            gridCopy = grid.clone()
            for nt in newTiles:
                gridCopy.insertTile((r,c), nt)
                lr, ud = self.heuristic2(gridCopy)
                val = (lr+ud)*0.2 + self.heuristic4(grid) + self.heuristic5(grid)*1
                val_list.append((val, r, c))
        
        val_list.sort()

        # using heuristic eval
        for _, r,c in val_list:
            gridCopy = grid.clone()
            for nt in newTiles:
                gridCopy.insertTile((r,c), nt)
                
                v = min(v, self.value(gridCopy, turn, depth + 1, alpha, beta, mergeVal))
                if v <= alpha:
                    return v
                beta = min(beta, v)
                    
        # print(depth, len(cells), cnt)
        return v
    
    def value(self, grid, turn, depth, alpha, beta, mergeVal):
        # print(depth)
        # global time_limit
        if depth == max_depth or not grid.canMove() or time.clock() - s_time > timeLimit:
            if time.clock() - s_time > timeLimit:
                global timeout
                timeout = 'out'
            lr, ud = self.heuristic2(grid) 
            val = (lr+ud)*0.2 + self.heuristic4(grid) + self.heuristic5(grid)*1
            return val
        
        if turn == PLAYER_TURN:
            return self.maxValue(grid, COMPUTER_TURN, depth, alpha, beta, mergeVal)
        if turn == COMPUTER_TURN:
            return self.minValue(grid, PLAYER_TURN, depth, alpha, beta, mergeVal)

        return -1

    def getMove(self, grid): # copy된 grid
        moves = grid.getAvailableMoves()
        max_v, max_move = -INF, -1
        lr_val, ud_val = self.heuristic2(grid)
        global s_time, max_depth
        s_time = time.clock()
        last_move = 0

        val_list = []
        # search the larger grid value
        for move in moves:
            gridCopy = grid.clone()
            gridCopy.move(move)
            lr, ud = self.heuristic2(gridCopy)
            val = (lr+ud)*0.2 + self.heuristic4(grid) + self.heuristic5(grid)*1
            val_list.append((val, move))
        val_list.sort(reverse=True)
        
        # depth 1 - player
        for _, move in val_list:
            gridCopy = grid.clone()
            gridCopy.move(move)
            new_mergeVal = ud_val if move <= 1 else lr_val

            v = self.value(gridCopy, COMPUTER_TURN, 2, -INF, INF, new_mergeVal)
            # print(move, v)
            if max_v < v:
                max_v = v
                max_move = move

        # print('\n' + timeout, max_move)
        return max_move if moves else None
        # moves[randint(0, len(moves) - 1)]

    def range_check(self, row, col):
        if row < 0 or row >= 4 or col < 0 or col >= 4:
            return False
        else:
            return True
    
    # free cells
    def heuristic1(self, grid):
        free_cells = grid.getAvailableCells()
        
        return len(free_cells)

    # count a mergerable cell and compute a weighted merge value
    def heuristic2(self, grid):
        weight = order1
        LR_val = 0
        for r in range(grid.size):
            curr_s = grid.map[r][0]
            curr_w = weight[r][0]
            if curr_s == 0:
                continue
            c = 1
            while c < grid.size:
                if grid.map[r][c] == 0:
                    c += 1
                    continue

                if grid.map[r][c] == curr_s:
                    LR_val += curr_s*curr_w + grid.map[r][c]*weight[r][c] 
                    if c+1 < grid.size:
                        curr_s = grid.map[r][c+1]
                        curr_w = weight[r][c+1]
                    c += 2
                    
                else:
                    curr_s = grid.map[r][c]
                    curr_w = weight[r][c]
                    c += 1

        UD_val = 0
        for c in range(grid.size):
            curr_s = grid.map[0][c]
            curr_w = weight[0][c]
            if curr_s == 0:
                continue
            r = 1
            while r < grid.size:
                if grid.map[r][c] == 0:
                    r += 1
                    continue

                if grid.map[r][c] == curr_s:
                    UD_val += curr_s*curr_w + grid.map[r][c]*weight[r][c] 
                    if r+1 < grid.size:
                        curr_s = grid.map[r+1][c]
                        curr_w = weight[r+1][c]
                    r += 2
                else:
                    curr_s = grid.map[r][c]
                    curr_w = weight[r][c]
                    r += 1
        
        return LR_val, UD_val
                
                
    def heuristic3(self, grid):
        
        flatten_list = list(chain.from_iterable(grid.map))
        flatten_list.sort(reverse=True)

        prod = 1
        for x in flatten_list[:3]:
            if x != 0:
                prod *= x
        top3_logsum = math.log2(prod)

        return top3_logsum

    def heuristic4(self, grid):
        
        val = 0
        order = order1
        for r in range(grid.size): 
            for c in range(grid.size):
                if c + 1 < grid.size and grid.map[r][c] != 0:
                    diff = grid.map[r][c+1] / grid.map[r][c]
                else:
                    diff = 1
                w = diff if diff > 1 else 1
                val += order[r][c] * grid.map[r][c]

        return val

    # order1 맞춤 보정
    def heuristic5(self, grid):
        
        val = 0
        order = order1
        
        # order1의 순서에 맞지 않는 cell들을 보정하기위하여 사용
        for c in range(grid.size-1):
            if c == 0 or c == 2:
                r = 0
            else:
                r = 3
            while r < grid.size and r >= 0:
                if grid.map[r][c] == 0:
                    if c == 0 or c == 2:
                        r += 1
                    else:
                        r -= 1
                    continue

                if (c == 0 or c == 2) and r+1 < 4 and grid.map[r][c] < grid.map[r+1][c]:
                    if grid.map[r][c+1] <= grid.map[r][c]:
                        val += grid.map[r][c]*order[r][c] + grid.map[r][c+1]*order[r][c]
                        # break
                    
                if (c == 1 or c == 3) and r-1 >= 0 and grid.map[r][c] < grid.map[r-1][c]:
                    if grid.map[r][c+1] <= grid.map[r][c]:
                        val += grid.map[r][c]*order[r][c] + grid.map[r][c+1]*order[r][c]
                        # break

                if c == 0 or c == 2:
                    r += 1
                else:
                    r -= 1


        # v3 - 옆칸이 순서 역행일 때 (3,1) - 추가 보정
        if grid.map[3][1] != 0 and grid.map[3][1] < grid.map[2][1]:
            if grid.map[3][2] <= grid.map[3][1] and grid.map[2][2] <= grid.map[3][2]:
                val += grid.map[2][2]*order[3][1]
            # v4
            if grid.map[3][2] <= grid.map[3][1] and grid.map[3][3] <= grid.map[3][2]:
                val += grid.map[3][3]*order[3][1]
        
        return val
