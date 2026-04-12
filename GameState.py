import pygame
import sys
from collections import deque
import copy
import random
import heapq


class LightsOutState:

    def __init__(self, board, parent=None):
        self.board = board # 2D list representing the grid (0 = on, 1 = off)

    # Returns True if all lights are off (puzzle solved)
    def winningState(self):
        return all(cell == 0 for row in self.board for cell in row)

    # Returns a new state after toggling a cell and its neighbours
    def move(self,row,column):
        new_board = copy.deepcopy(self.board)
        new_board[row][column] ^= 1 # Toggle the clicked cell

        if(row > 0):
            new_board[row-1][column] ^= 1 # Toggle cell above
        
        if(column > 0):
            new_board[row][column-1] ^= 1 # Toggle cell to the left
        

        if(column < len(self.board)-1):
            new_board[row][column+1] ^= 1 # Toggle cell to the right

        if(row < len(self.board)-1):
            new_board[row+1][column] ^= 1 # Toggle cell below
        return LightsOutState(new_board)
    
    # Prints the board to the console
    def printState(self):
        for i in self.board:
            print(i)

    # Returns all possible moves
    def getPossibleMoves(self):
        n = len(self.board)
        return [(r, c) for r in range(n) for c in range(n)]
    
    # Returns the first move of the optimal solution as a hint
    def getHint(self):
        sol = weighted_astar(self, heuristic1)
        if sol:
            return sol['moves'][0]
        return None

    
    def __eq__(self, other):
        return isinstance(other, LightsOutState) and self.board == other.board

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.board))

# Breadth-First Search: finds the shortest solution
def bfs(initial_state):
    if initial_state.winningState():
        return []
    queue = deque([(initial_state, [])])
    nodesExpanded = 0
    nodesCreated = 0
    visited = {initial_state}
    while queue:
        state, moves = queue.popleft()
        nodesExpanded += 1

        for (r, c) in state.getPossibleMoves():
            if((r, c)) in moves:
                continue
            new_state = state.move(r, c)
            nodesCreated += 1
            if new_state.winningState(): 
                return {
                    "moves": moves + [(r, c)],
                    "nodes_expanded": nodesExpanded,
                    "nodes_created": nodesCreated,
                }
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, moves + [(r,c)]))
    return None

# Depth-First Search: explores deep paths first
# Depth is capped at n² to avoid infinite loops
def dfs(initial_state, max_depth=20):
    if initial_state.winningState():
        return []
    stack = [(initial_state, [], frozenset([initial_state]))]
    nodesExpanded = 0
    nodesCreated = 0

    while stack:
        state, moves, path_visited = stack.pop()
        nodesExpanded += 1
        if len(moves) >= (len(initial_state.board)**2): continue # Skip if we've reached the maximum allowed depth
        for(r, c) in state.getPossibleMoves():
            if((r, c)) in moves:
                continue
            new_state = state.move(r,c)
            nodesCreated += 1
            if new_state.winningState(): 
                return {
                    "moves": moves + [(r, c)],
                    "nodes_expanded": nodesExpanded,
                    "nodes_created": nodesCreated,
                }
            if new_state not in path_visited:
                stack.append((new_state, moves + [(r,c)], path_visited | {new_state}))
    return None

# Iterative Deepening DFS: combines DFS memory efficiency with BFS optimality
def iddfs(initial_state):
    stats = {
        "nodes_expanded":   0,
        "nodes_created":    0,
    }
    def dls(state, moves, path_visited, limit): # Depth-Limited Search — explores up to a given depth limit
        stats["nodes_expanded"]   += 1
        if state.winningState():
            return moves
        if len(moves) == limit:
            return None # Reached depth limit without finding a solution
        for (r, c) in state.getPossibleMoves():
            if((r, c)) in moves:
                continue
            new_state = state.move(r, c)
            stats["nodes_created"] += 1
            if new_state not in path_visited:
                result = dls(new_state, moves + [(r, c)],
                            path_visited | {new_state}, limit)
                if result is not None:
                    return result
        return None
    for limit in range(1,50): # Gradually increase the depth limit until a solution is found
        result = dls(initial_state, [], {initial_state}, limit)
        if result is not None: 
            return {
                "moves":             result,
                "nodes_expanded":    stats["nodes_expanded"],
                "nodes_created":     stats["nodes_created"],
            }
    return None

# Uniform Cost Search: expands nodes by lowest cost (all costs = 1 here)
def ucs(initial_state):
    if initial_state.winningState():
        return []
    counter = 0
    heap = [(0, counter, initial_state, [])]
    visited = {}
    
    nodesExpanded = 0
    nodesCreated = 0


    while heap:
        cost, _, state, moves = heapq.heappop(heap)
        nodesExpanded += 1
        if state in visited: continue
        visited[state] = cost
        if state.winningState():
            return {
                "moves": moves,
                "nodes_expanded": nodesExpanded,
                "nodes_created": nodesCreated,
            }
        for (r,c) in state.getPossibleMoves():
            if((r, c)) in moves:
                continue
            new_state = state.move(r,c)
            nodesCreated += 1
            new_cost = cost + 1 
            if new_state not in visited:
                counter+= 1
                heapq.heappush(heap, (new_cost, counter, new_state, moves + [(r,c)]))
    return None

# Heuristic 1: count of lights still ON 
def heuristic1(state):
    return sum(cell for row in state.board 
                        for cell in row)

# Heuristic 2: number of rows that still have at least one light ON
def heuristic2(state):
    return sum(1 for row in state.board 
                if any(cell != 0 for cell in row))

# Greedy Best-First Search: always expands the most promising state by heuristic
def greedy(initial_state, heuristic):
    if initial_state.winningState():
        return []
    counter = 0
    heap = [(heuristic(initial_state), counter, initial_state, [])]
    visited = {initial_state}

    nodesExpanded = 0
    nodesCreated = 0

    while heap:
        _, _, state, moves = heapq.heappop(heap)
        nodesExpanded += 1

        for (r, c) in state.getPossibleMoves():
            if((r, c)) in moves:
                continue
            new_state = state.move(r, c)
            nodesCreated += 1
            if new_state.winningState():
                return {
                    "moves": moves + [(r,c)],
                    "nodes_expanded": nodesExpanded,
                    "nodes_created": nodesCreated,
                }
            if new_state not in visited:
                visited.add(new_state)
                counter+= 1
                heapq.heappush(heap, (heuristic(new_state), counter, new_state, moves + [(r, c)]))
    return None

# A*: balances path cost (g) and heuristic estimate (h)
def astar(initial_state, heuristic):
    if initial_state.winningState():
        return []
    counter = 0
    heap = [(heuristic(initial_state), counter, initial_state, [])]
    visited = {}

    nodesExpanded = 0
    nodesCreated = 0

    while heap:
        _, _, state, moves = heapq.heappop(heap)
        nodesExpanded += 1
        if state in visited : continue
        visited[state] = len(moves)
        if state.winningState():
            return {
                "moves": moves,
                "nodes_expanded": nodesExpanded,
                "nodes_created": nodesCreated,
            }
        for (r, c) in state.getPossibleMoves():
            if((r, c)) in moves:
                continue
            new_state = state.move(r, c)
            nodesCreated += 1
            if new_state not in visited:
                g = len(moves) + 1
                counter+= 1
                heapq.heappush(heap, (g + heuristic(new_state), counter, new_state, moves + [(r, c)]))
    return None

# Weighted A*: like A* but multiplies heuristic by w
def weighted_astar(initial_state, heuristic, w=2):
    if initial_state.winningState():
        return []
    counter = 0
    heap = [(w * heuristic(initial_state), counter, initial_state, [])]
    visited = {}

    nodesExpanded = 0
    nodesCreated = 0

    while heap:
        _, _, state, moves = heapq.heappop(heap)
        nodesExpanded += 1
        if state in visited : continue
        visited[state] = len(moves)
        if state.winningState():
            return {
                "moves": moves,
                "nodes_expanded": nodesExpanded,
                "nodes_created": nodesCreated,
            }
        for (r, c) in state.getPossibleMoves():
            new_state = state.move(r, c)
            nodesCreated += 1
            if new_state not in visited:
                g = len(moves) + 1
                counter+= 1
                heapq.heappush(heap, (g + w * heuristic(new_state), counter, new_state, moves + [(r, c)]))
    return None

# Generates a random solvable board by making random moves from a solved state
def randomBoard(n = 4, moves = 10):
    board = [[0]*n for _ in range(n)]
    state = LightsOutState(board)
    for _ in range(moves):
        r = random.randint(0, n-1)
        c = random.randint(0, n-1)
        state = state.move(r,c)
    return state
    
# Loads a board from a text file (one row per line, digits as 0/1)
def loadBoardFromFile(filepath):
    board = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                board.append([int(c) for c in line])
    return LightsOutState(board)