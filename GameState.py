import pygame
import sys
from collections import deque
import copy
import random
import heapq


class LightsOutState:

    def __init__(self, board, parent=None):
        self.board = board

    def winningState(self):
        return all(cell == 0 for row in self.board for cell in row)

    def move(self,row,column):
        new_board = copy.deepcopy(self.board)
        new_board[row][column] ^= 1

        if(row > 0):
            new_board[row-1][column] ^= 1
        
        if(column > 0):
            new_board[row][column-1] ^= 1
        

        if(column < len(self.board)-1):
            new_board[row][column+1] ^= 1

        if(row < len(self.board)-1):
            new_board[row+1][column] ^= 1
        return LightsOutState(new_board)

    def printState(self):
        for i in self.board:
            print(i)

    def getPossibleMoves(self):
        n = len(self.board)
        return [(r, c) for r in range(n) for c in range(n)]
    
    def getHint(self):
        sol = weighted_astar(self, heuristic1)
        if sol:
            return sol['moves'][0]
        return None

    
    def __eq__(self, other):
        return isinstance(other, LightsOutState) and self.board == other.board

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.board))


def bfs(initial_state):
    if initial_state.winningState():
        return []
    queue = deque([(initial_state, [])])
    nodesExpanded = 0
    nodesCreated = 0
    max_states_stored = 0
    visited = {initial_state}
    while queue:
        max_states_stored = max(max_states_stored, len(queue))
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
                    "max_states_stored": max_states_stored,
                }
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, moves + [(r,c)]))
    return None


def dfs(initial_state, max_depth=20):
    if initial_state.winningState():
        return []
    stack = [(initial_state, [], frozenset([initial_state]))]
    nodesExpanded = 0
    nodesCreated = 0
    max_states_stored = 0

    while stack:
        max_states_stored = max(max_states_stored, len(stack))
        state, moves, path_visited = stack.pop()
        nodesExpanded += 1
        if len(moves) >= (len(initial_state.board)**2): continue
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
                    "max_states_stored": max_states_stored,
                }
            if new_state not in path_visited:
                stack.append((new_state, moves + [(r,c)], path_visited | {new_state}))
    return None

def iddfs(initial_state):
    stats = {
        "nodes_expanded":   0,
        "nodes_created":    0,
        "max_states_stored": 0,
    }
    def dls(state, moves, path_visited, limit):
        stats["max_states_stored"] = max(stats["max_states_stored"], len(path_visited))
        stats["nodes_expanded"]   += 1
        if state.winningState():
            return moves
        if len(moves) == limit:
            return None
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
    for limit in range(1,50):
        result = dls(initial_state, [], {initial_state}, limit)
        if result is not None: 
            return {
                "moves":             result,
                "nodes_expanded":    stats["nodes_expanded"],
                "nodes_created":     stats["nodes_created"],
                "max_states_stored": stats["max_states_stored"],
            }
    return None

def ucs(initial_state):
    if initial_state.winningState():
        return []
    counter = 0
    heap = [(0, counter, initial_state, [])]
    visited = {}
    
    nodesExpanded = 0
    nodesCreated = 0
    max_states_stored = 0


    while heap:
        max_states_stored = max(max_states_stored,len(heap))
        cost, _, state, moves = heapq.heappop(heap)
        nodesExpanded += 1
        if state in visited: continue
        visited[state] = cost
        if state.winningState():
            return {
                "moves": moves,
                "nodes_expanded": nodesExpanded,
                "nodes_created": nodesCreated,
                "max_states_stored": max_states_stored,
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


def heuristic1(state):
    # nr celúlas ligadas
    return sum(cell for row in state.board 
                        for cell in row)
    
def heuristic2(state):
    #nr linhas liagdas
    return sum(1 for row in state.board 
                if any(cell != 0 for cell in row))

def greedy(initial_state, heuristic):
    if initial_state.winningState():
        return []
    counter = 0
    heap = [(heuristic(initial_state), counter, initial_state, [])]
    visited = {initial_state}

    nodesExpanded = 0
    nodesCreated = 0
    max_states_stored = 0

    while heap:
        max_states_stored = max(max_states_stored,len(heap))
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
                    "max_states_stored": max_states_stored,
                }
            if new_state not in visited:
                visited.add(new_state)
                counter+= 1
                heapq.heappush(heap, (heuristic(new_state), counter, new_state, moves + [(r, c)]))
    return None

def astar(initial_state, heuristic):
    if initial_state.winningState():
        return []
    counter = 0
    heap = [(heuristic(initial_state), counter, initial_state, [])]
    visited = {}

    nodesExpanded = 0
    nodesCreated = 0
    max_states_stored = 0

    while heap:
        max_states_stored = max(max_states_stored,len(heap))
        _, _, state, moves = heapq.heappop(heap)
        nodesExpanded += 1
        if state in visited : continue
        visited[state] = len(moves)
        if state.winningState():
            return {
                "moves": moves,
                "nodes_expanded": nodesExpanded,
                "nodes_created": nodesCreated,
                "max_states_stored": max_states_stored,
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


def weighted_astar(initial_state, heuristic, w=2):
    if initial_state.winningState():
        return []
    counter = 0
    heap = [(w * heuristic(initial_state), counter, initial_state, [])]
    visited = {}

    nodesExpanded = 0
    nodesCreated = 0
    max_states_stored = 0

    while heap:
        max_states_stored = max(max_states_stored,len(heap))
        _, _, state, moves = heapq.heappop(heap)
        nodesExpanded += 1
        if state in visited : continue
        visited[state] = len(moves)
        if state.winningState():
            return {
                "moves": moves,
                "nodes_expanded": nodesExpanded,
                "nodes_created": nodesCreated,
                "max_states_stored": max_states_stored,
            }
        for (r, c) in state.getPossibleMoves():
            new_state = state.move(r, c)
            nodesCreated += 1
            if new_state not in visited:
                g = len(moves) + 1
                counter+= 1
                heapq.heappush(heap, (g + w * heuristic(new_state), counter, new_state, moves + [(r, c)]))
    return None

def randomBoard(n = 4, moves = 10):
    board = [[0]*n for _ in range(n)]
    state = LightsOutState(board)
    for _ in range(moves):
        r = random.randint(0, n-1)
        c = random.randint(0, n-1)
        state = state.move(r,c)
    return state
    
def loadBoardFromFile(filepath):
    board = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                board.append([int(c) for c in line])
    return LightsOutState(board)