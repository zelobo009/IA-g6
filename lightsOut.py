import pygame
import sys
from collections import deque
import copy
import random

class LightsOutState:

    def __init__(self, board):
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
    
    def __eq__(self, other):
        return isinstance(other, LightsOutState) and self.board == other.board

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.board))


def bfs(initial_state):
    if initial_state.winningState():
        return []
    queue = deque([(initial_state, [])])
    visited = {initial_state}
    while queue:
        state, moves = queue.popleft()
        for (r, c) in state.getPossibleMoves():
            new_state = state.move(r, c)
            if new_state.winningState(): 
                return moves + [(r,c)]
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, moves + [(r,c)]))
    return None


def dfs(initial_state, max_depth=20):
    if initial_state.winningState():
        return []
    stack = [(initial_state, [], frozenset([initial_state]))]

    while stack:
        state, moves, path_visited = stack.pop()
        if len(moves) >= max_depth: continue
        for(r, c) in state.getPossibleMoves():
            new_state = state.move(r,c)
            if new_state.winningState(): 
                return moves + [(r,c)]
            if new_state not in path_visited:
                stack.append((new_state, moves + [(r,c)], path_visited | {new_state}))
    return None


def randomBoard(n = 4, moves = 10):
    board = [[0]*n for _ in range(n)]
    state = LightsOutState(board)
    for _ in range(moves):
        r = random.randint(0, n-1)
        c = random.randint(0, n-1)
        state = state.move(r,c)
    return state



def drawBoard(screen, board, on_img, off_img):
    LIGHT = 110
    ON_COLOR  = (255,215,0)
    OFF_COLOR = (0,0,0)
    BG_COLOR = (200,200,200)
    size = len(board)

    screen.fill(BG_COLOR)

    for i in range(size):
        for j in range(size):
            x = 20 + (LIGHT+10) * j
            y = 20 + (LIGHT+10) * i
            if board[i][j] == 0:
                screen.blit(on_img, (x, y))
            else:
                screen.blit(off_img, (x, y))
    
    pygame.display.flip()


def play():

    GAP       = 6
    PADDING   = 30

    state = randomBoard()
    initial_state = LightsOutState(copy.deepcopy(state.board))
    size = len(state.board)
    win_px = PADDING * 2 + size * 110 + (size - 1) * GAP
    pygame.init()
    on_img = pygame.image.load("sprites/lightson.jpg")
    off_img = pygame.image.load("sprites/lightsoff.jpg")
    on_img = pygame.transform.scale(on_img, (110, 110))
    off_img = pygame.transform.scale(off_img, (110, 110))
    screen = pygame.display.set_mode((win_px, win_px))
    clock = pygame.time.Clock()
    pc_moves = []
    pc_timer = 0

    while not state.winningState():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_b:        # BFS
                    sol = bfs(initial_state)
                    if sol:
                        state    = LightsOutState(copy.deepcopy(initial_state.board))
                        pc_moves = list(sol)
                        pc_timer = 0
                if event.key == pygame.K_d:            # DFS
                        sol = dfs(initial_state)
                        if sol:
                            state    = LightsOutState(copy.deepcopy(initial_state.board))
                            pc_moves = list(sol)
                            pc_timer = 0
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                col = (mx - 20) // (110 + 10)
                row = (my - 20) // (110 + 10)
                if 0 <= row < size and 0 <= col < size:
                    state = state.move(row, col)
                    pc_moves = []
            
        if pc_moves:
            pc_timer -= 1
            if pc_timer <= 0:
                r, c     = pc_moves.pop(0)
                state    = state.move(r, c)
                pc_timer = 40
                
        drawBoard(screen, state.board, on_img, off_img)
        clock.tick(60)
    
    print("You won")


play()