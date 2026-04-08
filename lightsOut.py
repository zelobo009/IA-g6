import pygame
import sys
from collections import deque
import copy
import random
import heapq

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
            if((r, c)) in moves:
                continue
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
            if((r, c)) in moves:
                continue
            new_state = state.move(r,c)
            if new_state.winningState(): 
                return moves + [(r,c)]
            if new_state not in path_visited:
                stack.append((new_state, moves + [(r,c)], path_visited | {new_state}))
    return None

def iddfs(initial_state):
    def dls(state, moves, path_visited, limit):
        if state.winningState():
            return moves
        if len(moves) == limit:
            return None
        for (r, c) in state.getPossibleMoves():
            if((r, c)) in moves:
                continue
            new_state = state.move(r, c)
            if new_state not in path_visited:
                result = dls(new_state, moves + [(r, c)],
                            path_visited | {new_state}, limit)
                if result is not None:
                    return result
        return None
    for limit in range(1,50):
        result = dls(initial_state, [], {initial_state}, limit)
        if result is not None: return result
    return None

def ucs(initial_state):
    if initial_state.winningState():
        return []
    counter = 0
    heap = [(0, counter, initial_state, [])]
    visited = {}

    while heap:
        cost, _, state, moves = heapq.heappop(heap)
        if state in visited: continue
        visited[state] = cost
        if state.winningState():
            return moves
        for (r,c) in state.getPossibleMoves():
            if((r, c)) in moves:
                continue
            new_state = state.move(r,c)
            new_cost = cost + 1 
            if new_state not in visited:
                counter+= 1
                heapq.heappush(heap, (new_cost, counter, new_state, moves + [(r,c)]))
    return None


def heuristic1(state):
    # nr celúlas ligadas
    return sum(cell for row in state.board for cell in row)

def heuristic2(state):
    #nr linhas totalmente desligadas
    return sum(1 for row in state.board if any(cell != 0 for cell in row))

def greedy(initial_state, heuristic):
    if initial_state.winningState():
        return []
    counter = 0
    heap = [(heuristic(initial_state), counter, initial_state, [])]
    visited = {initial_state}

    while heap:
        _, _, state, moves = heapq.heappop(heap)
        for (r, c) in state.getPossibleMoves():
            if((r, c)) in moves:
                continue
            new_state = state.move(r, c)
            if new_state.winningState():
                return moves + [(r, c)]
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

    while heap:
        _, _, state, moves = heapq.heappop(heap)
        if state in visited : continue
        visited[state] = len(moves)
        if state.winningState():
            return moves
        for (r, c) in state.getPossibleMoves():
            if((r, c)) in moves:
                continue
            new_state = state.move(r, c)
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

    while heap:
        _, _, state, moves = heapq.heappop(heap)
        if state in visited : continue
        visited[state] = len(moves)
        if state.winningState():
            return moves
        for (r, c) in state.getPossibleMoves():
            new_state = state.move(r, c)
            if new_state not in visited:
                g = len(moves) + 1
                counter+= 1
                heapq.heappush(heap, (g + w * heuristic(new_state), counter, new_state, moves + [(r, c)]))
    return None

def randomBoard(n = 5, moves = 10):
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
    

BUTTON_H      = 40
BUTTON_MARGIN = 10
BTN_W         = 120
BTN_COLS      = 5
BUTTONS = [
        ("BFS",       pygame.K_b),
        ("DFS",       pygame.K_d),
        ("IDDFS",     pygame.K_i),
        ("UCS",       pygame.K_u),
        ("Greedy H1", pygame.K_1),
        ("Greedy H2", pygame.K_2),
        ("A* H1",     pygame.K_3),
        ("A* H2",     pygame.K_4),
        ("WA* H1",    pygame.K_5),
        ("WA* H2",    pygame.K_6),
    ]

def drawButtons(screen, font, buttons, board_bottom):
    btn_w = 120
    for i, (label, _) in enumerate(buttons):
        col = i % 5
        row = i // 5
        x = 20 + col * (btn_w + 10)
        y = board_bottom + 20 + row * (BUTTON_H + BUTTON_MARGIN)
        rect = pygame.Rect(x, y, btn_w, BUTTON_H)
        pygame.draw.rect(screen, (70, 70, 70), rect, border_radius=8)
        text = font.render(label, True, (255, 255, 255))
        screen.blit(text, (rect.centerx - text.get_width()//2,
                        rect.centery - text.get_height()//2))
        

def get_button_rect(i, board_px):
    col = i % BTN_COLS
    row = i // BTN_COLS
    x   = 20 + col * (BTN_W + 10)
    y   = board_px + 20 + row * (BUTTON_H + BUTTON_MARGIN)
    return pygame.Rect(x, y, BTN_W, BUTTON_H)

def run_algorithm(key, initial_state):
    sol = None
    if   key == pygame.K_b: sol = bfs(initial_state)
    elif key == pygame.K_d: sol = dfs(initial_state)
    elif key == pygame.K_i: sol = iddfs(initial_state)
    elif key == pygame.K_u: sol = ucs(initial_state)
    elif key == pygame.K_1: sol = greedy(initial_state, heuristic1)
    elif key == pygame.K_2: sol = greedy(initial_state, heuristic2)
    elif key == pygame.K_3: sol = astar(initial_state, heuristic1)
    elif key == pygame.K_4: sol = astar(initial_state, heuristic2)
    elif key == pygame.K_5: sol = weighted_astar(initial_state, heuristic1)
    elif key == pygame.K_6: sol = weighted_astar(initial_state, heuristic2)
    return sol

def play():
    GAP     = 6
    PADDING = 30
    LIGHT   = 110

    state         = randomBoard()
    initial_state = LightsOutState(copy.deepcopy(state.board))
    size          = len(state.board)
    board_px      = PADDING * 2 + size * LIGHT + (size - 1) * GAP
    btn_rows      = (len(BUTTONS) + BTN_COLS - 1) // BTN_COLS
    btn_area_h    = btn_rows * (BUTTON_H + BUTTON_MARGIN) + 40
    win_w         = board_px
    win_h         = board_px + btn_area_h

    pygame.init()
    font    = pygame.font.SysFont("helvetica", 18)
    on_img  = pygame.transform.scale(pygame.image.load("sprites/lightson.jpg"),  (LIGHT, LIGHT))
    off_img = pygame.transform.scale(pygame.image.load("sprites/lightsoff.jpg"), (LIGHT, LIGHT))
    screen  = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Lights Out")
    clock   = pygame.time.Clock()

    pc_moves = []
    pc_timer = 0

    while not(state.winningState()):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                sol = run_algorithm(event.key, initial_state)
                if sol is not None:
                    state    = LightsOutState(copy.deepcopy(initial_state.board))
                    pc_moves = list(sol)
                    pc_timer = 0

            if event.type == pygame.MOUSEBUTTONDOWN:
                clicked_button = False
                for i, (_, key) in enumerate(BUTTONS):
                    if get_button_rect(i, board_px).collidepoint(event.pos):
                        sol = run_algorithm(key, state)
                        if sol is not None:
                            state    = LightsOutState(copy.deepcopy(state.board))
                            pc_moves = list(sol)
                            pc_timer = 0
                        clicked_button = True
                        break

                if not clicked_button and not state.winningState():
                    mx, my = event.pos
                    col = (mx - PADDING) // (LIGHT + GAP)
                    row = (my - PADDING) // (LIGHT + GAP)
                    if 0 <= row < size and 0 <= col < size:
                        state    = state.move(row, col)
                        pc_moves = []

        if pc_moves:
            pc_timer -= 1
            if pc_timer <= 0:
                r, c     = pc_moves.pop(0)
                state    = state.move(r, c)
                pc_timer = 40

        drawBoard(screen, state.board, on_img, off_img)
        drawButtons(screen, font, BUTTONS, board_px)
        pygame.display.flip() 
        clock.tick(60)

play()

