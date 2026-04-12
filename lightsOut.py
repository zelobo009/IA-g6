import pygame
import sys
import copy
import time
import tracemalloc
import GameState
import EndMenu
import MainMenu
import AlgoMenu

    
# Draws the current board state on screen
def drawBoard(screen, board, on_img, off_img, hint):
    LIGHT = 110
    ON_COLOR  = (255,215,0)
    OFF_COLOR = (0,0,0)
    size = len(board)

    screen.fill((20, 20, 40))
    hasHint = False
    if(hint[0] != -1):
        hasHint = True

    PADDING = int((screen.get_width() - (LIGHT+10)*size)/2)
    for i in range(size):
        for j in range(size):
            x = PADDING + (LIGHT+10) * j
            y = PADDING + (LIGHT+10) * i
            if hasHint and (i,j) == hint:
                pygame.draw.rect(screen, (0, 255, 100),
                                     (x, y, LIGHT, LIGHT), 4, border_radius=6)
            elif board[i][j] == 0:
                screen.blit(on_img, (x, y)) # Light is ON
            else:
                screen.blit(off_img, (x, y)) # Light is OFF
    

BUTTON_H      = 40
BUTTON_MARGIN = 10
BTN_W         = 120
BTN_COLS      = 5


# Runs the selected algorithm and returns the solution
def run_algorithm(key, initial_state):
    sol = None
    if   key == 0: sol = GameState.bfs(initial_state)
    elif key == 1: sol = GameState.dfs(initial_state)
    elif key == 2: sol = GameState.iddfs(initial_state)
    elif key == 3: sol = GameState.ucs(initial_state)
    elif key == 4: sol = GameState.greedy(initial_state, GameState.heuristic1)
    elif key == 5: sol = GameState.greedy(initial_state, GameState.heuristic2)
    elif key == 6: sol = GameState.astar(initial_state, GameState.heuristic1)
    elif key == 7: sol = GameState.astar(initial_state, GameState.heuristic2)
    elif key == 8: sol = GameState.weighted_astar(initial_state, GameState.heuristic1)
    elif key == 9: sol = GameState.weighted_astar(initial_state, GameState.heuristic2)
    return sol

# Main game loop
def play(): 
    GAP     = 5
    LIGHT   = 110
    PADDING = 30
    state         = GameState.randomBoard() # Generate a random starting board
    initial_state = GameState.LightsOutState(copy.deepcopy(state.board))
    size          = 5
    board_px      = PADDING * 2 + size * LIGHT + (size - 1) * GAP
    win_w         = board_px
    win_h         = board_px + 60 

    pygame.init()
    font    = pygame.font.SysFont("helvetica", 25)
    smaller_font =  pygame.font.SysFont("helvetica", 20)
    title_font    = pygame.font.SysFont("helvetica", 45)
    on_img  = pygame.transform.scale(pygame.image.load("sprites/lightson.jpg"),  (LIGHT, LIGHT))
    off_img = pygame.transform.scale(pygame.image.load("sprites/lightsoff.jpg"), (LIGHT, LIGHT))
    screen  = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Lights Out")
    clock   = pygame.time.Clock()

    pc_moves = []
    pc_timer = 0
    while (True):
        mode = MainMenu.menu(screen,font,title_font);
        # Player mode
        if(mode == 0):
            n = MainMenu.chooseSizeMenu(screen,font,title_font)
            if(n == -1):
                continue

            else:
                size = n
                state = GameState.randomBoard(n,20)
                PADDING = (screen.get_width() - (LIGHT+10)*size)/2

                while(True):
                    moves = 0
                    Nhints = 0
                    hint = (-1,-1)
                    while not(state.winningState()):
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit(); sys.exit()

                            if event.type == pygame.MOUSEBUTTONDOWN:
                                mx, my = event.pos
                                col = int((mx - PADDING) // (LIGHT + GAP))
                                row = int((my - PADDING) // (LIGHT + GAP))
                                if 0 <= row < size and 0 <= col < size:
                                    state = state.move(row, col)
                                    moves += 1
                                    hint = (-1, -1)

                            # Request a hint for the current state
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_h:
                                    hint = state.getHint()
                                    Nhints += 1

                        drawBoard(screen, state.board, on_img, off_img, hint)
                        msg = font.render("H = Hint", True, (150, 150, 150))
                        screen.blit(msg, (board_px/2 - 15, board_px + 15))
                        pygame.display.flip() 
                        clock.tick(60)
                    # Board solved show end screen
                    restart = EndMenu.endPlayerMenu(screen, font, title_font, moves, Nhints)
                    if(restart):
                        n = MainMenu.chooseSizeMenu(screen,font, title_font)
                        if(n == -1):
                            break
                        else:
                            size = n
                            state = GameState.randomBoard(n,20)
                            PADDING = (screen.get_width() - (LIGHT+10)*size)/2

                    else:
                        return
        else: # Algorithm mode
            while(True):
                (sel,loaded_state,algName) = AlgoMenu.algoMenu(screen,smaller_font,title_font)
                if(sel == -1):
                    break

                else:
                    if(loaded_state == None):
                        loaded_state = GameState.randomBoard()
                    # Run the algorithm and measure time and memory usage
                    tracemalloc.start()
                    t0  = time.time()
                    res = run_algorithm(sel, loaded_state)
                    elapsed = time.time() - t0
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    pc_moves = copy.copy(res['moves'])
                    pc_timer = 40
                    state = copy.deepcopy(loaded_state)
                    while(len(pc_moves) > 0): # Replay each move step by step
                        pc_timer -= 1
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit(); sys.exit()
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    pc_moves = []  # stop playback

                        if pc_timer <= 0:
                            r, c     = pc_moves.pop(0)
                            state    = state.move(r, c)
                            pc_timer = 40

                        if state.winningState():
                            break
                        
                        drawBoard(screen,state.board, on_img, off_img, (-1,-1))
                        pygame.display.flip()
                        clock.tick(60)

                    sel = EndMenu.AlgStatMenu(screen, font, title_font, res, elapsed, algName, loaded_state, peak)

                    if(sel == -1):
                        break

                    return
    

play()

