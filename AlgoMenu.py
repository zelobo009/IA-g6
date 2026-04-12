import pygame
import GameState
import sys
import tkinter as tk
from tkinter import filedialog

# All available algorithm options, paired with their GameState function calls
ALGO_OPTIONS = [
    ("BFS",          lambda s: GameState.bfs(s)),
    ("DFS",          lambda s: GameState.dfs(s)),
    ("Iteritative Depth DFS",        lambda s: GameState.iddfs(s)),
    ("UCS",          lambda s: GameState.ucs(s)),
    ("Greedy (Light Heuristic)",    lambda s: GameState.greedy(s, GameState.heuristic1)),
    ("Greedy (Line Heuristic)",    lambda s: GameState.greedy(s, GameState.heuristic2)),
    ("A* (Light Heuristic)",        lambda s: GameState.astar(s, GameState.heuristic1)),
    ("A* (Line Heuristic)",        lambda s: GameState.astar(s, GameState.heuristic2)),
    ("Weighted A* (Light Heuristic)",  lambda s: GameState.weighted_astar(s, GameState.heuristic1)),
    ("Weighted A* (Line Heuristic)",  lambda s: GameState.weighted_astar(s, GameState.heuristic2)),

]

# Draws the algorithm selection menu
def drawAlgoMenu(screen, font, title_font, selected, loaded_state):
    screen.fill((20, 20, 40))
    title = title_font.render("Choose Algorithm", True, (255, 215, 0))
    screen.blit(title, (screen.get_width()//2 - title.get_width()//2, 40))
    # Draw a button for each algorithm, highlighting the selected one
    for i, (label, _) in enumerate(ALGO_OPTIONS):
        color = (255, 215, 0) if i == selected else (180, 180, 180)
        bg    = (60, 60, 80)  if i == selected else (40, 40, 60)
        text  = font.render(label, True, color)
        rect  = pygame.Rect(50, 130 + i * 45, 220, 40)
        pygame.draw.rect(screen, bg,    rect, border_radius=10)
        pygame.draw.rect(screen, color, rect, 2, border_radius=10)
        screen.blit(text, (rect.centerx - text.get_width()//2,
                           rect.centery - text.get_height()//2))
    

    loadBtn = pygame.Rect(screen.get_width()//2 + 70 , screen.get_height() - 120, 210, 50)
    board_label = "Click to load Board" if loaded_state is None else "Board: Loaded"
    btn_color   = (255, 215, 0) if loaded_state is None else (180, 180, 180)
    pygame.draw.rect(screen, (60, 60, 80), loadBtn, border_radius=10)
    pygame.draw.rect(screen, btn_color, loadBtn , 2, border_radius=10)
    text = font.render(board_label, True, btn_color)
    screen.blit(text, (loadBtn.centerx - text.get_width()//2,
                        loadBtn.centery - text.get_height()//2))

    hint = font.render("UP/DOWN   ENTER to run   ESC to go back", True, (80, 80, 80))
    screen.blit(hint, (screen.get_width()//2 - hint.get_width()//2,
                       screen.get_height() - 40))
    pygame.display.flip()

# Opens a native file picker dialog and returns the selected file path
def openFileDialog():
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select Board File",
        filetypes=[("Text files", "*.txt")]
    )
    root.destroy()
    return filepath if filepath else None

# Handles algorithm menu logic and input
def algoMenu(screen, font, title_font):
    selected = 0
    clock    = pygame.time.Clock()
    loaded_state = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return (-1,loaded_state, None)
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(ALGO_OPTIONS)
                if event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(ALGO_OPTIONS)
                if event.key == pygame.K_RETURN:
                    return (selected,loaded_state, ALGO_OPTIONS[selected][0])
            if event.type == pygame.MOUSEBUTTONDOWN:
                loadBtn = pygame.Rect(screen.get_width()//2 + 70 , screen.get_height() - 120, 210, 50)
                if(loadBtn.collidepoint(event.pos)):
                    filepath = openFileDialog()
                    if filepath:
                        loaded_state = GameState.loadBoardFromFile(filepath)

                for i in range(len(ALGO_OPTIONS)):
                    rect = pygame.Rect(50, 130 + i * 45, 220, 40)
                    if rect.collidepoint(event.pos):
                        return i

        drawAlgoMenu(screen, font, title_font, selected, loaded_state)
        clock.tick(60)

