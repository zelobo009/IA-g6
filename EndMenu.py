import pygame
import datetime as dt
import sys
import copy
import GameState

# Draws the "You Won!" screen with move/hint stats and Restart/Exit buttons
def drawPlayerStats(screen, font, title_font, moves, hints,selected):
    screen.fill((20, 20, 40))

    title = title_font.render("You Won!", True, (255, 215, 0))
    screen.blit(title, (screen.get_width()//2 - title.get_width()//2, 60))

    # Display move and hint counts
    lines = [
        f"Moves:        {moves}",
        f"Hints:        {hints}",
    ]
    for i, line in enumerate(lines):
        color = (180, 230, 180)
        text  = font.render(line, True, color)
        screen.blit(text, (screen.get_width()//2 - text.get_width()//2, 150 + (i * 30)))
    # Draw Restart and Exit buttons
    options = ["Restart", "Exit"]
    for i, option in enumerate(options):
        color = (255, 215, 0) if i == selected else (180, 180, 180)
        bg    = (60, 60, 80)  if i == selected else (40, 40, 60)
        text  = font.render(option, True, color)
        rect  = pygame.Rect(screen.get_width()//2 - 120, 250 + i * 90, 240, 60)
        pygame.draw.rect(screen, bg, rect, border_radius=12)
        pygame.draw.rect(screen, color, rect, 2, border_radius=12)
        screen.blit(text, (rect.centerx - text.get_width()//2,
                           rect.centery - text.get_height()//2))
    # Navigation hint at the bottom
    hint = font.render("UP/DOWN   ENTER to confirm", True, (80, 80, 80))
    screen.blit(hint, (screen.get_width()//2 - hint.get_width()//2,
                       screen.get_height() - 50))
    pygame.display.flip()

# Handles the end-of-game menu for the player
# Returns True if the player wants to restart, False to exit
def endPlayerMenu(screen, font, title_font, moves, hints):
    restart = False
    clock    = pygame.time.Clock()
    selected = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % 2
                if event.key == pygame.K_DOWN:
                    selected = (selected + 1) % 2
                if event.key == pygame.K_RETURN:
                    return selected == 0 
            if event.type == pygame.MOUSEBUTTONDOWN:
                for i in range(2):
                    rect = pygame.Rect(screen.get_width()//2 - 120, 250 + i * 90, 240, 60)
                    if rect.collidepoint(event.pos):
                        return i
            
            drawPlayerStats(screen,font,title_font,moves,hints,selected)
            pygame.display.flip() 
            clock.tick(60)

# Draws the algorithm results screen with performance statistics
def drawAlgStatMenu(screen, font, title_font, stats, time_elapsed, algname, initialState, peakMem):

    screen.fill((20, 20, 40))

    title = title_font.render("Results", True, (255, 215, 0))
    screen.blit(title, (screen.get_width()//2 - title.get_width()//2, 50))

    algo_text = font.render(f"Algorithm:  {algname}", True, (255, 215, 0))
    screen.blit(algo_text, (screen.get_width()//2 - algo_text.get_width()//2, 130))
    # Display all performance stats
    lines = [
        f"Moves:           {len(stats['moves'])}",
        f"Nodes Expanded:  {stats['nodes_expanded']}",
        f"Nodes Generated: {stats['nodes_created']}",
        f"Max Memory Used: {peakMem/1024:.0f} KB",
        f"Time:            {time_elapsed:.3f} s",
    ]
    for i, line in enumerate(lines):
        text = font.render(line, True, (180, 230, 180))
        screen.blit(text, (screen.get_width()//2 - text.get_width()//2,
                            200 + i * 55))

    hint = font.render("ESC = go back", True, (80, 80, 80))
    screen.blit(hint, (screen.get_width()//2 - hint.get_width()//2,
                        screen.get_height() - 40))
    pygame.display.flip()

    return

# Handles the algorithm stats screen; saves results and waits for ESC
def AlgStatMenu(screen, font, title_font, stats, time_elapsed, algname, initialState, peakMem):
        clock = pygame.time.Clock()
        # Save the results to a file as soon as this screen is shown
        saveResults(algname, initialState, stats['moves'], stats, time_elapsed, peakMem)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return -1
            
            drawAlgStatMenu(screen, font, title_font, stats, time_elapsed, algname,  initialState, peakMem)
            clock.tick(60)

# Saves algorithm results and the full move-by-move board replay to a file
def saveResults(algo_name, initial_state, moves, stats, time_elapsed, peakMem):
    # Build a unique filename using the algorithm name and current timestamp
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S") 
    algo_slug = algo_name.replace(" ", "_").replace("*", "star")
    filepath  = f"outputs/{algo_slug}_{timestamp}.txt"

    with open(filepath, "a") as f:
        f.write(f"Algorithm: {algo_name}\n")

        f.write("="*30 + "\n")
        f.write("Stats:\n")
        f.write(f"  Moves:           {len(stats['moves'])}\n")
        f.write(f"  Nodes Expanded:  {stats['nodes_expanded']}\n")
        f.write(f"  Nodes Generated: {stats['nodes_created']}\n")
        f.write(f"Max Memory Used: {peakMem/1024:.0f} KB\n")
        f.write(f"  Time:            {time_elapsed:.3f}s\n")

        f.write("="*30 + "\n\n")

        state = GameState.LightsOutState(copy.deepcopy(initial_state.board))
        f.write("Initial Board:\n")
        for row in state.board:
            f.write(" ".join(str(c) for c in row) + "\n")
        f.write("\n")

        for i, (r, c) in enumerate(moves):
            state = state.move(r, c)
            f.write(f"Move {i+1}: ({r}, {c})\n")
            for row in state.board:
                f.write(" ".join(str(c) for c in row) + "\n")
            f.write("\n")




