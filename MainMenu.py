import pygame
import sys



def drawMenu(screen, font, title_font, selected):
    screen.fill((20, 20, 40))

    # title
    title = title_font.render("Lights Out", True, (255, 215, 0))
    screen.blit(title, (screen.get_width()//2 - title.get_width()//2, 80))

    # options
    options = ["Play", "Algorithms"]
    for i, option in enumerate(options):
        color = (255, 215, 0) if i == selected else (180, 180, 180)
        bg    = (60, 60, 80)  if i == selected else (40, 40, 60)
        text  = font.render(option, True, color)
        rect  = pygame.Rect(screen.get_width()//2 - 120, 250 + i * 90, 240, 60)
        pygame.draw.rect(screen, bg, rect, border_radius=12)
        pygame.draw.rect(screen, color, rect, 2, border_radius=12)
        screen.blit(text, (rect.centerx - text.get_width()//2,
                           rect.centery - text.get_height()//2))

    hint = font.render("UP/DOWN   ENTER to confirm", True, (80, 80, 80))
    screen.blit(hint, (screen.get_width()//2 - hint.get_width()//2,
                       screen.get_height() - 50))
    pygame.display.flip()


def menu(screen, font, title_font):
    selected = 0
    clock    = pygame.time.Clock()

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
                    return selected   # 0 = Play, 1 = Algorithms
            if event.type == pygame.MOUSEBUTTONDOWN:
                for i in range(2):
                    rect = pygame.Rect(screen.get_width()//2 - 120, 250 + i * 90, 240, 60)
                    if rect.collidepoint(event.pos):
                        return i

        drawMenu(screen, font, title_font, selected)
        clock.tick(60)

def drawSizeMenu(screen, font, title_font, selected):
    screen.fill((20, 20, 40))

    # title
    title = title_font.render("Choose Board Size", True, (255, 215, 0))
    screen.blit(title, (screen.get_width()//2 - title.get_width()//2, 80))

    # options
    options = ["3x3", "4x4", "5x5"]
    for i, option in enumerate(options):
        color = (255, 215, 0) if i == selected else (180, 180, 180)
        bg    = (60, 60, 80)  if i == selected else (40, 40, 60)
        text  = font.render(option, True, color)
        rect  = pygame.Rect(screen.get_width()//2 - 120, 250 + i * 90, 240, 60)
        pygame.draw.rect(screen, bg, rect, border_radius=12)
        pygame.draw.rect(screen, color, rect, 2, border_radius=12)
        screen.blit(text, (rect.centerx - text.get_width()//2,
                           rect.centery - text.get_height()//2))

    hint = font.render("UP/DOWN   ENTER to confirm", True, (80, 80, 80))
    screen.blit(hint, (screen.get_width()//2 - hint.get_width()//2,
                       screen.get_height() - 50))
    pygame.display.flip()
    return

def chooseSizeMenu(screen, font, title_font):
    selected = 0
    clock    = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return -1
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % 3
                if event.key == pygame.K_DOWN:
                    selected = (selected + 1) % 3
                
                if event.key == pygame.K_RETURN:
                    return selected+3
            if event.type == pygame.MOUSEBUTTONDOWN:
                for i in range(3):
                    rect = pygame.Rect(screen.get_width()//2 - 120, 250 + i * 90, 240, 60)
                    if rect.collidepoint(event.pos):
                        return i

        drawSizeMenu(screen, font, title_font, selected)
        clock.tick(60)