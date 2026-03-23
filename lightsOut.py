import pygame
import sys


class LightsOutState:

    def __init__(self, board):
        self.board = board

    def winningState(self):
        return self.board == [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    def move(self,row,column):

        self.board[row][column] ^= 1

        if(row > 0):
            self.board[row-1][column] ^= 1
        
        if(column > 0):
            self.board[row][column-1] ^= 1
        

        if(column < len(self.board)-1):
            self.board[row][column+1] ^= 1

        if(row < len(self.board)-1):
            self.board[row+1][column] ^= 1

    def printState(self):
        for i in self.board:
            print(i)

def drawBoard(screen, board, on_img, off_img):
    LIGHT = 110
    ON_COLOR  = (255,215,0)
    OFF_COLOR = (0,0,0)
    BG_COLOR = (200,200,200)
    size = len(board)

    screen.fill(BG_COLOR)

    for i in range(size):
        for j in range(size):
            x = 20 + (LIGHT+10) * i
            y = 20 + (LIGHT+10) * j
            if board[i][j] == 0:
                screen.blit(on_img, (x, y))
            else:
                screen.blit(off_img, (x, y))
    
    pygame.display.flip()






def play():

    GAP       = 6
    PADDING   = 30

    state = LightsOutState([[1,1,0,0],
                    [1,0,0,0],
                    [0,0,0,1],
                    [0,0,1,1]])
    
    size = len(state.board)
    win_px = PADDING * 2 + size * 110 + (size - 1) * GAP
    pygame.init()
    on_img = pygame.image.load("sprites/lightson.jpg")
    off_img = pygame.image.load("sprites/lightsoff.jpg")
    on_img = pygame.transform.scale(on_img, (110, 110))
    off_img = pygame.transform.scale(off_img, (110, 110))
    screen = pygame.display.set_mode((win_px, win_px))
    clock = pygame.time.Clock()

    while not(state.winningState()):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                row = (mx - 20) // (110 + 10)
                col = (my - 20) // (110 + 10)
                if 0 <= row < size and 0 <= col < size:
                    state.move(row, col)

        drawBoard(screen, state.board, on_img, off_img)
        clock.tick(60)
    
    print("You won")


play()