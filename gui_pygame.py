import pygame

WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8

RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
FONT_SIZE = 20


FPS = 60
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Checkers')


class Square:
    def __init__(self, square_size, num=0, color=WHITE):
        self.num = num
        self.color = color
        self.square_size = square_size-2
        self.label = pygame.font.SysFont("monospace", FONT_SIZE).render(str(self.num), 1, (255, 0, 0))

    def render(self, win, x, y):
        pygame.draw.rect(win, self.color, (x+1, y+1, self.square_size, self.square_size))
        win.blit(self.label, (x+(self.square_size-FONT_SIZE)//2, y+(self.square_size-FONT_SIZE)//2))


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.square_size = WIDTH//cols
        self.grid = []
        self._init_grid()

    def _init_grid(self):
        count = 0
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                row.append(Square(self.square_size, count, WHITE))
                count += 1
            self.grid.append(row)

    def render(self, win):
        win.fill(BLACK)
        for row in range(self.rows):
            for col in range(self.cols):
                self.grid[row][col].render(win, col*self.square_size, row*self.square_size)


def main():
    run = True
    pygame.init()
    clock = pygame.time.Clock()
    grid = Grid(ROWS, COLS)

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
   
        grid.render(WIN)
        pygame.display.update()

    pygame.quit()


main()
