import pygame

WIDTH, HEIGHT = 800, 800
ROWS, COLS = 4, 4

RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
FONT_SIZE = 20
BORDER_SIZE = 2

FPS = 60
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Checkers')


class Square:
    def __init__(self, square_size, x, y, num=0, color=WHITE):
        self.num = num
        self.color = color
        self.square_size = square_size-BORDER_SIZE*2
        self.label = pygame.font.SysFont("monospace", FONT_SIZE).render(str(self.num), 1, (255, 0, 0))
        self.x = x
        self.y = y

    def render(self, win):
        pygame.draw.rect(win, self.color, (self.x+BORDER_SIZE, self.y+BORDER_SIZE, self.square_size, self.square_size))
        win.blit(self.label, (self.x+(self.square_size-FONT_SIZE)//2, self.y+(self.square_size-FONT_SIZE)//2))


class Room:
    def __init__(self, num, rows, cols, x, y, square_size):
        self.num = num
        self.rows = rows
        self.cols = cols
        self.x = x
        self.y = y
        self.square_size = square_size
        self.grid = []
        self._init_grid()

    def __repr__(self):
        return "Room {} ({}, {})".format(self.num, self.x, self.y)

    def _init_grid(self):
        count = 0
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                row.append(Square(self.square_size//self.cols, self.x+c*self.square_size//self.cols,
                                self.y+r*self.square_size//self.rows, count, WHITE))
                count += 1
            self.grid.append(row)

    def render(self, win):
        pygame.draw.rect(win, BLACK, (self.x+BORDER_SIZE, self.y+BORDER_SIZE, self.square_size-BORDER_SIZE*2, self.square_size-4))

        for row in range(self.rows):
            for col in range(self.cols):
                self.grid[row][col].render(win)


class Container:
    def __init__(self, rows, cols, x_rooms, y_rooms):
        self.rows = rows
        self.cols = cols
        self.x_rooms = x_rooms
        self.y_rooms = y_rooms
        self.grid = []
        self.room_size = WIDTH//x_rooms
        self._init_rooms()

    def _init_rooms(self):
        count = 0
        for r in range(self.y_rooms):
            row = []
            for c in range(self.x_rooms):
                row.append(Room(count, self.rows, self.cols, c*self.room_size, r*self.room_size, self.room_size))
                count += 1
            self.grid.append(row)
    
    def render(self, win):
        win.fill(BLUE)
        for row in range(self.y_rooms):
            for col in range(self.x_rooms):
                self.grid[row][col].render(win)

def main():
    run = True
    pygame.init()
    clock = pygame.time.Clock()
    container = Container(ROWS, COLS, 2, 2)

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
   
        container.render(WIN)
        pygame.display.update()

    pygame.quit()


main()
