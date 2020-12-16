import pygame

WIDTH, HEIGHT = 600, 600
ROWS, COLS = 5, 5
X_ROOMS, Y_ROOMS = 2, 2

RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
FONT_SIZE = 20
BORDER_SIZE = 2

FPS = 60
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Checkers')


grid_dict = {}


class Agent:
    def __init__(self, room, pos, rows, cols, color=BLUE):
        self.room = room
        self.pos = pos
        self.rows = rows
        self.cols = cols
        self.color = color
        self.radius = WIDTH//X_ROOMS//COLS//4

    def move(self, action):
        if (action == 0):  # left
            self.pos -= 1
        elif (action == 1):  # right
            self.pos += 1
        elif (action == 2):  # up
            self.pos -= self.cols
        elif (action == 3):  # down
            self.pos += self.cols
        else:
            raise ValueError("Incorrect Value")

        assert(self.pos >= 0 and self.pos < self.rows*self.cols)

    def render(self, win):
        pygame.draw.circle(win, self.color, grid_dict[(self.room, self.pos)], self.radius, 0)


class Square:
    def __init__(self, x, y, num, square_size, color=WHITE):
        self.num = num
        self.color = color
        self.square_size = square_size-BORDER_SIZE*2
        self.label = pygame.font.SysFont("monospace", FONT_SIZE).render(str(self.num), 1, (255, 0, 0))
        self.x = x
        self.y = y

    def render(self, win):
        pygame.draw.rect(win, self.color, (self.x+BORDER_SIZE, self.y+BORDER_SIZE, self.square_size, self.square_size))
        win.blit(self.label, (self.x+BORDER_SIZE*2.5+(self.square_size-FONT_SIZE)//2, self.y+BORDER_SIZE+(self.square_size-FONT_SIZE)//2))


class Room:
    def __init__(self, x, y, rows, cols, num, square_size, color, exits):
        self.num = num
        self.rows = rows
        self.cols = cols
        self.x = x
        self.y = y
        self.color = color
        self.square_size = square_size
        self.exits = exits
        self.grid = []
        self._init_grid()

    def __repr__(self):
        return "Room {} ({}, {})".format(self.num, self.x, self.y)

    def _init_grid(self):
        count = 0
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                x = self.x+c*self.square_size//self.cols
                y = self.y+r*self.square_size//self.rows
                grid_dict[(self.num, count)] = (x+self.square_size//(self.cols*2), y+self.square_size//(self.rows*2))
                row.append(Square(x, y, count, self.square_size//self.cols, GREEN if (self.num, count) in self.exits else WHITE))
                count += 1
            self.grid.append(row)

    def render(self, win):
        pygame.draw.rect(win, self.color, (self.x+BORDER_SIZE, self.y+BORDER_SIZE, self.square_size-BORDER_SIZE*2, self.square_size-4))

        for row in range(self.rows):
            for col in range(self.cols):
                self.grid[row][col].render(win)


class Container:
    def __init__(self, rows, cols, x_rooms, y_rooms, exits):
        self.rows = rows
        self.cols = cols
        self.x_rooms = x_rooms
        self.y_rooms = y_rooms
        self.grid = []
        self.exits = exits  # {(room, pos), ...}
        self.room_size = WIDTH//x_rooms
        self._init_rooms()
        self.agent = Agent(0, 0, rows, cols)

    def _init_rooms(self):
        count = 0
        for r in range(self.y_rooms):
            row = []
            for c in range(self.x_rooms):
                row.append(Room(c*self.room_size, r*self.room_size, self.rows, self.cols, count, self.room_size, BLACK, self.exits))
                count += 1
            self.grid.append(row)

    def move_agent(self, action):
        self.agent.move(action)

    def render(self, win):
        win.fill(BLUE)
        for row in range(self.y_rooms):
            for col in range(self.x_rooms):
                self.grid[row][col].render(win)
        self.agent.render(win)


def main():
    run = True
    pygame.init()
    clock = pygame.time.Clock()
    exits = {(0, 14), (0, 22), (1, 10), (1, 22), (2, 2), (2, 14), (3, 2), (3, 10)}
    container = Container(ROWS, COLS, X_ROOMS, Y_ROOMS, exits)

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        container.render(WIN)
        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
