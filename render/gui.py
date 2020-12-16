import pygame
import render.render_consts as Consts
import threading


class Agent:
    def __init__(self, room, pos, rows, cols, color=Consts.BLUE):
        self.room = room
        self.pos = pos
        self.rows = rows
        self.cols = cols
        self.color = color
        self.radius = Container.WIDTH//Container.X_ROOMS//Container.COLS//4

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

        assert(self.pos >= 0 and self.pos < self.rows*self.cols, self.pos)

    def render(self):
        pygame.draw.circle(Container.WIN, self.color, Container.grid_dict[(self.room, self.pos)], self.radius, 0)


class Square:
    def __init__(self, x, y, num, square_size, color=Consts.WHITE):
        self.num = num
        self.color = color
        self.square_size = square_size-Consts.BORDER_SIZE*2
        self.label = pygame.font.SysFont("monospace", Consts.FONT_SIZE).render(str(self.num), 1, (255, 0, 0))
        self.x = x
        self.y = y

    def render(self):
        pygame.draw.rect(Container.WIN, self.color, (self.x+Consts.BORDER_SIZE, self.y+Consts.BORDER_SIZE, self.square_size, self.square_size))
        Container.WIN.blit(self.label, (self.x+Consts.BORDER_SIZE*2.5+(self.square_size-Consts.FONT_SIZE)//2, self.y+Consts.BORDER_SIZE+(self.square_size-Consts.FONT_SIZE)//2))


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
                Container.grid_dict[(self.num, count)] = (x+self.square_size//(self.cols*2), y+self.square_size//(self.rows*2))
                row.append(Square(x, y, count, self.square_size//self.cols, Consts.GREEN if (self.num, count) in self.exits else Consts.WHITE))
                count += 1
            self.grid.append(row)

    def render(self):
        pygame.draw.rect(Container.WIN, self.color, (self.x+Consts.BORDER_SIZE, self.y+Consts.BORDER_SIZE, self.square_size-Consts.BORDER_SIZE*2, self.square_size-4))

        for row in range(self.rows):
            for col in range(self.cols):
                self.grid[row][col].render()


class Container:

    def __init__(self, win, width, height, rows, cols, x_rooms, y_rooms, exits):
        Container.WIDTH = width
        Container.HEIGHT = height
        Container.COLS = cols
        Container.ROWS = rows
        Container.X_ROOMS = x_rooms
        Container.Y_ROOMS = y_rooms
        Container.WIN = win
        Container.grid_dict = {}

        self.rows = rows
        self.cols = cols
        self.grid = []
        self.exits = exits  # {(room, pos), ...}
        self.room_size = width//x_rooms
        self._init_rooms()
        self.agent = Agent(0, 0, rows, cols)

    def _init_rooms(self):
        count = 0
        for r in range(Container.Y_ROOMS):
            row = []
            for c in range(Container.X_ROOMS):
                row.append(Room(c*self.room_size, r*self.room_size, self.rows, self.cols, count, self.room_size, Consts.BLACK, self.exits))
                count += 1
            self.grid.append(row)

    def move_agent(self, action):
        self.agent.move(action)

    def render(self):
        Container.WIN.fill(Consts.BLUE)

        for row in range(Container.Y_ROOMS):
            for col in range(Container.X_ROOMS):
                self.grid[row][col].render()
        self.agent.render()


def setup(width, height, rows, cols, x_rooms, y_rooms, exits):
    global container
    WIN = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Room Env')
    pygame.init()
    clock = pygame.time.Clock()
    container = Container(WIN, width, height, rows, cols, x_rooms, y_rooms, exits)
    run = True


'''    while run:
        clock.tick(Consts.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        container.render()
        pygame.display.update()

    pygame.quit()
'''

def take_action(action):
    container.move_agent(action)

def render():
    container.render()
    pygame.display.update()


'''
if __name__ == "__main__":
    exits = {(0, 14), (0, 22), (1, 10), (1, 22), (2, 2), (2, 14), (3, 2), (3, 10)}
    setup(600, 600, 5, 5, 2, 2, exits)
'''
