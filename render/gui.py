import render.render_consts as Consts
import multiprocessing
import pygame
import sys
import enum


class Agent:
    def __init__(self, rows, cols, color=Consts.BLUE):
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

        assert self.pos >= 0 and self.pos < self.rows*self.cols, self.pos

    def render(self, agent_loc):
        pygame.draw.circle(Container.WIN, self.color, Container.grid_dict[agent_loc], self.radius, 0)


class Square:
    def __init__(self, x, y, num, square_size, color=Consts.WHITE):
        self.num = num
        self.color = color
        self.square_size = square_size-Consts.BORDER_SIZE*2
        self.label = pygame.font.SysFont("monospace", Consts.FONT_SIZE).render(str(self.num), 1, (255, 0, 0))
        self.x = x
        self.y = y
        self.arrow = None

    def render(self):
        pygame.draw.rect(Container.WIN, self.color, (self.x+Consts.BORDER_SIZE, self.y+Consts.BORDER_SIZE, self.square_size, self.square_size))
        Container.WIN.blit(self.label, (self.x+Consts.BORDER_SIZE*2.5+(self.square_size-Consts.FONT_SIZE)//2, self.y+Consts.BORDER_SIZE+(self.square_size-Consts.FONT_SIZE)//2))
        if self.arrow is not None:
            arrow_coords = get_arrow(self.arrow, self.x, self.y, self.square_size, self.square_size)
            pygame.draw.polygon(Container.WIN, Consts.BLACK, arrow_coords)


class Room:
    def __init__(self, x, y, rows, cols, num, square_size, exits, color=Consts.BLACK):
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
                Container.grid_dict[(count, self.num)] = (x+self.square_size//(self.cols*2), y+self.square_size//(self.rows*2))
                if (count, self.num) in self.exits:
                    square_color = Consts.GREEN
                else:
                    if len(Container.target) == 1:  # room
                        square_color = Consts.YELLOW if self.num == Container.target[0] else Consts.WHITE
                    else:  # pos in room
                        square_color = Consts.YELLOW if (count, self.num) == Container.target else Consts.WHITE
                row.append(Square(x, y, count, self.square_size//self.cols, square_color))
                count += 1
            self.grid.append(row)

    def render(self):
        pygame.draw.rect(Container.WIN, self.color, (self.x+Consts.BORDER_SIZE, self.y+Consts.BORDER_SIZE, self.square_size-Consts.BORDER_SIZE*2, self.square_size-4))

        for row in range(self.rows):
            for col in range(self.cols):
                self.grid[row][col].render()


class Container:
    def __init__(self, win, width, height, rows, cols, x_rooms, y_rooms, target, exits):
        Container.WIDTH = width
        Container.HEIGHT = height
        Container.COLS = cols
        Container.ROWS = rows
        Container.X_ROOMS = x_rooms
        Container.Y_ROOMS = y_rooms
        Container.target = target
        Container.WIN = win
        Container.grid_dict = {}

        self.x_rooms = x_rooms
        self.y_rooms = y_rooms
        self.rows = rows
        self.cols = cols
        self.grid = []
        self.exits = exits  # {(room, pos), ...}
        self.room_size = width//x_rooms
        self._init_rooms()

    def _init_rooms(self):
        count = 0
        for r in range(Container.Y_ROOMS):
            row = []
            for c in range(Container.X_ROOMS):
                row.append(Room(c*self.room_size, r*self.room_size, self.rows, self.cols, count, self.room_size, self.exits, Consts.BLACK))
                count += 1
            self.grid.append(row)

    def render(self):
        Container.WIN.fill(Consts.BLUE)

        for row in range(Container.Y_ROOMS):
            for col in range(Container.X_ROOMS):
                self.grid[row][col].render()


def get_arrow(arrow, x, y, w, h):
    if arrow == 0:  # left
        arrow_coords = [(0, h/2), (w/2, 0), (w/2, 3*h/8), (w, 3*h/8), (w, 5*h/8), (w/2, 5*h/8), (w/2, h)]
    elif arrow == 1:  # right
        arrow_coords = [(0, 5*h/8), (w/2, 5*h/8), (w/2, 0), (w, h/2), (w/2, h), (w/2, 3*h/8), (0, 3*h/8)]
    elif arrow == 2:  # up
        arrow_coords = [(0, h/2), (w/2, 0), (w, h/2), (5*w/8, h/2), (5*w/8, h), (3*w/8, h), (3*w/8, h/2)]
    elif arrow == 3:  # down
        arrow_coords = [(0, h/2), (w/2, h), (w, h/2), (5*w/8, w/2), (5*w/8, 0), (3*w/8, 0), (3*w/8, w/2)]
    else:
        raise ValueError("arrow " + str(arrow) + " unrecognized")

    arrow_coords = tuple([(coord[0]+x, coord[1]+y) for coord in arrow_coords])
    return arrow_coords


def get_square(container, coord):
    room_row = coord[1] // container.x_rooms
    room_col = coord[1] % container.y_rooms
    room = container.grid[room_row][room_col]
    pos_row = coord[0] // container.cols
    pos_col = coord[0] % container.cols
    square = room.grid[pos_row][pos_col]
    return square


class EventType(enum.Enum):
    QUIT = 0
    QVAL = 1

class Event():
    def __init__(self, kind, data):
        self.kind = kind
        self.data = data

    def __repr__(self):
        return "kind: {}\ndata: {}".format(self.kind, self.data)

class GUI():
    def __init__(self, width, height, rows, cols, x_rooms, y_rooms, target, exits, queue):
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols
        self.x_rooms = x_rooms
        self.y_rooms = y_rooms
        self.target = target
        self.exits = exits
        self.queue = queue
        self.done = False
        self.process = multiprocessing.Process(target=self.start, args=(queue,))
        self.process.start()

    def start(self, queue):
        self.WIN = pygame.display.set_mode((self.width, self.height))
        pygame.init()
        pygame.display.set_caption('Room Env')
        self.container = Container(self.WIN, self.width, self.height, self.rows, self.cols, self.x_rooms, self.y_rooms, self.target, self.exits)
        self.agent = Agent(self.rows, self.cols)

        self.run = True
        clock = pygame.time.Clock()

        while self.run:
            clock.tick(Consts.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                    break

            if not queue.empty():
                event = queue.get()

                if event.kind == EventType.QVAL:
                    exit_square = self.get_square(event.data['exit'])
                    exit_square.color = Consts.RED
                    arrow_squares = self.add_arrows(event.data['mdps'])
                    self.container.render()
                    pygame.display.update()

                    skip = False
                    while not skip:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.run = False
                                skip = True
                                break
                            elif event.type == pygame.MOUSEBUTTONDOWN:
                                skip = True
                                break

                    # reset squares
                    exit_square.color = Consts.GREEN
                    for square in arrow_squares:
                        square.arrow = None

        pygame.quit()

    def get_square(self, coord):
        room_row = coord[1] // self.container.x_rooms
        room_col = coord[1] % self.container.y_rooms
        room = self.container.grid[room_row][room_col]
        pos_row = coord[0] // self.container.cols
        pos_col = coord[0] % self.container.cols
        square = room.grid[pos_row][pos_col]
        return square

    def render_q_values(self, arrow_list):
        for arrows in arrow_list:
            for exit in arrows:
                self.queue.put(Event(EventType.QVAL, {'exit': exit, 'mdps': arrows[exit]}))

    def add_arrows(self, q_values):
        '''
        q_values are in format {state_var, arrow}
        '''
        squares = []
        for state in q_values:
            square = self.get_square(state)
            square.arrow = q_values[state]
            squares.append(square)
        return squares

def wait_for_click():
    skip = False
    while not skip:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                skip = True
                sys.exit()
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                skip = True
                break
