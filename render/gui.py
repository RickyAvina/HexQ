import render.render_consts as Consts
import multiprocessing
import pygame


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
            print('here 3')
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

        self.rows = rows
        self.cols = cols
        self.grid = []
        self.exits = exits  # {(room, pos), ...}
        self.room_size = width//x_rooms
        self._init_rooms()
        self.agent = Agent(rows, cols)

    def _init_rooms(self):
        count = 0
        for r in range(Container.Y_ROOMS):
            row = []
            for c in range(Container.X_ROOMS):
                row.append(Room(c*self.room_size, r*self.room_size, self.rows, self.cols, count, self.room_size, self.exits, Consts.BLACK))
                count += 1
            self.grid.append(row)

    def move_agent(self, action):
        self.agent.move(action)

    def render(self, agent_loc):
        Container.WIN.fill(Consts.BLUE)

        for row in range(Container.Y_ROOMS):
            for col in range(Container.X_ROOMS):
                self.grid[row][col].render()
        self.agent.render(agent_loc)


container = None

def setup(width, height, rows, cols, x_rooms, y_rooms, target, exits, action_queue):
    p1 = multiprocessing.Process(target=start, args=(width, height, rows, cols, x_rooms, y_rooms, target, exits, action_queue))
    p1.start()

def start(width, height, rows, cols, x_rooms, y_rooms, target, exits, action_queue):
    global container
    WIN = pygame.display.set_mode((width, height))
    pygame.init()
    pygame.display.set_caption('Room Env')
    container = Container(WIN, width, height, rows, cols, x_rooms, y_rooms, target, exits)
    run = True
    clock = pygame.time.Clock()

    while run:
        clock.tick(Consts.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        try:
            if len(action_queue) > 0:
                #print("action queue: {}".format(len(action_queue)))
                pos = action_queue.pop(0)
                container.render(pos)
        except:
            print("action queue: {}".format(action_queue))
            pygame.quit()
        pygame.display.update()
    pygame.quit()

def change_title(title):
    pygame.display.set_caption(title)

def render_q_values(q_values):
    global container
    '''
    q values should be {Exit MDP:
        {mdp: {a: q, a', q'}}}
    '''
    print("here 1")
    if container is not None:
        print("here 2")
        for exit_mdp in q_values:
            for mdp in q_values[exit_mdp]:
                # find max Q
                assert mdp.level == 0
                max_action = max(q_values[exit_mdp][mdp], key=lambda k: q_values[exit_mdp][mdp].get(k))

                # find tile associated with mdp
                square = get_square(mdp.state_var)
                square.arrow = max_action

def get_arrow(arrow, x, y, w, h):
    if arrow == 0:  # left
        arrow_coords = [(0, h/2), (w/2, 0), (w/2, 3*h/8), (w, 3*h/8), (2, 5*h/8), (w/2, 5*h/8), (w/2, h)]
    elif arrow == 1:  # up
        arrow_coords = [(0, h/2), (w, h/2), (5*w/8, h/2), (5*w/8, h), (3*w/8, h), (3*w/8, h/2), (0, h/2)]
    elif arrow == 2:  # right
        arrow_coords = [(0, 3*h/8), (0, 5*h/8), (w/2, 0), (w/2, 3*h/8), (w, h/2), (w/2, h), (w/2, 5*h/8)]
    elif arrow == 3:  # down
        arrow_coords = [(0, h/2), (w/2, h), (w, h/2), (5*w/8, w/2), (5*w/8, 0), (3*w/8, 0), (3*w/8, w/2)]
    else:
        raise ValueError("arrow " + str(arrow) + " unrecognized")

    arrow_coords = tuple([(coord[0]+x, coord[1]+y) for coord in arrow_coords])
    return arrow_coords


def get_square(coord):
    global container
    room_row = coord[0] // container.cols
    room_col = coord[0] % container.cols
    room = container.grid[room_row][room_col]
    pos_row = coord[1] // container.cols
    pos_col = coord[1] % container.cols
    square = room.grid[pos_row][pos_col]
    return square

'''
def set_exits(exits):
    global container

    for exit in exits:
        room_row = exit[0] // container.cols
        room_col = exit[0] % container.cols
        room = container.grid[room_row][room_col]
        pos_row = exit[1] // container.cols
        pos_col = exit[1] % container.cols
        square = room.grid[pos_row][pos_col]
        square.color = Consts.GREEN  # change color of square to indicate exit
'''
