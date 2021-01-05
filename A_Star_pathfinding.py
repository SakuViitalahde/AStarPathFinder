import pygame
import math
from queue import PriorityQueue

WIDTH = 1000
HEIGHT = 1000
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Path Finding")

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUISE = (64, 224, 208)


class Spot:
    """
    This is one node in grid.
    """

    def __init__(self, row, col, width, height, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.height = height
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == PURPLE

    def reset(self):
        self.color = WHITE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_start(self):
        self.color = ORANGE

    def make_end(self):
        self.color = PURPLE

    def make_path(self):
        self.color = TURQUISE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # Under
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): #Up
            self.neighbors.append(grid[self.row - 1][self.col])        

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # Right
            self.neighbors.append(grid[self.row][self.col + 1])       

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # Left
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


def h(p1, p2):
    """
    Distance between 2 points p1 and p2.

    We use manhattan distance with means shortest L between spots.
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current.make_path()
        current = came_from[current]
        draw()


def algorithm(draw, grid, start, end):
    """
        A* Algorythm
    """
    count = 0
    open_set = PriorityQueue() # We neeed priorityqueue to get always on lowest node possible fast.
    open_set.put((0, count, start)) # Put start node in open set
    came_from = {} # dict to keep where we moved to where
    g_score = {spot:float("inf") for row in grid for spot in row} # Create G score dict got every spot, start value is infinite (G score is distance to spot from start)
    g_score[start] = 0 #reset start spot value to 0
    f_score = {spot:float("inf") for row in grid for spot in row} # Create F score dict got every spot, start value is infinite (F score is gessed score from spot to spot)
    f_score[start] = h(start.get_pos(), end.get_pos()) # Set f score for start, its manhanttan distance from start to end 

    open_set_hash = {start} # create hash to know what is inside prioque

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        current = open_set.get()[2] # read open sets first items spot object
        open_set_hash.remove(current) # remote is from hash because it is not anymore in open_set

        if current == end: # If this is true we found shortest line
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors: #If current node have neighbors, we loop them.
            temp_g_score = g_score[current] + 1 # G score is calculeted, in this case every distance is 1 so we can just add 1

            if temp_g_score < g_score[neighbor]: # if g score to neighbor is better than found before.
                came_from[neighbor] = current # we unpdate where we came from
                g_score[neighbor] = temp_g_score # Update new g score value
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos()) # Update new f score value
                if neighbor not in open_set_hash: # if neighbor is not already in hash
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor)) # put it in open set
                    open_set_hash.add(neighbor) # add it in hash
                    neighbor.make_open() # make it show in correct color in draw

        draw() # Draw the system

        if current != start:
            current.make_closed()
            
    return False

def make_grid(rows, width):
    """
    Make grid i is rows and j is columns.
    Gap is width and height of spot.
    """
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, gap, rows)
            grid[i].append(spot)
    return grid

def draw_grid(window, rows, width, height):
    """
    Draw net to display grid.
    """
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(window, GREY, (0, i * gap), (width, i * gap))

    for j in range(rows):
        pygame.draw.line(window, GREY, (j * gap, 0), (j * gap, height))

def draw(window, grid, rows, width, height):
    window.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(window)

    draw_grid(window, rows, width, height)
    pygame.display.update()

def get_clicked_position(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

def main(win, width, height):
    ROWS = 50
    
    grid = make_grid(ROWS, width)

    start = None
    end = None

    search_started = False
    while True:
        draw(win, grid, ROWS, width, height)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            
            if search_started:
                continue
            
            if pygame.mouse.get_pressed()[0]: # Left
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_position(pos, ROWS, width)
                spot = grid[row][col]

                if not start and spot != end:
                    start = spot
                    spot.make_start()
                elif not end and spot != start:
                    end = spot
                    spot.make_end()
                elif spot != end and spot != start:
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]: # Right
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_position(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not search_started:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    
                    algorithm(lambda: draw(win, grid, ROWS, width, height), grid, start, end)
            
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS,width)


main(WIN,WIDTH,HEIGHT)