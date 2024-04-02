# Class_inheritances
Implement Python class inheritance with Pygames and AI for agent queue learning, 🐑💬 ➰ This is preparation for multi-agent queue learning and cascade machine learning algorithms. This note has inspiration from the course Python 3 Programming Specialization [ Link ]( https://coursera.org/share/ba047d1c5738f9bba3b08a5ac883569d ) about the Python class inheritance and communications.

### 🧸💬 Sample screen monitoring from Agent Queue class.

🐑💬 ➰ There are class variables and definition variables in class present of use or group by ```self``` and ```group``` for communication between definition methods, internal class, parent and child classes and internal class usage depends on variable types and configurations.

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/Class_inheritances/blob/main/player_location.png">
</p>

### 🧸💬 Library import.
```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Import libraries
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import sys
import pygame
import ple

import tensorflow as tf

from ple import PLE
from ple.games.snake import Snake as Snake_Game

from pygame.constants import K_a, K_s, K_d, K_w, K_h
```

### 🧸💬 Global project variables.
```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
actions = { "none_1": K_h, "left_1": K_a, "down_1": K_s, "right1": K_d, "up___1": K_w }
nb_frames = 100000000000
```

### 🧸💬 Create learning environment.
```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
game_console = Snake_Game(width=512, height=512, init_length=3);
p = PLE(game_console, fps=30, display_screen=True, reward_values={})
p.init()

obs = p.getScreenRGB()
```

### 🧸💬 Snake player class implementation
```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class Snake():

    def __init__ ( self, name ):
        self.name = name;
        self.snake_head_x = None;
        self.snake_head_y = None;
        self.food_x = None;
        self.food_y = None;
        self.snake_body = [];
        self.snake_body_pos = [];
        
        ###
        self.dist_x = None;
        self.dist_y = None;       
        self.previous_dist_x = None;
        self.previous_dist_y = None;
        
        return
    
    def __call__( self ):
        self.read_gamestate();
        
        return
    
    def read_gamestate( self ):
        gamestate = p.getGameState( );
        
        self.snake_head_x = gamestate["snake_head_x"];
        self.snake_head_y = gamestate["snake_head_y"];
        self.food_x = gamestate["food_x"];
        self.food_y = gamestate["food_y"];
        self.snake_body = gamestate["snake_body"];
        self.snake_body_pos = gamestate["snake_body_pos"];
        
        self.previous_dist_x = self.dist_x;
        self.previous_dist_y = self.dist_y;
        self.dist_x = self.snake_head_x - self.food_x;
        self.dist_y = self.snake_head_y - self.food_y;
        
        ###
        if not self.previous_dist_x :
            self.previous_dist_x = 0;
        if not self.dist_x :
            self.dist_x = 0;
        if not self.previous_dist_y :
            self.previous_dist_y = 0;
        if not self.dist_y :
            self.dist_y = 0;

        return
        
    def get_head_x( self ):
        
        return self.snake_head_x;
        
    def get_head_y( self ):
        
        return self.snake_head_y;
        
    def get_food_x( self ):
    
        return self.food_x;
        
    def get_food_y( self ):
    
        return self.food_y;
        
    def get_distance_x( self ):
    
        return self.dist_x;
        
    def get_distance_y( self ):
    
        return self.dist_y;
        
        
    def get_snakebody( self ):
    
        return self.snake_body;
        
    def get_snakebody_pos( self ):
    
        return self.snake_body_pos;
        
        
    def get_possibleactions( self ):

            
        # ...
    
        return K_h;
```

### 🧸💬 Agent Queue class implementation
```
class AgentQueue():

    def __init__ ( self, name ):
        self.name = name;
        self.reward = 0;
        self.step = 0;
        
        ###
        self.new_Snake = Snake( "Snake_01" );
    
        return
    
    def next_step( self, action ):
        self.reward = p.act( action )
        self.step = self.step + 1;
    
        return
        
    def game_over( self ):
        self.reward = 0;
        self.step = 0;
    
        return
        
    def read_gamestate( self ):

        distance_x = self.new_Snake.get_distance_x();
        distance_y = self.new_Snake.get_distance_y();
        
        print( f"distance x: { distance_x } distance y: { distance_y }" );

        return
```

### 🧸💬 Running tasks iterations.
```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Tasks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
new_snake = AgentQueue( "Snake_01" );

for i in range(nb_frames):

    if p.game_over():
        p.init();
        p.reset_game();
        new_snake.game_over();

    input("...")

    possible_actions = new_snake.new_Snake.get_possibleactions();
    
    print( [ x for (x, y) in list(actions.items()) if y == possible_actions ] )
    new_snake.next_step( possible_actions ); 
```
