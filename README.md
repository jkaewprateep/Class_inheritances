# Class_inheritances
Implement Python class inheritance with Pygames and AI for agent queue learning, 🐑💬 ➰ This is preparation for multi-agent queue learning and cascade machine learning algorithms. This note has inspiration from the course Python 3 Programming Specialization [ Link ]( https://coursera.org/share/ba047d1c5738f9bba3b08a5ac883569d ) about the Python class inheritance and communications.

### 🧸💬 Sample screen monitoring from Agent Queue class.

🐑💬 ➰ There are class variables and definition variables in the class present of use or group by ```self``` and ```group``` for communication between definition methods, internal class, parent and child classes, and internal class usage depending on variable types and configurations. </br>
👧💬 🎈 Internal class variable defined at the class level and access of this class variable required interaction with ```user``` and ```group``` by ```method``` or ```references```. There is no reference now to create an instance of a class with ```<objectId>``` and access to the same values they used in Python because Python is translation code programming </br>
🐑💬 ➰ I remembered you can create of a new instance with the same ```<objectId>``` in C that someone is called modern memory hacks but it is now not compatible because of the variable memory assignment method. A programmer who are working with C or programming level knows that ```<objectId>``` is only reference and you can create of a fake message request with the same ```<objectId>``` in communication and now stageless communication prevents them from reuse of the ```<object variables>```. You can create APIs and remove of create messages and required communication methods to prevent third-party applications from reusing the same message but they can do it by communication networks level anyway. </br>

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/Class_inheritances/blob/main/player_location.png">
</p>

### 🧸💬 Library import.

🐑💬 ➰ Refer to class or directory structure by the import reference commands, we have control of changes of class method files in the library by ```summation value``` and ```external library```. </br>
🦭💬 ```Complied``` codes or ```backup``` running code is a good method when ```separation of execution path``` from method and data are good practice for working with third-party applications or someone calls sensitive data. </br>
🐐💬 In design the Pygame environment constant is similar to console games players and communication of controller and games player device controller accepted of both ```decimal numeric values``` and ```heximal numeric values``` when some games player modification can use ```binary numeric values``` communications but the communication types is start from initial message. Do not shift the jumper otherwise, you may return to ```A-B``` mode as default. ( It is the controller at last button A and B is working or most contrast control is working - can explain in electronics communication and neurons sciences ) </br>

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Import libraries
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import sys                                             # 🧸💬 import library # 🧸💬 os.linesep is useful for multi-culture communications.
import pygame                                          # 🧸💬 import pygame library, drawing, structs, variables, and many of sample games environments.
import ple                                             # 🧸💬 import Python learning environment, games wrapper, and communications method.

import tensorflow as tf                                # 🧸💬 import tensorflow machine learning library.

from ple import PLE                                    # 🧸💬 import Python learning environment.
from ple.games.snake import Snake as Snake_Game        # 🧸💬 import Sanke game from ple.

from pygame.constants import K_a, K_s, K_d, K_w, K_h   # 🧸💬 import of console constant variables.
```

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/Class_inheritances/blob/main/pulse%20code%20modulation.png">
</p>

👧💬 🎈 ```Pluse code modulation``` or ```time-shifted channel``` was developed by Nintendo and distributed during our youth time, long years after supporting of the government they launched the games players ```Nintendo PlayStation``` . </br>
🐑💬 ➰ This may be the event called ```time capsules ``` because of the ability to recover communication messages and transfer rates vary by device negotiation compatibilities. In the ```time capsules``` can store validation matrixes and summary values of the designed communication channel's message you can categorize and summarize computer players for their actions responses and feedback as in console games players save CPU time process when the console box use compatibilities specification for decrypted communication message and response. Of course, random variances create a of variety actions and possibilities by the actions played. </br>

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
