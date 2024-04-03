# Class_inheritances
Implement Python class inheritance with Pygames and AI for agent queue learning, 🐑💬 ➰ This is preparation for multi-agent queue learning and cascade machine learning algorithms. This note has inspiration from the course Python 3 Programming Specialization [ Link ]( https://coursera.org/share/ba047d1c5738f9bba3b08a5ac883569d ) about the Python class inheritance and communications.

### 🧸💬 Sample screen monitoring from Agent Queue class.

🐑💬 ➰ There are class variables and definition variables in the class present of use or group by ```self``` and ```group``` for communication between definition methods, internal class, parent and child classes, and internal class usage depending on variable types and configurations. </br>
👧💬 🎈 Internal class variable defined at the class level and access of this class variable required interaction with ```user``` and ```group``` by ```method``` or ```references```. There is no reference now to create an instance of a class with ```<objectId>``` and access to the same values they used in Python because Python is translation code programming </br>
🐑💬 ➰ I remembered you can create of a new instance with the same ```<objectId>``` in C that someone is called modern memory hacks but it is now not compatible because of the variable memory assignment method. A programmer who are working with C or programming level knows that ```<objectId>``` is only reference and you can create of a fake message request with the same ```<objectId>``` in communication and now stageless communication prevents them from reuse of the ```<object variables>```. You can create APIs and remove of create messages and required communication methods to prevent third-party applications from reusing the same message but they can do it by communication networks level anyway. </br>

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/Class_inheritances/blob/main/player_location.png"> </br>
    <b> Processor run debugging </b>
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
    <img width="40%" src="https://github.com/jkaewprateep/Class_inheritances/blob/main/pulse%20code%20modulation.png"> </br>
    <b> Pluse codes modulation </b>
</p>

👧💬 🎈 ```Pluse code modulation``` or ```time-shifted channel``` was developed by Nintendo and distributed during our youth time, long years after supporting of the government they launched the games players ```Nintendo PlayStation``` . </br>
🐑💬 ➰ This may be the event called ```time capsules ``` because of the ability to recover communication messages and transfer rates vary by device negotiation compatibilities. In the ```time capsules``` can store validation matrixes and summary values of the designed communication channel's message you can categorize and summarize computer players for their actions responses and feedback as in console games players save CPU time process when the console box use compatibilities specification for decrypted communication message and response. Of course, random variances create a of variety actions and possibilities by the actions played. </br>

### 🧸💬 Global project variables.

🧸💬 We like to create constructs to manage variables and transform value, looking into micro-controller devices and PLC application programming we would like to define the response variables and interfaces before constructing a method for communication because of the same behavior inherited from Nintendo. </br>
🐑💬 ➰ The number of frames is only a number it does not require a large value since the communication is still online and happens. </br>

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
actions = { "none_1": K_h, "left_1": K_a, "down_1": K_s, "right1": K_d, "up___1": K_w }
nb_frames = 100000000000
```

### 🧸💬 Create a learning environment.

🐨🎁🎵🎶 In a learning environment is an application with the construct of possible variables they are setup sample the Half-life games for modern environment simulation games and modification games are defined construct variables from the learning environment application games. They are also called ```learning environments``` . 👤🗯️ ```ไปเรียนแม่‼️```

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
game_console = Snake_Game(width=512, height=512, init_length=3);            # 🧸💬 Create instance of class environment with initial values.
p = PLE(game_console, fps=30, display_screen=True, reward_values={})        # 🧸💬 Reflecter, there are many type of reflectors including
                                                                            # 🧸💬 screen environment, matrix environment, linear and
                                                                            # 🧸💬 logarithms and printter communications.
p.init()                                                                    # 🧸💬 Initialize

obs = p.getScreenRGB()                                                      # 🧸💬 Sample of screen arrays return value collection.
```

### 🧸💬 Snake player class implementation

🐐💬 First we need to know the favorites expectation method ```___init___``` and ```___call___``` we defined ```___init___``` for initialize class variables and setup class running environment when creating the class and initial. In some programs, you can create a Python class without initial as you ```copy``` . </br>
💃( 👩‍🏫 )💬 Some applications ```seeker``` find the ```invoker``` step to by pass for application validation, the ```__call___``` definition function and ```___init___``` definition function is created into difference place and working continuously as a sequence. Once a ```seeker``` can access the class method initialization and guess of the ```next method in rows``` and ```multiplication variables``` but they cannot skip of the ```___call___``` definition method without modify the games function. </br>
🐐💬 I can randomly place the same variable name somewhere and forget, the ```seeker``` can invoke of the target function by the ```function invoker``` but they cannot modify the values since they are the same variable names. </br>
👧💬 🎈 The games cheaters do not have class they do not know how to types ```self``` 💃( 👩‍🏫 )💬 🛍️👠💄 . </br>

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/Class_inheritances/blob/main/asian-nerd-girl.jpg"> </br>
    <b> Picture from the Internet </b>
</p>

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class Snake():                                               # 🧸💬 Create class construction

    def __init__ ( self, name ):                             # 🧸💬 Class initialization function definition
        self.name = name;                                    # 🧸💬 Class variables definition
        self.snake_head_x = None;                            # 🧸💬 Class variables definition
        self.snake_head_y = None;                            # 🧸💬 Class variables definition
        self.food_x = None;                                  # 🧸💬 Class variables definition
        self.food_y = None;                                  # 🧸💬 Class variables definition
        self.snake_body = [];                                # 🧸💬 Class variables definition
        self.snake_body_pos = [];                            # 🧸💬 Class variables definition

        ###
        self.dist_x = None;                                  # 🧸💬 Class variables definition
        self.dist_y = None;                                  # 🧸💬 Class variables definition
        self.previous_dist_x = None;                         # 🧸💬 Class variables definition
        self.previous_dist_y = None;                         # 🧸💬 Class variables definition
        
        return
    
    def __call__( self ):                                    # 🧸💬 Expectation class definition function invoke every time call action
        self.read_gamestate();                               # 🧸💬 Call update variables method
        
        return
    
    def read_gamestate( self ):                              # 🧸💬 Update variables method
        gamestate = p.getGameState( );                       # 🧸💬 Read console or environment variable output
        
        self.snake_head_x = gamestate["snake_head_x"];       # 🧸💬 Saved target variable status
        self.snake_head_y = gamestate["snake_head_y"];       # 🧸💬 Saved target variable status
        self.food_x = gamestate["food_x"];                   # 🧸💬 Saved target variable status
        self.food_y = gamestate["food_y"];                   # 🧸💬 Saved target variable status
        self.snake_body = gamestate["snake_body"];           # 🧸💬 Saved target variable status
        self.snake_body_pos = gamestate["snake_body_pos"];   # 🧸💬 Saved target variable status
        
        self.previous_dist_x = self.dist_x;                  # 🧸💬 Saved target variable status
        self.previous_dist_y = self.dist_y;                  # 🧸💬 Saved target variable status
        self.dist_x = self.snake_head_x - self.food_x;       # 🧸💬 Saved target variable status
        self.dist_y = self.snake_head_y - self.food_y;       # 🧸💬 Saved target variable status
        
        ###
        if not self.previous_dist_x :                        # 🧸💬 Error preventing stage
            self.previous_dist_x = 0;                        # 🧸💬 Error preventing stage
        if not self.dist_x :                                 # 🧸💬 Error preventing stage
            self.dist_x = 0;                                 # 🧸💬 Error preventing stage
        if not self.previous_dist_y :                        # 🧸💬 Error preventing stage
            self.previous_dist_y = 0;                        # 🧸💬 Error preventing stage
        if not self.dist_y :                                 # 🧸💬 Error preventing stage
            self.dist_y = 0;                                 # 🧸💬 Error preventing stage

        return
        
    def get_head_x( self ):                                  # 🧸💬 get_head_x function definition
        
        return self.snake_head_x;                            # 🧸💬 return self.snake_head_x
        
    def get_head_y( self ):                                  # 🧸💬 get_head_y function definition
        
        return self.snake_head_y;                            # 🧸💬 return self.snake_head_y
        
    def get_food_x( self ):                                  # 🧸💬 get_food_x function definition
    
        return self.food_x;                                  # 🧸💬 return self.food_x
        
    def get_food_y( self ):                                  # 🧸💬 get_food_y function definition
    
        return self.food_y;                                  # 🧸💬 return self.food_y
        
    def get_distance_x( self ):                              # 🧸💬 get_distance_x function definition
    
        return self.dist_x;                                  # 🧸💬 return self.dist_x
        
    def get_distance_y( self ):                              # 🧸💬 get_distance_y function definition
    
        return self.dist_y;                                  # 🧸💬 return self.dist_y
        
    def get_snakebody( self ):                               # 🧸💬 get_snakebody function definition
    
        return self.snake_body;                              # 🧸💬 return self.snake_body
        
    def get_snakebody_pos( self ):                           # 🧸💬 get_snakebody_pos function definition
    
        return self.snake_body_pos;                          # 🧸💬 return self.snake_body_pos
        
        
    def get_possibleactions( self ):                         # 🧸💬 get_possibleactions function definition

            
        # ...
    
        return K_h;                                          # 🧸💬 Implementation of prediction function return
                                                             # possible action and mapped constant value.
```

### 🧸💬 Agent Queue class implementation

🐨🎁🎵🎶 In ```Agent Queue``` implementation we prepared for cascade machine learning with multi-agents action classification, or known as ```parallel process``` or ```distribution processing units``` . </br> 

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/Class_inheritances/blob/main/multi-process.png"> </br>
    <b> Multi-processes and TF-Agents </b>
</p>

```
class AgentQueue():                                          # 🧸💬 Agent Queue class constructor

    def __init__ ( self, name ):                             # 🧸💬 Initialization method with name input parameter
        self.name = name;                                    # 🧸💬 Class variable define
        self.reward = 0;                                     # 🧸💬 Class variable define
        self.step = 0;                                       # 🧸💬 Class variable define
        
        ###
        self.new_Snake = Snake( "Snake_01" );                # 🧸💬 Create instance class object with reference
                                                             # class objectId reference is in class and calling the
                                                             # class __name__ is return name or custom message.
        return
    
    def next_step( self, action ):                           # 🧸💬 next_step action with action input parameter
        self.reward = p.act( action )                        # 🧸💬 Class variable assignment from action return
                                                             # 🧸💬 Good for reinforcement machine learning.
        self.step = self.step + 1;                           # 🧸💬 Additional step variable.
    
        return
        
    def game_over( self ):                                   # 🧸💬 Reset game environment variable
        self.reward = 0;                                     # 🧸💬 Reset game environment variable
        self.step = 0;                                       # 🧸💬 Reset game environment variable
    
        return
        
    def read_gamestate( self ):                              # 🧸💬 Define read game state or can blocked reference

        distance_x = self.new_Snake.get_distance_x();        # 🧸💬 Call internal class object function return
        distance_y = self.new_Snake.get_distance_y();        # 🧸💬 Call internal class object function return
        
        print( f"distance x: { distance_x } distance y: { distance_y }" );    # 🧸💬 Console display

        return
```

### 🧸💬 Running tasks iterations.

🦭💬 In applications working with ```communication methods``` from ```defined definition solution``` build and pre-trained, there are run timeout as call external method function and ```no response condition``` . </br>
🐑💬 ➰ Study can create of the exception flow or learning application from the environment working patterns to create of defined applications with various purposes. </br>

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/Class_inheritances/blob/main/Snank_AI_vs_Random_10_minutes.gif"> </br>
    <b> Sample Snake Agent Queue with AI object detection - * training 10 minutes time. </b>
</p>

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

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/Class_inheritances/blob/main/FlappyBird_small.gif"> </br>
    <b> Flappy birds with accerelation variables </b> </br>
    <img width="40%" src="https://github.com/jkaewprateep/Class_inheritances/blob/main/Snakes.gif"> </br>
    <b> Snakes </b>
</p>



