import time
from random import choice
import cv2 as cv
import numpy as np

def game(colors: dict={'Blue': (255, 0, 0), 'Yellow': (0, 255, 255)}, game_time: int=30, per_color_time: int=5, repeating: bool=False, visual: bool=False):
    """
    Runs a simple Reaction Game

    Args:
        colors: List of available Colors (the more colors, the more difficult the game)
        game_time: Duration of the Game in Seconds
        per_color_time: Time between each Color in Seconds
        repeating: if True, two sequential colors can be identical; if False, no sequentially repeating colors
    """

    print(f"Game Duration: {game_time}")
    print(f"Round Duration: {per_color_time}")
    print(f"Game will start in: ")
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("GO!\n")
    time.sleep(0.5)

    color_names = list(colors.keys())
    duration = game_time # fix duration for display

    if visual == True:
        previous_color = None
        while game_time > 0:
            if not repeating:
                available_colors = [c for c in color_names if c != previous_color]
                current_color = choice(available_colors)
            else:
                current_color = choice(color_names)
    
            img = np.zeros((500, 500, 3), dtype=np.uint8)
            img[:] = colors[current_color]  
            
            cv.imshow(f'Reaction Game | Duration={duration}s | Round Time={per_color_time}s', img)
            
            previous_color = current_color
            
            if cv.waitKey(per_color_time * 1000) & 0xFF == ord('q'):
                break
                
            game_time -= per_color_time
        
        cv.destroyAllWindows()
    else:
        if repeating == False:
            previous_color = None
            while game_time > 0:
                available_colors = [c for c in color_names if c != previous_color]
                current_color = choice(available_colors)

                print(current_color)
                previous_color = current_color

                time.sleep(per_color_time)
                game_time -= per_color_time
        else:
            while game_time > 0:
                print(choice(color_names))
                time.sleep(per_color_time)
                game_time -= per_color_time

    print("\nEnd of Game")

def main():
    colors = {
        'Blue': (255, 0, 0),
        'Red': (0, 0, 255),
        'Green': (0, 255, 0),
        'Yellow': (0, 255, 255)
    }
    game_time = 30
    per_color_time = 2
    repeating = False
    visual = True
    game(colors, game_time, per_color_time, repeating, visual)

if __name__ == '__main__':
    main()