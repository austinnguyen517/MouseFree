#for mouse interaction using pyauto
import pyautogui as pg
#define a class
'''
Make sure to have the screen of the computer smaller than detection region
Associations:

Hand: cursor
Two: double left click
Hang: single right click
Okay: left hold
'''

class Mouse():
    def __init__(self, dim):
        self.dim = dim # dimensions of the screen

    def moveCursorTo(self, x, y): #takes a ratio of x and y
        assert x <= 1 and y <= 1
        pg.moveTo(x*self.dim[0], y*self.dim[1], pg.MINIMUM_DURATION/2)

    def singleClickLeft(self):
        pg.click(button = 'left', clicks = 1)

    def doubleClickLeft(self):
        pg.click(button = 'left', clicks = 2)

    def holdLeft(self):
        pg.mouseDown()

    def releaseLeft(self):
        pg.mouseUp()
