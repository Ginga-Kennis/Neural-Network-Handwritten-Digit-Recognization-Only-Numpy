from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
from network import Network
import mnist_data_loader


net = Network([784,30,10])
training_data,test_data = mnist_data_loader.create_data()
net.SGD(training_data,3,10,3,test_data=test_data)



def get_ans(img,net):
    # resize image to 28x28 pixels
    img = img.resize((28, 28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    img = 1 - img / 255
    #reshaping to support our model input and normalizing
    img = np.reshape(img,(784,1))
    res = net.classify(img)
    return res

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=400, height=400, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command = self.classify_handwriting)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0)
        self.label.grid(row=0, column=1)
        self.classify_btn.grid(row=1, column=1)
        self.button_clear.grid(row=1, column=0)

        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        im = ImageGrab.grab((10,30,600,600))
        digit = get_ans(im,net)
        self.label.configure(text= str(digit))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=10
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()