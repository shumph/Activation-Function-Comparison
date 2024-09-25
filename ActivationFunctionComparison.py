import numpy as np
import tkinter as tk
import ttkbootstrap as ttk
import matplotlib.pyplot as plt
from HyperbolicTangent import * 
from RectifiedLinearUnit import * 
from Sigmoid import * 

#setup window
window =  tk.Tk()
window.title('Activation Function Comparison')
window.geometry('1000x1000')
select_iterations = ttk.Label(window, text = 'Input Desired Number Of Iterations', font  = 'calibri 40')
select_iterations.pack()

enter_iter = tk.Entry(window)
enter_iter.pack(padx = 10)
iterations = set()
def close_one(iterations, close1):
    iterations.add(int(enter_iter.get()))
    window.destroy()

close1 = tk.Button(window, 
                   text = 'Enter',
                   command = lambda: close_one(iterations, close1)
                   )
close1.pack()

#run
window.mainloop()

iterations = int(str(iterations).strip('{}'))

#second window
window =  tk.Tk()
window.title('Activation Function Comparison')
window.geometry('1000x1000')
select_learn_rate = ttk.Label(window, text = 'Input Desired Learn Rate (0.15 Recommended)', font  = 'calibri 37')
select_learn_rate.pack()

enter_learn_rate = tk.Entry(window)
enter_learn_rate.pack(padx = 10)
learn_rate = set()
def close_two(learn_rate, close2):
    learn_rate.add(float(enter_learn_rate.get()))
    window.destroy()

close2 = tk.Button(window, 
                   text = 'Enter',
                   command = lambda: close_two(learn_rate, close2)
                   )
close2.pack()

#run
window.mainloop()
learn_rate = float(str(learn_rate).strip('{}'))

#intiialzation for grad descent
(input_train, target_train), (dummy, dummy) = mnist.load_data()
w = input_train.shape[1]
h = input_train.shape[2]
input_train = input_train.reshape(input_train.shape[0],w*h).T / 255



print('Running HyperbolicTangent.py...')
weight_one, bias_one, weight_two, bias_two, tanh_accuracy_list, tanh_image_list = gradient_descent_tanh(input_train, target_train, learn_rate, iterations, w, h)


print('Running RectifiedLinearUnit.py...')
weight_one, bias_one, weight_two, bias_two, relu_accuracy_list, relu_image_list = gradient_descent_relu(input_train, target_train, learn_rate, iterations, w, h)


print('Running Sigmoid.py...')
weight_one, bias_one, weight_two, bias_two, sigmoid_accuracy_list, sigmoid_image_list = gradient_descent_sigmoid(input_train, target_train, learn_rate, iterations, w, h)


print('All three gradient descents are complete!')


window = tk.Tk()
window.title('Activation Function Comparison')
window.geometry('2000x1000')
window.columnconfigure(0, weight = 2)
window.columnconfigure(1, weight = 2)
window.columnconfigure(2, weight = 2)
window.columnconfigure(3, weight = 1)
window.columnconfigure(4, weight = 1)
window.rowconfigure(0, weight = 1)
window.rowconfigure(1, weight = 10)
window.rowconfigure(2, weight = 1)

zero_label = tk.Label(window, text = '0%').grid(row = 1, column = 3, sticky = 'sw')
fifty_label = tk.Label(window, text = '50%').grid(row = 1, column = 3, sticky = 'w')
one_hundred_label = tk.Label(window, text = '100%').grid(row = 1, column = 3, sticky = 'nw')

tanh_label = tk.Label(window, text = 'Tanh').grid(row = 2, column = 0, sticky = 'sew')
relu_label = tk.Label(window, text = 'RelU').grid(row = 2, column = 1, sticky = 'sew')
sigmoid_label = tk.Label(window, text = 'Sigmoid').grid(row = 2, column = 2, sticky = 'sew')




def load_graph(window):
    i = min(slider.get(), iterations-1)
    tanh_accuracy = tanh_accuracy_list[i]
    relu_accuracy = relu_accuracy_list[i]
    sigmoid_accuracy = sigmoid_accuracy_list[i]

    iteration_label = tk.Label(window, text = f'Iteration: {i+1}').grid(row=1, column=4, sticky= 's')

    label_one = tk.Label(window, text = f'{tanh_accuracy:.3%}').grid(row = 0, column = 0, sticky = 'sew')
    label_two = tk.Label(window, text = f'{relu_accuracy:.3%}').grid(row = 0, column = 1, sticky = 'sew')
    label_three = tk.Label(window, text = f'{sigmoid_accuracy:.3%}').grid(row = 0, column = 2, sticky = 'sew')

    tanh_height = int(900 - 900 * tanh_accuracy)
    relu_height = int(900 -900 * relu_accuracy)
    sigmoid_height = int(900 - 900 * sigmoid_accuracy)

    canvas_one = tk.Canvas(window, bg = 'white')
    canvas_two = tk.Canvas(window, bg = 'white')
    canvas_three = tk.Canvas(window, bg = 'white')

    canvas_one.delete(all)
    canvas_two.delete(all)
    canvas_three.delete(all)

    canvas_one.create_rectangle((125, tanh_height, 425, 900), fill= 'red')
    canvas_two.create_rectangle((125, relu_height, 425, 900), fill= 'green')
    canvas_three.create_rectangle((125, sigmoid_height, 425, 900), fill= 'blue')

    canvas_one.grid(row = 1, column = 0, sticky = 'nesw')
    canvas_two.grid(row = 1, column = 1, sticky = 'nesw')
    canvas_three.grid(row = 1, column = 2, sticky = 'nesw')



slider = tk.Scale(window, from_= int(iterations), to=0, length = 500, command = lambda x :load_graph(window))
slider.grid(row=1, column = 4)


window.mainloop()


window = tk.Tk()
window.title('View Specific Guesses')
window.geometry('500x300')
window.columnconfigure(0, weight = 1)
window.columnconfigure(1, weight  = 2)
window.columnconfigure(2, weight  = 2)
window.columnconfigure(3, weight  = 2)
window.columnconfigure(4, weight  = 1)
window.rowconfigure(0, weight=1)
window.rowconfigure(1, weight=3)
window.rowconfigure(2, weight=3)
window.rowconfigure(3, weight=3)
window.rowconfigure(4, weight=1)
iteration = tk.IntVar() 
instructions = tk.Label(window, text = f'Enter image from 1 to {iterations} to view:').grid(row=0, column=1, sticky = 'sw')
entry = tk.Entry(window, textvariable = iteration).grid(row=1, column = 1, sticky = 'w')



def show():
    i = iteration.get()-1
    tanh_prediction = tanh_image_list[i][0]
    relu_prediction = relu_image_list[i][0]
    sigmoid_prediction = sigmoid_image_list[i][0]
    label1 = tk.Label(window, text = f'Tanh Predicts: {tanh_prediction}').grid(row = 3, column=1)
    label1 = tk.Label(window, text = f'RelU Predicts: {relu_prediction}').grid(row = 3, column=2)
    label1 = tk.Label(window, text = f'SigmoidPredcits: {sigmoid_prediction}').grid(row = 3, column=3)
    plt.gray()
    plt.imshow(relu_image_list[i][1], interpolation='nearest')
    plt.show()
    

show = tk.Button(window, text = 'Show', command=show).grid(row=1, column = 1)


window.mainloop()



