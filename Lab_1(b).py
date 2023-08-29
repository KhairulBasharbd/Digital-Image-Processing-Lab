import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageTk
import tkinter as tk

# Load a grayscale image of size 512x512 using PIL
image_path = r"C:\Users\Khairul_Bashar\Desktop\Lab\Digital Image Processing\aaa.jpg"  # Replace with the actual path to your image
original_image = Image.open(image_path).convert('L')

# Create a window using tkinter
window = tk.Tk()
window.title("Intensity Level Resolution Reduction")

# Create a canvas to display images
canvas = tk.Canvas(window, width=512, height=512)
canvas.pack()

# Perform the intensity level resolution reduction and display images
reduced_images = [original_image]
while original_image.mode == 'L':
    original_image = original_image.point(lambda p: p >> 1)  # Decrease by one bit
    reduced_images.append(original_image)

for idx, image in enumerate(reduced_images):
    tk_image = ImageTk.PhotoImage(image=image)
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    canvas.update()
    canvas.delete("all")  # Clear the canvas

    if idx < len(reduced_images) - 1:
        window.after(1000)  # Display each image for 1 second (adjust as needed)

# Close the window when done
window.destroy()
