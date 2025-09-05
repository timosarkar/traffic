from tkinter import Tk, Label
from PIL import Image, ImageTk, ImageSequence

root = Tk()
root.title("GIF Viewer")

im = Image.open(input("filepath>"))
frames = [ImageTk.PhotoImage(frame.copy()) for frame in ImageSequence.Iterator(im)]

label = Label(root)
label.pack()

def update(ind):
    label.configure(image=frames[ind])
    root.after(100, update, (ind + 1) % len(frames))  # 100ms per frame

update(0)
root.mainloop()

