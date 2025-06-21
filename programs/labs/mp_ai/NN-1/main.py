from viewer import BW_BMP_Viewer
import tkinter as tk


if __name__ == "__main__":
    images_folder = r"C:\Users\User\python-study\programs\labs\mp_ai\NN-1\images"

    root = tk.Tk()
    root.geometry("800x600")
    app = BW_BMP_Viewer(root, images_folder)
    root.mainloop()
