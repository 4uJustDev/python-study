import tkinter as tk
from gui import ImageDenoisingApp


def toggle_fullscreen(event=None):
    root.attributes("-fullscreen", not root.attributes("-fullscreen"))


if __name__ == "__main__":
    root = tk.Tk()
    root.attributes("-fullscreen", True)  # Запуск в полноэкранном режиме
    root.bind("<Escape>", toggle_fullscreen)  # Выход из полноэкранного режима по Escape
    app = ImageDenoisingApp(root)
    root.mainloop()
