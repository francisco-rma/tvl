import tkinter as tk
import customtkinter

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.geometry("500x300")



def login():
    print('login')

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=50, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Hello World")
label.pack(pady=12, padx=10)

entry1 = customtkinter.CTkEntry(master=frame, placeholder_text ='Username')
entry1.pack(pady=12, padx=10)

entry2 = customtkinter.CTkEntry(master=frame, placeholder_text ='Password', show='*')
entry2.pack(pady=12, padx=10)


button = customtkinter.CTkButton(master=frame, text = 'Login', command=login)
button.pack(pady=12, padx=10)


root.mainloop()

# class Aplikacja(tk.Frame):
#     def __init__(self, parent):
#         tk.Frame.__init__(self, parent)
#         self.grid()

#         zakladki=tk.ttk.Notebook(parent)
#         entries = {}
#         for title in ('Czapki', 'Dodatki', 'buty', 'spodnie', 'kurtka',
#                       'T-shirt', 'sweter', 'skarpetki', 'koszula'):
#             frame = tk.ttk.Frame(zakladki)
#             for row, txt in (0, 'Nazwa'), (1, "Kolor"), (2, "Firma"):
#                 tk.Label(frame, text=txt).grid(row=row, column=0)
#                 entry = tk.Entry(frame)
#                 entries[title, txt] = entry
#                 entry.grid(row=row, column=1)
#             zakladki.add(frame, text=title)
#         zakladki.grid()

# root= tk.Tk()
# root.title("Szaffa")
# app= Aplikacja(root)
# root.mainloop()