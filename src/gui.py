import tkinter as tk
import customtkinter
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as an

from structs.tvl_struct import tvl_struct
from tvl import tvl

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.geometry("500x300")


def simulate():
    struct = tvl_struct()
    struct.iteration_number = int(
        iteration_number.get() or tvl_struct.iteration_number)
    struct.population_number = int(
        population_number.get() or tvl_struct.population_number)
    struct.talent_lower_bound = float(
        talent_lower_bound.get() or tvl_struct.talent_lower_bound)
    struct.talent_upper_bound = float(
        talent_upper_bound.get() or tvl_struct.talent_upper_bound)
    struct.talent_avg = float(talent_avg.get() or tvl_struct.talent_avg)
    struct.talent_std = float(talent_std.get() or tvl_struct.talent_std)
    struct.le = float(le.get() or tvl_struct.le)
    struct.ue = float(ue.get() or tvl_struct.ue)

    sim = tvl(struct)

    sim.run()


frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=50, fill="both", expand=True)

button = customtkinter.CTkButton(
    master=frame, text='TvL - Single run', command=simulate)
button.pack(pady=12, padx=10)

label = customtkinter.CTkLabel(master=frame, text="Talent vs Luck Simulator")
label.pack(pady=12, padx=10)

iteration_number = customtkinter.CTkEntry(
    master=frame, placeholder_text='Number of iterations')
iteration_number.pack(pady=12, padx=10)

population_number = customtkinter.CTkEntry(
    master=frame, placeholder_text='Size of the population')
population_number.pack(pady=12, padx=10)

talent_lower_bound = customtkinter.CTkEntry(
    master=frame, placeholder_text='Talent lower bound')
talent_lower_bound.pack(pady=12, padx=10)

talent_upper_bound = customtkinter.CTkEntry(
    master=frame, placeholder_text='Talent upper bound')
talent_upper_bound.pack(pady=12, padx=10)

talent_avg = customtkinter.CTkEntry(
    master=frame, placeholder_text='Average talent value')
talent_avg.pack(pady=12, padx=10)

talent_std = customtkinter.CTkEntry(
    master=frame, placeholder_text='Average talent value')
talent_std.pack(pady=12, padx=10)

runs = customtkinter.CTkEntry(
    master=frame, placeholder_text='Number of runs')
runs.pack(pady=12, padx=10)


le = customtkinter.CTkEntry(
    master=frame, placeholder_text='Probability of lucky event')
le.pack(pady=12, padx=10)


ue = customtkinter.CTkEntry(
    master=frame, placeholder_text='Probability of unlucky event')
ue.pack(pady=12, padx=10)

ne = customtkinter.CTkEntry(
    master=frame, placeholder_text='Probability of no events')
ne.pack(pady=12, padx=10)

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
