import tkinter as tk
import customtkinter
import functions as tvl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as an

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.geometry("500x300")


def login():
    np.set_printoptions(precision=3)

    # iter_n: number of iterations to go through
    iter_n = 500

    # pop_n: number of individuals in the popoulation
    pop_n = 1000

    # lb: lower bound of talent
    # ub: upper bound of talent
    lb, ub = 0, 1

    # mu: average value of the talent distribution
    # std: standard deviation of the talent distribution
    mu, std = 0.6, 0.1

    # le: chance for an individual to go through a lucky event
    le = 0.25

    # ue: chance for an individual to go through an unlucky event
    ue = 0.25

    # runs: number of runs to aggregate over
    runs = 100

    talent, talent_index = tvl.populate(pop_n, lb, ub, mu, std)

    # Running the simulations:
    mst, msp, successful = tvl.many_runs(talent, iter_n, ue, le, runs)

    # msc: Most Successful Capital (final capital of the most succesful individual)
    msc = tvl.map_to_capital(msp)

    # print(np.column_stack([msc, msp]))

    plt.hist(successful[:, 0], bins=100, range=(0, 1))
    plt.title("Histogram of the talent of successful individuals")
    plt.xlabel("Talent")
    plt.ylabel("Number of occurences")
    plt.legend(["Iterations: " + str(iter_n)], loc="upper left")
    plt.savefig("successful_individuals")
    plt.show()

    print("\nMean position of successful individuals: ", np.mean(successful[:, 1]))
    print(
        "Mean capital of successful individuals: ",
        np.mean(tvl.map_to_capital(successful[:, 1])),
    )
    print("Mean talent of successful individuals: ", np.mean(successful[:, 0]))

    plt.clf()

    plt.hist(mst, bins=100, range=(0, 1))
    plt.title("Histogram of the talent of the most successful individual")
    plt.xlabel("Talent")
    plt.ylabel("Number of occurences")
    plt.legend(["Iterations: " + str(iter_n)], loc="upper left")
    plt.savefig("mst")
    plt.show()

    print("\nMean maximum position: ", np.mean(msp))
    print("Mean maximum capital: ", np.mean(msc))
    print("Mean associated talent: ", np.mean(mst))


frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=50, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Hello World")
label.pack(pady=12, padx=10)

entry1 = customtkinter.CTkEntry(master=frame, placeholder_text="Username")
entry1.pack(pady=12, padx=10)

entry2 = customtkinter.CTkEntry(master=frame, placeholder_text="Password", show="*")
entry2.pack(pady=12, padx=10)


button = customtkinter.CTkButton(master=frame, text="TvL - Many runs", command=login)
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
