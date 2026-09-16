import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GOLDEN_FILE = BASE_DIR / "golden_set.csv"


# ============================================================
# LOAD GOLDEN SET
# ============================================================

df = pd.read_csv(GOLDEN_FILE)

# Make sure label columns exist
if "gold_intent" not in df.columns:
    df["gold_intent"] = ""

if "gold_handling" not in df.columns:
    df["gold_handling"] = ""


# ============================================================
# INTENTS
# ============================================================

intents = sorted(
    df["intent"]
    .dropna()
    .unique()
    .tolist()
)

handling_options = [
    "AUTOMATIC",
    "HUMAN"
]


# ============================================================
# STATE
# ============================================================

current_index = 0


# ============================================================
# SAVE CURRENT LABEL
# ============================================================

def save_current():

    global current_index

    selected_intent = intent_var.get()
    selected_handling = handling_var.get()

    if not selected_intent:
        messagebox.showwarning(
            "Missing Intent",
            "Please select an intent."
        )
        return

    if not selected_handling:
        messagebox.showwarning(
            "Missing Handling",
            "Please select AUTOMATIC or HUMAN."
        )
        return

    df.at[
        current_index,
        "gold_intent"
    ] = selected_intent

    df.at[
        current_index,
        "gold_handling"
    ] = selected_handling

    df.to_csv(
        GOLDEN_FILE,
        index=False
    )

    next_example()


# ============================================================
# NEXT EXAMPLE
# ============================================================

def next_example():

    global current_index

    if current_index >= len(df) - 1:

        df.to_csv(
            GOLDEN_FILE,
            index=False
        )

        messagebox.showinfo(
            "Completed",
            "All 200 examples have been labelled!"
        )

        root.destroy()

        return

    current_index += 1

    show_example()


# ============================================================
# PREVIOUS EXAMPLE
# ============================================================

def previous_example():

    global current_index

    if current_index > 0:

        current_index -= 1

        show_example()


# ============================================================
# DISPLAY EXAMPLE
# ============================================================

def show_example():

    row = df.iloc[current_index]

    progress_label.config(
        text=f"Example {current_index + 1} / {len(df)}"
    )

    conversation_text.delete(
        "1.0",
        tk.END
    )

    conversation_text.insert(
        tk.END,
        str(row["conversation"])
    )

    # Existing gold label
    existing_intent = str(
        row["gold_intent"]
    )

    existing_handling = str(
        row["gold_handling"]
    )

    if existing_intent in intents:
        intent_var.set(existing_intent)
    else:
        intent_var.set("")

    if existing_handling in handling_options:
        handling_var.set(existing_handling)
    else:
        handling_var.set("")


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Golden Evaluation Set - Labeling Tool"
)

root.geometry(
    "1000x700"
)


# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(
    root,
    text="Golden Evaluation Set",
    font=("Arial", 22, "bold")
)

title_label.pack(
    pady=(20, 5)
)


progress_label = tk.Label(
    root,
    text="",
    font=("Arial", 14)
)

progress_label.pack(
    pady=5
)


# ============================================================
# CONVERSATION
# ============================================================

conversation_frame = tk.Frame(root)

conversation_frame.pack(
    fill="both",
    expand=True,
    padx=30,
    pady=15
)


conversation_text = tk.Text(
    conversation_frame,
    wrap="word",
    font=("Arial", 12)
)

conversation_text.pack(
    side="left",
    fill="both",
    expand=True
)


scrollbar = tk.Scrollbar(
    conversation_frame,
    command=conversation_text.yview
)

scrollbar.pack(
    side="right",
    fill="y"
)

conversation_text.config(
    yscrollcommand=scrollbar.set
)


# ============================================================
# LABEL SECTION
# ============================================================

label_frame = tk.Frame(root)

label_frame.pack(
    fill="x",
    padx=30,
    pady=10
)


# Intent

tk.Label(
    label_frame,
    text="Gold Intent:",
    font=("Arial", 12, "bold")
).grid(
    row=0,
    column=0,
    padx=10,
    pady=10,
    sticky="w"
)


intent_var = tk.StringVar()

intent_dropdown = ttk.Combobox(
    label_frame,
    textvariable=intent_var,
    values=intents,
    state="readonly",
    width=40
)

intent_dropdown.grid(
    row=0,
    column=1,
    padx=10,
    pady=10
)


# Handling

tk.Label(
    label_frame,
    text="Gold Handling:",
    font=("Arial", 12, "bold")
).grid(
    row=1,
    column=0,
    padx=10,
    pady=10,
    sticky="w"
)


handling_var = tk.StringVar()

handling_dropdown = ttk.Combobox(
    label_frame,
    textvariable=handling_var,
    values=handling_options,
    state="readonly",
    width=40
)

handling_dropdown.grid(
    row=1,
    column=1,
    padx=10,
    pady=10
)


# ============================================================
# BUTTONS
# ============================================================

button_frame = tk.Frame(root)

button_frame.pack(
    pady=20
)


previous_button = tk.Button(
    button_frame,
    text="← Previous",
    command=previous_example,
    width=15
)

previous_button.grid(
    row=0,
    column=0,
    padx=10
)


save_button = tk.Button(
    button_frame,
    text="Save & Next →",
    command=save_current,
    width=20
)

save_button.grid(
    row=0,
    column=1,
    padx=10
)


# ============================================================
# START
# ============================================================

show_example()

root.mainloop()