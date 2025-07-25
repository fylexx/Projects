'''
Ollama needs to be started locally for it to work
Currently using deepseek-r1:1.5b because of limited available local resources
Need to implement correct handling of LaTeX for correct display of math
'''
import tkinter as tk
from tkinter import ttk, messagebox
import ollama
import re

def generate_response():
    user_prompt = prompt_input.get("1.0", tk.END).strip()
    if not user_prompt:
        messagebox.showwarning("Input Needed", "Please enter a prompt.")
        return

    try:
        response = ollama.generate(model='deepseek-r1:1.5b', prompt=user_prompt)
        
        # Strips any <think>...</think> content
        cleaned = re.sub(r"<think>.*?</think>", "", response['response'], flags=re.DOTALL).strip()
        
        output_text.configure(state='normal')
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, cleaned)
        output_text.configure(state='disabled')
    except Exception as e:
        messagebox.showerror("Error", str(e))


# ------- GUI Setup -------
root = tk.Tk()
root.title("LLM Prompt Interface")
root.geometry("700x600")
root.configure(bg="#f9f9f9")

style = ttk.Style(root)
style.theme_use("clam")
style.configure("TLabel", background="#ff8e2b", font=("Segoe UI", 28))
style.configure("TButton", font=("Segoe UI", 28), padding=6)
style.configure("TFrame", background="#f9f9f9")

frame = ttk.Frame(root, padding=20)
frame.pack(fill=tk.BOTH, expand=True)

ttk.Label(frame, text="Enter full prompt:").pack(anchor="w", pady=(0, 5))
prompt_input = tk.Text(frame, height=12, font=("Segoe UI", 20), wrap="word", relief="solid", borderwidth=1, bg="#ffffff", fg="#000000")
prompt_input.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

ttk.Button(frame, text="Generate Response", command=generate_response).pack(pady=10)

ttk.Label(frame, text="Model Response:").pack(anchor="w", pady=(10, 5))
output_text = tk.Text(
    frame, height=10, font=("Segoe UI", 20),
    wrap="word", relief="solid", borderwidth=1,
    bg="#ffffff", fg="#000000"
)
output_text.pack(fill=tk.BOTH, expand=True)
output_text.configure(state='disabled')

root.mainloop()