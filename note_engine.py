from llama.index.core.tools import FunctionTool
import os

note_file = os.path.join("data", "notes.txt")
def read_notes():
    """Read notes from the file."""
    if not os.path.exists(note_file):
        return "No notes found."
    with open(note_file, "r") as file:
        return file.read()

    return "Note saved successfully."

note_engine = FunctionTool.from_default(
    name="note_saver",
    description="this tool can save a text based note to a file for the user",
    fn=save_notes,
)
print("Note engine initialized with read_notes function.")