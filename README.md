# IA-g6 — Lights Out

## Run Instructions

### Prerequisites

* Python 3.12
* pip pygame

### 1. Install dependencies

pip install pygame

### 2. Run the program

python lightsOut.py 
**OR**
python3 lightsOut.py

## How to Play

When you launch the game, you will see the main menu with two options: **Play** and **Algorithms**.

### Play (Free Play)

Select **Play** to play the game manually. You will be asked to choose a board size: **3×3**, **4×4**, or **5×5**.

Click any cell to toggle it and its neighbours. The goal is to turn off all the lights.

> 💡 **Hint:** Press **H** during the game to get a hint on which cell to click next.

### Algorithms (Auto Solver)

Select **Algorithms** to watch the game be solved automatically. You will see a list of available algorithms — pick one and the solver will play the game for you.

You can also select the option to load a board from a text file. The text tile must only include a square matrix with 1s and 0s.