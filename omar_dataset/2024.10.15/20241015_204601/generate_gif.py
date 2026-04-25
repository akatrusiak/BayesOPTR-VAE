import pandas as pd
import ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Load data from CSV
df = pd.read_csv('training_set.csv')
# Parse the 'train_x' column
df['train_x'] = df['train_x'].apply(ast.literal_eval)

# Convert all items in train_x to float to ensure a numeric array
df['train_x'] = df['train_x'].apply(lambda lst: [float(x) for x in lst])

# Now stack the arrays
X_data = np.vstack(df['train_x'].values)

iterations = len(df)
ys = df['train_y'].values
num_dims = X_data.shape[1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.tight_layout()

# --- Setup the left subplot (Y over iterations) ---
line_y, = ax1.plot([], [], color='blue', marker='o', label='train_y', linewidth=1)
ax1.set_title('Progress of Y over Iterations')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Y Value')
ax1.grid(True)
ax1.legend(loc='best')
ax1.set_xlim(0, iterations)    # Set x-limits to full range of iterations
ax1.set_ylim(min(ys), max(ys)) # Set y-limits to range of Y

# --- Setup the right subplot (Inputs over iterations) ---
# Create a line for each dimension
lines_x = []
colors = plt.cm.get_cmap('tab10', num_dims)  # a colormap with a variety of colors
for d in range(num_dims):
    line, = ax2.plot([], [], label=f"Dim {d}", color=colors(d % 10), linewidth=2) 
    lines_x.append(line)

ax2.set_title('Inputs over Iterations')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Input Value')
ax2.grid(True)
ax2.set_xlim(0, iterations)

# Set y-limits based on the overall min/max across all dimensions
min_val = X_data.min()
max_val = X_data.max()
# Add a little padding
padding = (max_val - min_val)*0.05 if (max_val > min_val) else 1
ax2.set_ylim(min_val - padding, max_val + padding)

if num_dims <= 20:
    ax2.legend(loc='best')

def animate(i):
    # Update Y line
    line_y.set_data(range(i+1), ys[:i+1])
    
    # Update each dimension line with data up to iteration i
    for d, line in enumerate(lines_x):
        line.set_data(range(i+1), X_data[:i+1, d])
    return [line_y] + lines_x

ani = animation.FuncAnimation(fig, animate, frames=iterations, interval=100, blit=True, repeat=False)

ani.save('animation.gif', writer='pillow')