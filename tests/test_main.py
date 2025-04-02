import sys
from pathlib import Path
sys.path.insert(0, '../')
# project_path = Path(__file__).resolve().parent.parent
# sys.path.append(str(project_path))
from minirl.core.agent import Agent

import matplotlib.pyplot as plt
import numpy as np 


if __name__ == '__main__':
    v = np.array([2, 1])
    A = np.array([[-1, 0], [0, 1]])  # 90° rotation matrix
    u = A @ v  # Transformed vector

    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot original vector v
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', label=r'$\vec{v}$')

    # Plot transformed vector u
    ax.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='red', label=r'$\vec{u} = A\vec{v}$')

    # Annotate vectors
    ax.text(v[0]*1.05, v[1]*1.05, r'$\vec{v}$', fontsize=14, color='blue')
    ax.text(u[0]*1.05, u[1]*1.05, r'$\vec{u}$', fontsize=14, color='red')

    # Set limits and grid
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_yticks(np.arange(-5, 6, 1))
    ax.grid(True, which="both")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    # Labels and legend
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Vector Transformation: $\\vec{u} = A \\vec{v}$')
    ax.legend()

    plt.show()