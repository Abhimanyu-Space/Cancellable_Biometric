import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def collatz_steps(n):
    """Generates chaotic mathematical noise based on the Collatz Conjecture."""
    n = int(abs(n))
    n = 1 if n == 0 else n
    steps = 0
    while n != 1:
        if n % 2 == 0: 
            n = n // 2
        else: 
            n = 3 * n + 1
        steps += 1
    return steps

def normalize_with_convex_hull(minutiae_array):
    """Anchors the fingerprint to its own geometric centroid."""
    if len(minutiae_array) < 3: 
        return minutiae_array, (0,0)
    
    hull = ConvexHull(minutiae_array)
    hull_points = minutiae_array[hull.vertices]
    centroid_x = np.mean(hull_points[:, 0])
    centroid_y = np.mean(hull_points[:, 1])
    
    # --- VISUALIZATION BLOCK (Modified to show only once) ---
    # We check if a custom attribute 'shown' exists on the function itself
    if not hasattr(normalize_with_convex_hull, 'shown'):
        plt.figure(figsize=(6, 6))
        plt.scatter(minutiae_array[:, 0], minutiae_array[:, 1], c='blue', alpha=0.5, label='Minutiae')
        for simplex in hull.simplices:
            plt.plot(minutiae_array[simplex, 0], minutiae_array[simplex, 1], 'r-', linewidth=2)
        plt.plot(centroid_x, centroid_y, 'gX', markersize=15, label='Centroid (Anchor Point)')
        plt.title("Convex Hull & Centroid Anchor")
        plt.legend()
        plt.gca().invert_yaxis() # Match image coordinates
        plt.show() # SAVE THIS AS convex_hull_overlay.png
        
        # Set the flag so it doesn't show again during the loop
        normalize_with_convex_hull.shown = True
    # -------------------------------------------------------

    normalized_array = np.zeros_like(minutiae_array, dtype=float)
    normalized_array[:, 0] = minutiae_array[:, 0] - centroid_x
    normalized_array[:, 1] = minutiae_array[:, 1] - centroid_y
    
    return normalized_array, (centroid_x, centroid_y)

def collatz_biometric_vault(minutiae_array, user_pin):
    """The Hybrid Lock: Anchors geometry, then applies Collatz chaos."""
    # This call triggers the visualization once
    normalized_points, _ = normalize_with_convex_hull(minutiae_array)
    folded_template = np.zeros_like(normalized_points, dtype=float)
    
    for i in range(len(normalized_points)):
        x_val, y_val = normalized_points[i, 0], normalized_points[i, 1]
        
        # Seed logic verified up to 2^68
        collatz_seed = int(abs(x_val * user_pin)) + 1000
        chaos_multiplier = collatz_steps(collatz_seed)
        
        folded_template[i, 0] = x_val + (chaos_multiplier % 15)
        folded_template[i, 1] = y_val + (chaos_multiplier % 45)
        
    return folded_template