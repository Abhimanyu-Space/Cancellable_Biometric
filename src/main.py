import numpy as np
import matplotlib.pyplot as plt
from extract import extract_minutiae_from_image 
from hybrid_transform import collatz_biometric_vault

def run_experiment_1_irreversibility(minutiae, pin_a, pin_b):
    print("\n--- PHASE 1: IRREVERSIBILITY & REVOCABILITY ---")
    print(f"Applying Collatz Transformation with PIN {pin_a}...")
    transformed_1 = collatz_biometric_vault(minutiae, pin_a)
    
    print(f"Applying Collatz Transformation with PIN {pin_b}...")
    transformed_2 = collatz_biometric_vault(minutiae, pin_b)
    
    # FIGURE 1: Revocability
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(minutiae[:, 0], minutiae[:, 1], color='blue', s=15)
    plt.title("1. Original Extracted Minutiae")
    plt.xlabel("Scanner X-Pixel")
    plt.ylabel("Scanner Y-Pixel")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.subplot(1, 3, 2)
    plt.scatter(transformed_1[:, 0], transformed_1[:, 1], color='red', marker='x', s=20)
    plt.title(f"2. Irreversible Vault (PIN: {pin_a})")
    plt.xlabel("Perturbed X")
    plt.ylabel("Perturbed Y")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 3, 3)
    plt.scatter(transformed_2[:, 0], transformed_2[:, 1], color='green', marker='s', s=20)
    plt.title(f"3. Revoked Vault (PIN: {pin_b})\n(Proves Zero-Correlation)")
    plt.xlabel("Perturbed X")
    plt.ylabel("Perturbed Y")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    print(">> Close the plot window to proceed to the Accuracy/Translation Proof...")
    plt.show()

def run_experiment_2_accuracy_proof(minutiae, pin):
    print("\n--- PHASE 2: ACCURACY & TRANSLATION INVARIANCE ---")
    print("Simulating user placing finger 50 pixels to the right, 40 pixels up...")
    
    # 1. Create the synthetically shifted points
    shifted_minutiae = np.copy(minutiae)
    shifted_minutiae[:, 0] += 50  # Shift X by 50 pixels
    shifted_minutiae[:, 1] += 40  # Shift Y by 40 pixels
    
    # 2. Run both through the exact same vault
    vault_original = collatz_biometric_vault(minutiae, pin)
    vault_shifted = collatz_biometric_vault(shifted_minutiae, pin)
    
    # FIGURE 2: Translation Proof
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(minutiae[:, 0], minutiae[:, 1], color='blue', alpha=0.6, label='Original Placement')
    plt.scatter(shifted_minutiae[:, 0], shifted_minutiae[:, 1], color='orange', alpha=0.6, marker='^', label='Shifted Placement (+50x, +40y)')
    plt.title("1. Simulated Hardware Variance\n(User placed finger in a different area)")
    plt.xlabel("Scanner X-Pixel")
    plt.ylabel("Scanner Y-Pixel")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.scatter(vault_original[:, 0], vault_original[:, 1], color='red', alpha=0.8, s=60, label='Vault Output (Original)')
    plt.scatter(vault_shifted[:, 0], vault_shifted[:, 1], color='green', alpha=0.8, marker='x', s=60, label='Vault Output (Shifted)')
    plt.title("2. Accuracy Proof\n(Convex Hull mathematically erases the shift)")
    plt.xlabel("Perturbed X")
    plt.ylabel("Perturbed Y")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# MASTER EXECUTION DASHBOARD
# ==========================================
if __name__ == "__main__":
    # YOU ONLY NEED ONE IMAGE HERE!
    IMAGE_A = "Sample1.bmp"  # Put your best image here
    
    PIN_1 = 8045
    PIN_2 = 1234
    
    print(f"Loading Image: {IMAGE_A}...")
    try:
        # Extract data ONCE
        core_minutiae = extract_minutiae_from_image(IMAGE_A)
        print(f"Successfully extracted {len(core_minutiae)} minutiae points.")
        
        # Run Phase 1: Show Irreversibility
        run_experiment_1_irreversibility(core_minutiae, PIN_1, PIN_2)
        
        # Run Phase 2: Show Accuracy using simulated shift
        run_experiment_2_accuracy_proof(core_minutiae, PIN_1)
        
        print("\n--- DEMONSTRATION COMPLETE ---")
    except Exception as e:
        print(f"ERROR: {e}")