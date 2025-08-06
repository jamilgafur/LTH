from codecarbon import EmissionsTracker
import time
import os

os.makedirs('emissions_output', exist_ok=True)
# Create the tracker
tracker = EmissionsTracker(output_dir='emissions_output', measure_power_secs=1)

# Start tracking
tracker.start()

# Simulate some workload
print("Running simulated workload...")
time.sleep(5)  # Replace with your real workload

# Stop tracking
emissions = tracker.stop()

print(f"\nEstimated CO₂ emissions: {emissions:.6f} kg")
