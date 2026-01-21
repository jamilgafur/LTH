#!/bin/bash

# Get the absolute path of the current directory to ensure we can always return
START_DIR=$(pwd)

echo "Searching for submission scripts inside: $START_DIR"
echo "---------------------------------------------------"

# Find all 'submit_collapse_jobs.sh' files recursively
find . -name "submit_collapse_jobs.sh" | sort | while read script_path; do
    
    # Get the directory where the script lives
    SCRIPT_DIR=$(dirname "$script_path")
    SCRIPT_NAME=$(basename "$script_path")
    
    echo "[+] Found script in: $SCRIPT_DIR"
    
    # 1. Enter the directory (Critical for qsub relative paths)
    cd "$SCRIPT_DIR" || continue
    
    # 2. Ensure executable permissions
    chmod +x "$SCRIPT_NAME"
    
    # 3. Execute the script
    # We use ./ to ensure it runs from the current dir context
    ./"$SCRIPT_NAME"
    
    # 4. Return to start for the next iteration (safety measure)
    cd "$START_DIR"
    
    echo "[✓] Submitted."
    echo "---------------------------------------------------"
done

echo "All tasks processed."
