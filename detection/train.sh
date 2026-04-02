#!/bin/bash
CURRENT_DATE=$(date +"%Y-%m-%d")
LOG_FILE="training.$CURRENT_DATE.log"

# Record the start time
echo "========================================" >> "$LOG_FILE"
echo "Training started at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

#for i in {0..9}; do echo "Starting fold $i"; python luna16_training.py -e ./config/environment_luna16_fold$i.json -c ./config/config_train_luna16_16g.json >> $LOG_FILE 2>&1; done
# Run the loop using seq for maximum compatibility
for i in {0..9}; do
    echo "Starting fold $i at $(date +"%H:%M:%S")..." | tee -a "$LOG_FILE"
    python luna16_training.py -e ./config/environment_luna16_fold$i.json -c ./config/config_train_luna16_16g.json | tee -a "$LOG_FILE"
done

# Record the end time
echo "========================================" >> "$LOG_FILE"
echo "Training finished at: $(date)" | tee -a "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
