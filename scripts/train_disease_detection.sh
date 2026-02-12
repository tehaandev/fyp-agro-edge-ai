source .venv/bin/activate
python scripts/train_disease_detection.py
echo "Finished training disease detection model."

# run with `nohup scripts/train_disease_detection.sh & > output.log 2>&1 &` to keep it running after closing terminal