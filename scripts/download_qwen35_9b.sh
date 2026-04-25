#!/bin/bash
set -e
DIR="/home/jinxu/.cache/tiny-agents/models/Qwen/Qwen3.5-9B"
BASE="https://hf-mirror.com/Qwen/Qwen3.5-9B/resolve/main"

echo "[Download] Qwen3.5-9B from hf-mirror.com"

# Small files: sequential download with curl
curl -sL -o "$DIR/config.json" "$BASE/config.json" && echo "config.json OK"
curl -sL -o "$DIR/tokenizer_config.json" "$BASE/tokenizer_config.json" && echo "tokenizer_config.json OK"
curl -sL -o "$DIR/tokenizer.json" "$BASE/tokenizer.json" && echo "tokenizer.json OK"
curl -sL -o "$DIR/vocab.json" "$BASE/vocab.json" && echo "vocab.json OK"
curl -sL -o "$DIR/merges.txt" "$BASE/merges.txt" && echo "merges.txt OK"
curl -sL -o "$DIR/chat_template.jinja" "$BASE/chat_template.jinja" && echo "chat_template.jinja OK"
curl -sL -o "$DIR/preprocessor_config.json" "$BASE/preprocessor_config.json" && echo "preprocessor_config.json OK"
curl -sL -o "$DIR/.gitattributes" "$BASE/.gitattributes" && echo ".gitattributes OK"
curl -sL -o "$DIR/README.md" "$BASE/README.md" && echo "README.md OK"
curl -sL -o "$DIR/model.safetensors.index.json" "$BASE/model.safetensors.index.json" && echo "index.json OK"

# Large files: parallel aria2c download
aria2c -c -x 8 -s 8 --max-connection-per-server=8 --timeout=300 --max-tries=5 --allow-overwrite=true --auto-file-renaming=false -d "$DIR" -o model.safetensors-00001-of-00004.safetensors "$BASE/model.safetensors-00001-of-00004.safetensors" &
PID1=$!
aria2c -c -x 8 -s 8 --max-connection-per-server=8 --timeout=300 --max-tries=5 --allow-overwrite=true --auto-file-renaming=false -d "$DIR" -o model.safetensors-00002-of-00004.safetensors "$BASE/model.safetensors-00002-of-00004.safetensors" &
PID2=$!
aria2c -c -x 8 -s 8 --max-connection-per-server=8 --timeout=300 --max-tries=5 --allow-overwrite=true --auto-file-renaming=false -d "$DIR" -o model.safetensors-00003-of-00004.safetensors "$BASE/model.safetensors-00003-of-00004.safetensors" &
PID3=$!
aria2c -c -x 8 -s 8 --max-connection-per-server=8 --timeout=300 --max-tries=5 --allow-overwrite=true --auto-file-renaming=false -d "$DIR" -o model.safetensors-00004-of-00004.safetensors "$BASE/model.safetensors-00004-of-00004.safetensors" &
PID4=$!

echo "Started part downloads: PIDs $PID1 $PID2 $PID3 $PID4"
wait $PID1 && echo "Part 1 done"
wait $PID2 && echo "Part 2 done"
wait $PID3 && echo "Part 3 done"
wait $PID4 && echo "Part 4 done"

echo "[Download] Qwen3.5-9B complete!"
