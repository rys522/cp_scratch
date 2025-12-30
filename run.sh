#!/bin/bash

VISUALIZE_FLAG=""
ASSET_DIR=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --visualize)
            VISUALIZE_FLAG="--visualize"
            shift 
            ;;
        --asset-dir)
            ASSET_DIR="$2"
            shift 
            shift 
            ;;
        *)
            shift 
            ;;
    esac
done

datasets=("zara1" "zara2" "hotel" "univ" "eth")
controllers=("ecp-mpc" "acp-mpc" "cc")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for dataset in "${datasets[@]}"; do
    for controller in "${controllers[@]}"; do
        echo "Evaluating dataset: $dataset / controller: $controller"

        cmd=(python "$SCRIPT_DIR/evaluate_controller.py" \
             --dataset "$dataset" \
             --controller "$controller")
        if [[ -n $VISUALIZE_FLAG ]]; then
            cmd+=("$VISUALIZE_FLAG")
            if [[ -n $ASSET_DIR ]]; then
                cmd+=(--asset_dir "$ASSET_DIR")
            fi
        fi

        echo "${cmd[@]}"
        "${cmd[@]}"
    done
done
