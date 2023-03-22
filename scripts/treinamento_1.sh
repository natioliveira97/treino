python3.7 treino/src/main.py \
    --experiment-name "treinamento_resnet_3_6_2" \
    --model-type-list "RawToTile_MobileNet" "TileToTile_Transformer" "TileToTileImage_SpatialViT" \
    --omit-list "omit_no_xml" \
    --error-as-eval-loss \
    --use-image-preds \
    --batch-size 2 \
    --series-length 2 \
    --accumulate-grad-batches 16 \
    --num-workers 6 \
    --train-split-path '/kaggle/input/dataset-artigo/dataset_artigo/train_files.txt' \
    --val-split-path '/kaggle/input/dataset-artigo/dataset_artigo/val_files.txt' \
    --test-split-path '/kaggle/input/dataset-artigo/dataset_artigo/test_files.txt' \
    --no-early-stopping \
    --tile-embedding-size 1000 \
    --raw-data-path '/kaggle/input/dataset-artigo/dataset_artigo/raw_images' \
    --labels-path '/kaggle/input/dataset-artigo/dataset_artigo/drive_clone_numpy_new' \
    --metadata-path './treino/metadata_dataset_artigo.pkl' \
    --original-height 1536 \
    --original-width 2048 \
    --resize-height 1080 \
    --resize-width 1920 \
    --crop-height 1080 \
    --tile-size 300 \
    --tile-overlap 0 \
    --max-epochs 80 \
    --no-resize-crop-augment \
    --smoke-threshold 100