python convert_json_list_to_lmdb.py \
--json_list ./annotations/WIDER_train_annotations.txt \
--dataset_path ./datasets/WIDER_Face/WIDER_train/images/ \
--dest ./datasets/lmdb/ \
--train

python convert_json_list_to_lmdb.py \
--json_list ./annotations/WIDER_val_annotations.txt \
--dataset_path ./datasets/WIDER_Face/WIDER_val/images/ \
--dest ./datasets/lmdb