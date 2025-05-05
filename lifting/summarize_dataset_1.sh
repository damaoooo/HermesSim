echo "building training_Dataset-1"
python lifting/dataset_summary.py \
    --cfg_summary dbs/Dataset-1/cfg_summary/training \
    --dataset_info_csv dbs/Dataset-1/training_Dataset-1.csv \
    --cfgs_folder dbs/Dataset-1/features/training/acfg_features_training_Dataset-1

echo "building validation_Dataset-1"
python lifting/dataset_summary.py \
    --cfg_summary dbs/Dataset-1/cfg_summary/validation \
    --dataset_info_csv dbs/Dataset-1/validation_Dataset-1.csv \
    --cfgs_folder dbs/Dataset-1/features/validation/acfg_features_validation_Dataset-1

echo "building testing_Dataset-1"
python lifting/dataset_summary.py \
    --cfg_summary dbs/Dataset-1/cfg_summary/testing \
    --dataset_info_csv dbs/Dataset-1/testing_Dataset-1.csv \
    --cfgs_folder dbs/Dataset-1/features/testing/acfg_features_testing_Dataset-1
