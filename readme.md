## 1. Modifying the config file in configs/stdrunet.yml
- gt_folder_train: 'path/RGBW_train_dataset_fullres/GT_bayer/train_bayer_full_gt'
- QB_0db_folder_train: 'path/RGBW_train_dataset_fullres/input/train_RGBW_full_input_0dB'
- QB_24db_folder_train: 'path/RGBW_train_dataset_fullres/input/train_RGBW_full_input_24dB'
- QB_42db_folder_train: 'path/RGBW_train_dataset_fullres/input/train_RGBW_full_input_42dB'

- QB_0db_folder_val: 'path/RGBW_validation_dataset_fullres/input/valid_RGBW_full_input_0dB'
- QB_24db_folder_val: 'path/RGBW_validation_dataset_fullres/input/valid_RGBW_full_input_24dB'
- QB_42db_folder_val: 'path/RGBW_validation_dataset_fullres/input/valid_RGBW_full_input_42dB'

- QB_0db_folder_test: 'path/RGBW_test_dataset_fullres/input/test_RGBW_full_input_0dB'
- QB_24db_folder_test: 'path/RGBW_test_dataset_fullres/input/test_RGBW_full_input_24dB'
- QB_42db_folder_test: 'path/RGBW_test_dataset_fullres/input/test_RGBW_full_input_42dB'

- test_model_path: 'path/file/experiments/model_best.pt'
- save_path: 'path/file/'** 

## 2. Training:

```
$ bash train.sh
```

## 3. Validation:

```
$ bash val.sh
```

## 4. Testing:

```
$ bash test.sh
```