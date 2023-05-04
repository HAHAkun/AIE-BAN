# AIE-BAN
The code of the paper " Adaptive Image Enhancement based on Brightness-aware Network for Non-uniform Lighting Image"
## Requirements
1. Python 3.7 
2. Pytorch 1.0.0
3. opencv
4. torchvision 0.2.1
5. cuda 10.0

### Test: 

```
python lowlight_test.py 
```
The script will process the images in the sub-folders of "test_data" folder and make a new folder "result" in the "data". You can find the enhanced images in the "result_data" folder.

### Train:
```
python lowlight_train.py 
```
