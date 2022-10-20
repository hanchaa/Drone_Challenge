# AGC2022_round3_task1

### Anaconda environment
```
CONDAT_ROOT=[to your Anaconda root directory]
conda env create -f drone_task1.yml --prefix $CONDA_ROOT/envs/drone_task1
conda activate drone_task1
```

### Drone toy dataset location (Server 66)
- location: `/home/eulrang/workspace/temp_dataset/drone_task1_toy`
- making symbolic link:
```
cd ./data_toy
ln -s /home/eulrang/worksapce/temp_dataset/drone_task1_toy drone_task1_toy
```

### Inference test
- Case 1: One image clue only
```
python run.py --video_path ./data_toy/drone_task1_toy/22_cam.mp4 --clue_path ./data_toy/drone_task1_toy/22_case1
```
- Case 2: One text clue only (in progress)
```
python run.py --video_path ./data_toy/drone_task1_toy/21_cam.mp4 --clue_path ./data_toy/drone_task1_toy/case2
```
- Case 3: One image clue and one text clue (in progress)
```
python run.py --video_path ./data_toy/drone_task1_toy/21_cam.mp4 --clue_path ./data_toy/drone_task1_toy/case3
```