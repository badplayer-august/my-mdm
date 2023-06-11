# my-mdm
Modify from [DiT](https://github.com/facebookresearch/DiT) and [MDM](https://github.com/GuyTevet/motion-diffusion-model)
## Train
Check point should save in `results` directory
```
python my_train.py
```
## Sample
Motion animations should save in `animations` directory
```
python my_sample --ckpt /path/to/ckpt
```
## Dataset
### Get T2M evaluators
```
echo -e "Downloading T2M evaluators"
gdown --fuzzy https://drive.google.com/file/d/1DSaKqWX2HlwBtVH5l7DdW96jeYUIXsOP/view
gdown --fuzzy https://drive.google.com/file/d/1tX79xk0fflp07EZ660Xz1RAFE33iEyJR/view
rm -rf t2m
rm -rf kit

unzip t2m.zip
unzip kit.zip
echo -e "Cleaning\n"
rm t2m.zip
rm kit.zip

echo -e "Downloading done!"
```
### Get glove
```
echo -e "Downloading glove (in use by the evaluators, not by MDM itself)"
gdown --fuzzy https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing
rm -rf glove

unzip glove.zip
echo -e "Cleaning\n"
rm glove.zip

echo -e "Downloading done!"
```
### Get HumanML3d dataset
**HumanML3D** - Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:

```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

**KIT** - Download from [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git) (no processing needed this time) and the place result in `./dataset/KIT-ML`
</details>

<details>
  <summary><b>Action to Motion</b></summary>

**UESTC, HumanAct12** 
```bash
bash prepare/download_a2m_datasets.sh
```
</details>

<details>
  <summary><b>Unconstrained</b></summary>

**HumanAct12** 
```bash
bash prepare/download_unconstrained_datasets.sh
```
</details>

