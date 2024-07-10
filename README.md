# CLIPTrase

## [ECCV24] Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation

## 1. Introduction
> CLIP, as a vision-language model, has significantly advanced Open-Vocabulary Semantic Segmentation (OVSS) with its zero-shot capabilities. Despite its success, its application to OVSS faces challenges due to its initial image-level alignment training, which affects its performance in tasks requiring detailed local context. Our study delves into the impact of CLIP's [CLS] token on patch feature correlations, revealing a dominance of "global" patches that hinders local feature discrimination. To overcome this, we propose CLIPtrase, a novel training-free semantic segmentation strategy that enhances local feature awareness through recalibrated self-correlation among patches. This approach demonstrates notable improvements in segmentation accuracy and the ability to maintain semantic coherence across objects.
Experiments show that we are 22.3\% ahead of CLIP on average on 9 segmentation benchmarks, outperforming existing state-of-the-art training-free methods.

Full paper and supplementary materials: arxiv

### 1.1. Global Patch

![global patch](/images/reason.png)

### 1.2. Model Architecture

![model architecture](/images/model.svg)

## 2. Code

### 2.1. Environments

+ base environment: pytorch==1.12.1, torchvision==0.13.1 (CUDA11.3)
```
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
+ Detectron2 version: install detectron2==0.6 additionally
```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```
<br/>

### 2.2. Data preparation

+ We follow the detectron2 format of the datasets:

  The specific processing process can refer to [MaskFormer](https://github.com/facebookresearch/MaskFormer/blob/main/datasets/README.md) and [SimSeg](https://github.com/MendelXu/zsseg.baseline)

  Update  `configs/dataset_cfg.py` to your own path
```
datasets/
--coco/
----...
----val2017/
----stuffthingmaps_detectron2/
------val2017/

--VOC2012/
----...
----images_detectron2/
------val/
----annotations_detectron2/
------val/

--pcontext/
----...
----val/
------image/
------label/

----pcontext_full/
----...
----val/
------image/
------label/

--ADEChallengeData2016/
----...
----images/
------validation/
----annotations_detectron2/
------validation/

--ADE20K_2021_17_01/
----...
----images/
------validation/
----annotations_detectron2/
------validation/       
```

+ You also can use your own dataset, mask sure that it has `image` and `gt` file, and the value of each pixel in the gt image is its corresponding label.
<br/>

### 2.3. Global patch demo
+ We provide a demo of the global patch in the notebook `global_patch_demo.ipynb`, where you can visualize the global patch phenomenon mentioned in our paper.
<br/>

### 2.4. Training-free OVSS
+ Running with single GPU
```
python clip_self_correlation.py
```
+ Running with multiple GPUs in the detectron2 version
  
  Update: We provide detectron2 framework version, the clip state keys are modified and can be found [here](https://drive.google.com/file/d/1mZtNhYCJzL1jDfc4oO6e7rqbKiKSBGz9/view?usp=drive_link), you can download and put it in `outputs` folder.
  
  Note: The results of the d2 version are slightly different from those in the paper due to differences in preprocessing and resolution.
```
python -W ignore train_net.py --eval-only --config-file configs/clip_self_correlation.yaml --num-gpus 4 OUTPUT_DIR your_output_path MODEL.WEIGHTS your_model_path
```
+ Results

  single 3090, CLIP-B/16, evaluate in 9 situations on COCO, ADE, PASCAL CONTEXT, and VOC.

  Our results do not use any post-processing such as densecrf.

<table border=0 cellpadding=0 cellspacing=0 width=864 style='border-collapse:
 collapse;table-layout:fixed;width:648pt'>
 <col width=72 span=12 style='width:54pt'>
 <tr height=19 style='height:14.25pt'>
  <td height=19 class=xl65 style='height:14.25pt'></td>
  <td class=xl65></td>
  <td colspan=6 class=xl65>w/o. background</td>
  <td colspan=3 class=xl65>w. background</td>
 </tr>
 <tr height=19 style='height:14.25pt'>
  <td height=19 class=xl65 style='height:14.25pt'>Resolution</td>
  <td class=xl65>Metrics</td>
  <td class=xl65>coco171</td>
  <td class=xl65>voc20</td>
  <td class=xl65>pc59</td>
  <td class=xl65>pc459</td>
  <td class=xl65>ade150</td>
  <td class=xl65>adefull</td>
  <td class=xl65>coco80</td>
  <td class=xl65>voc21</td>
  <td class=xl65>pc60</td>
 </tr>
 <tr height=19 style='height:14.25pt'>
  <td rowspan=4 height=76 class=xl65 style='height:57.0pt'>224</td>
  <td class=xl65>pAcc</td>
  <td class=xl65>38.9</td>
  <td class=xl65>89.68</td>
  <td class=xl65>58.94</td>
  <td class=xl65>44.18</td>
  <td class=xl65>38.57</td>
  <td class=xl65>25.45</td>
  <td class=xl65>50.08</td>
  <td class=xl65>78.63</td>
  <td class=xl65>52.14</td>
 </tr>
 <tr height=19 style='height:14.25pt'>
  <td height=19 class=xl65 style='height:14.25pt'>mAcc</td>
  <td class=xl65>44.47</td>
  <td class=xl65>91.4</td>
  <td class=xl65>57.08</td>
  <td class=xl65>21.53</td>
  <td class=xl65>39.17</td>
  <td class=xl65>18.78</td>
  <td class=xl65>62.5</td>
  <td class=xl65>84.11</td>
  <td class=xl65>56.08</td>
 </tr>
 <tr height=19 style='height:14.25pt'>
  <td height=19 class=xl65 style='height:14.25pt'>fwIoU</td>
  <td class=xl65>26.87</td>
  <td class=xl65>82.49</td>
  <td class=xl65>45.28</td>
  <td class=xl65>35.22</td>
  <td class=xl65>27.96</td>
  <td class=xl65>18.99</td>
  <td class=xl65>38.19</td>
  <td class=xl65>67.67</td>
  <td class=xl65>37.61</td>
 </tr>
 <tr height=19 style='height:14.25pt'>
  <td height=19 class=xl65 style='height:14.25pt'>mIoU</td>
  <td class=xl65>22.84</td>
  <td class=xl65>80.95</td>
  <td class=xl65>33.83</td>
  <td class=xl65>9.36</td>
  <td class=xl65>16.35</td>
  <td class=xl65>6.31</td>
  <td class=xl65>43.56</td>
  <td class=xl65>50.88</td>
  <td class=xl65>29.87</td>
 </tr>
 <tr height=19 style='height:14.25pt'>
  <td rowspan=4 height=76 class=xl65 style='height:57.0pt'>336</td>
  <td class=xl65>pAcc</td>
  <td class=xl65>40.14</td>
  <td class=xl65>89.51</td>
  <td class=xl65>60.15</td>
  <td class=xl65>45.61</td>
  <td class=xl65>39.92</td>
  <td class=xl65>26.73</td>
  <td class=xl65>50.01</td>
  <td class=xl65>79.93</td>
  <td class=xl65>53.21</td>
 </tr>
 <tr height=19 style='height:14.25pt'>
  <td height=19 class=xl65 style='height:14.25pt'>mAcc</td>
  <td class=xl65>45.09</td>
  <td class=xl65>91.77</td>
  <td class=xl65>57.47</td>
  <td class=xl65>21.26</td>
  <td class=xl65>37.75</td>
  <td class=xl65>17.99</td>
  <td class=xl65>62.55</td>
  <td class=xl65>85.24</td>
  <td class=xl65>56.43</td>
 </tr>
 <tr height=19 style='height:14.25pt'>
  <td height=19 class=xl65 style='height:14.25pt'>fwIoU</td>
  <td class=xl65>27.96</td>
  <td class=xl65>82.15</td>
  <td class=xl65>46.64</td>
  <td class=xl65>36.66</td>
  <td class=xl65>29.17</td>
  <td class=xl65>20.3</td>
  <td class=xl65>38.24</td>
  <td class=xl65>69.1</td>
  <td class=xl65>38.76</td>
 </tr>
 <tr height=19 style='height:14.25pt'>
  <td height=19 class=xl65 style='height:14.25pt'>mIoU</td>
  <td class=xl65>24.06</td>
  <td class=xl65>81.2</td>
  <td class=xl65>34.92</td>
  <td class=xl65>9.95</td>
  <td class=xl65>17.04</td>
  <td class=xl65>5.89</td>
  <td class=xl65>44.84</td>
  <td class=xl65>53.04</td>
  <td class=xl65>30.79</td>
 </tr>
</table>

## Citation 
+ If you find this project useful, please consider citing:
```
@InProceedings{shao2024explore,
    title={Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation},
    author={Tong Shao and Zhuotao Tian and Hang Zhao and Jingyong Su},
    booktitle={European Conference on Computer Vision},
    organization={Springer},
    year={2024}
}
```


