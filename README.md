# Wavelet-CUT (Overlap Consistency + Wavelet Regularization)

This repository is based on **CUT / FastCUT (Contrastive Unpaired Translation)** with two enhancements tailored for **WSI/large image tiling** scenarios:

- **Overlap Consistency**: During training, an additional "right neighbor overlapping patch" (A2/B2) is found for each patch. It enforces consistency in the overlapping regions of generated results to reduce stitching seams in whole images/WSIs.
- **Wavelet Regularization (Haar Wavelet)**: Applies constraints to **low-frequency (LL)** and **high-frequency (LH/HL/HH)** components separately in the wavelet domain, encouraging structural/morphological preservation while allowing style changes.

In addition, it includes utility scripts:
- `tools/wsi_tiler.py`: Splits WSIs/large images into coordinate-named patches using a sliding window (supports overlap and blank skipping)
- `tools/wsi_infer_fusion.py`: Sliding window inference + **Hann weighted fusion** back to whole images, further reducing stitching artifacts

---

## Table of Contents

- [Environment and Dependencies](#environment-and-dependencies)
- [Quick Start](#quick-start-your-verified-commands)
- [Data Organization and Naming Conventions](#data-organization-and-naming-conventions)
- [Output Directory Explanation](#output-directory-explanation)
- [Pretrained Model](#pretrained-model)
- [Acknowledgements and Citations](#acknowledgements-and-citations)

---

## Environment and Dependencies

If your environment can already run CUT, it should work directly with this repository.

Optional (only if processing WSI formats like `.svs/.ndpi/...`):
- `openslide-python` (Python package)
- System OpenSlide dynamic library (install according to your OS)

For a one-click installation of Python packages (including optional WSI dependencies), run the following command:
```bash
pip install -r requirements.txt
```

---

## Quick Start
To verify the integrity of the model and ensure the entire pipeline runs smoothly without errors, a small-scale dataset has been created at `datasets/smalltest`. The following commands are all verified with this dataset.

### 1) smalltest Training 
```bash
python train.py --dataroot "./datasets/smalltest" --name smalltest_CUT --CUT_mode CUT --phase small_train --display_id -1 --num_threads 0
```
#### Optional parameters for full version train.py

##### 1.1 Core General Optional Parameters

- `--direction AtoB` / `--direction BtoA`: Specify the image translation direction (default is usually `AtoB`)
- `--display_id -1`: Disable Visdom visualization (to avoid errors in environments without visualization support)
- `--num_threads 0`: Set the number of data loading threads to 0 (stable first choice for Windows systems; can be set to 4/8, etc. for Linux/macOS)
- `--preprocess none`: Disable additional image preprocessing (no resizing or cropping)
- `--load_size 512`: Set the image loading size to 512×512
- `--crop_size 512`: Set the image cropping size to 512×512

##### 1.2 Overlap Consistency Exclusive Optional Parameters

- `--use_overlap_pair`: Enable the overlap consistency constraint; the dataset returns right neighbor overlapping patches (A2/B2)
- `--patch_size 512`: Patch size used for neighbor search, must be consistent with `--tile_size` in `wsi_tiler.py`
- `--overlap_ratio 0.25`: Overlap ratio, must be consistent with `--overlap_ratio` in `wsi_tiler.py` (determines the sliding stride)
- `--lambda_OL 1.0`: Weight of the Overlap loss (default: 0; only effectively enabled when the value is > 0)

##### 1.3 Wavelet Regularization (Haar Wavelet) Exclusive Optional Parameters

- `--lambda_W 1.0`: Total weight of the Wavelet loss (default: 0; only effectively enabled when the value is > 0)
- `--lambda_low 2.0`: Constraint strength for low-frequency (LL) components
- `--lambda_high 0.1`: Constraint strength for high-frequency (LH/HL/HH) components
- `--wavelet_luma_only`: Calculate the Wavelet loss only on the luminance channel (improves stability and reduces color jitter)

### 2) smalltest Testing 
```bash
python test.py --dataroot "./datasets/smalltest" --name smalltest_CUT --CUT_mode CUT --phase small_train --num_threads 0
```
#### Optional parameters for full version test.py:

- `--direction AtoB` / `--direction BtoA`: Specify the image translation direction, which must be consistent with the setting used during training.
- `--epoch latest`: Specify the epoch of the model to load (default is `latest`; a specific number such as `--epoch 100` can also be specified).
- `--results_dir "./results"`: Specify the output directory for test results (default is `./results/`).
- `--num_test N`: Limit the number of test images (N is a specific number; if not set, all images in the test set will be processed).
- `--num_threads 0`: Set the number of data loading threads to 0 (stable first choice for Windows systems).
- 
### 3) WSI/Large Image Tiling
```bash
python wsi_tiler.py --sourceA "./data/WSI/A" --sourceB "./data/WSI/B" --dataroot "./datasets/medical_tiled" --phase train --tile_size 512 --overlap_ratio 0.25 --skip_blank
```
#### Optional parameters for full version wsi_tiler.py & wsi_infer_fusion.py:

##### 3.1 WSI Tiling (wsi_tiler.py) Optional Parameters

- `--level 0`: WSI pyramid level (default is 0, corresponding to the highest resolution).
- `--out_format png|jpg|jpeg|tif|tiff`: Specify the output image format for tiled patches (default is png).
- `--max_patches_per_slide 10000000`: Maximum number of patches allowed per WSI/large image.
- `--skip_blank`: Skip blank patches (reduces invalid data, recommended to enable).
- `--blank_white_ratio 0.8`: White pixel ratio threshold for blank patch detection (works with `--skip_blank`).
- `--verbose`: Print detailed log information during the tiling process (facilitates debugging).

##### 3.2 WSI Inference & Fusion (wsi_infer_fusion.py) Optional Parameters

- `--level 0`: WSI pyramid level (default is 0, corresponding to the highest resolution).
- `--tile_size 512`: Sliding window tile size for inference, must be consistent with `--patch_size` used during training.
- `--overlap_ratio 0.25`: Overlap ratio of the inference sliding window, must be consistent with the parameters in `wsi_tiler.py` and those used during training.
- `--gpu_ids 0`: Specify the GPU ID to use (set to `0,1` for multiple GPUs; set to `-1` for CPU).
- `--input "<path/to/wsi_or_big_img>"`: Specify the path to the input WSI/large image.
- `--output "<path/to/output_img>"`: Specify the output path for the fused whole image.

## Data Organization and Naming Conventions

### 1) Data Source
Relevant datasets and pre-trained models can be accessed via: [https://drive.google.com/drive/folders/1fNf-F_aplm6ACJTWO1vGqbb-DdaP4K_r](https://drive.google.com/drive/folders/1fNf-F_aplm6ACJTWO1vGqbb-DdaP4K_r)

### 2) CUT Unaligned Data Structure
Both training and testing follow this structure:

```plaintext
<<dataroot>/
  <phase>A/
  <phase>B/
```
### 3) For Enabling Overlap Consistency: Patch Filenames Must Contain Coordinates

When `--use_overlap_pair` is enabled, the dataset parses coordinates from filenames to find right neighbors:

```plaintext
<slide_id>_x<X>_y<Y>.png
```
tools/wsi_tiler.py automatically exports patches following this convention.

If your data does not use this naming format, --use_overlap_pair will still run, but A2/B2 will not be generated if right neighbors are not found, and the Overlap loss will be effectively disabled.

## Output Directory Explanation

### Training Output (checkpoints)
Default path:
```plaintext
.\checkpoints\<name>\
```
Typically contains:
- Network weights (e.g., latest_net_G.pth, etc.)
- Training configuration saves (*_opt.txt)

### Testing Output (results)
Default path:
```plaintext
.\results\<name>\
```

## Pretrained Model
Due to the large size of the pre-trained weights, we have hosted the data (including pre-trained weights) on Baidu Netdisk.
- Link: https://pan.baidu.com/s/1l3d6pTYo21MZbmLaB-yJpQ
- Key: 7P33

## Acknowledgements and Citations

CUT / FastCUT (Contrastive Unpaired Translation, ECCV 2020) serves as the base framework and training pipeline for this repository.

This repository extends the original work by adding three key components:
- Overlap Consistency (overlapping region consistency constraint)
- Haar Wavelet Regularization (wavelet-domain structural regularization)
- WSI/large image tiling and Hann-window fusion inference scripts

For academic papers, reports, or derivative works, we recommend citing the original CUT paper and explicitly noting the use of the above extensions in the method description.


