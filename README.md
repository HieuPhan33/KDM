# KDM
Efficient Hyperspectral Image Segmentation for Biosecurity Scanning Using Knowledge Distillation from Multi-head Teacher

### 1. Install Python packages:
* pytorch
* segmentation_models.pytorch
* comet-ml
### 2. Data Preparation:
Download data at documents.uow.edu.au/ phung/Bio-HSI.html.

### 3. Train SGR-Net teacher:
Edit configure files in configs/teacher/hsi_sgrnet_res101.yml and point to your data folder.
`python train_teacher.py --config configs/teacher/hsi_sgrnet_res101.yml`

### 4. Train student via KD from Multi-head Teacher (KDM):
Edit pretrained_file in configs/hsi_res101_kdm.yml with the path to the teacher checkpoint file stored in `experiments` folder.

You can download the pretrained teacher model [here](https://uowmailedu-my.sharepoint.com/:u:/r/personal/vmhp806_uowmail_edu_au/Documents/best_model.pth?csf=1&web=1&e=eFgUTl)

`python train_kd.py --config hsi_res101_kdm.yml`

If you find this code useful, please cite our paper:

```
@article{phan2022efficient,
  title={Efficient Hyperspectral Image Segmentation for Biosecurity Scanning Using Knowledge Distillation from Multi-head Teacher},
  author={Phan, Minh Hieu and Phung, Son Lam and Luu, Khoa and Bouzerdoum, Abdesselam},
  journal={Neurocomputing},
  year={2022},
  publisher={Elsevier}
}
```