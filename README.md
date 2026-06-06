# machine_learning_license_plate_detect

Phat hien bien so xe bang pipeline co dien HOG + scikit-learn SVM.

Pipeline gom 2 buoc chinh:

- Class 1: tien xu ly patch anh, xam hoa, resize ve cung kich thuoc, trich xuat HOG,
  chuan hoa feature bang `StandardScaler`, tune tham so `C`, train `LinearSVC`.
- Class 2: dung sliding window quet anh tu tren xuong duoi, trai qua phai. Moi window
  duoc trich HOG va dua vao sklearn Pipeline de lay `decision_function`. Cac window
  vuot threshold duoc loc bang NMS, roi chon bbox co score cao nhat.

## Cau truc code

```text
plate_hog_svm/
  config.py          # cau hinh HOG, train SVM, NMS, detection
  features.py        # tien xu ly + HOG extractor
  dataset.py         # doc CSV train plate/non-plate
  training.py        # StandardScaler + LinearSVC + RandomizedSearchCV + metrics
  svm_utils.py       # load/save sklearn Pipeline, decision scores
  windows.py         # sinh sliding window
  nms.py             # Non-Maximum Suppression
  detector.py        # sliding window + HOG + sklearn SVM + NMS
  hard_negatives.py  # mining hard negatives theo batch
  io_utils.py        # doc/ghi anh, duyet thu muc anh
  report.py          # CSV/HTML report

train_plate_classifier.py  # CLI train Class 1
detect_plates.py           # CLI detect Class 2
mine_hard_negatives.py     # CLI mining hard negatives
```

## Cai dat

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chuan bi du lieu train

CSV train can co cot `label` va mot cot duong dan anh, vi du `candidate_path`.

```csv
candidate_path,label
data/plates/plate_001.png,1
data/background/bg_001.png,0
```

Hoac dung anh goc kem bbox:

```csv
image,x,y,w,h,label
images/train/car_001.png,120,220,180,55,1
images/train/car_001.png,20,30,180,55,0
```

Nhan duong: `1`, `plate`, `positive`, `bien_so`.
Nhan am: `0`, `-1`, `non_plate`, `background`, `negative`.

## Class 1: train HOG + sklearn SVM

```powershell
python train_plate_classifier.py --labels outputs/candidates/labels.csv --model models/plate_svm.joblib
```

Ket qua:

- `models/plate_svm.joblib`: sklearn Pipeline gom `StandardScaler` + `LinearSVC`.
- `models/plate_svm.json`: metadata gom HOG config, best C, classes, metrics.

Train script se in:

- accuracy
- precision
- recall
- F1
- F1 macro
- confusion matrix
- classification report
- best `C` tu `RandomizedSearchCV`

Tuy chinh C va class weight:

```powershell
python train_plate_classifier.py --labels outputs/candidates/labels.csv --c-values 0.01,0.1,1,10 --class-weight balanced
```

Neu ban da can bang dataset thu cong, co the dung:

```powershell
python train_plate_classifier.py --labels outputs/candidates/labels.csv --class-weight none
```

## Hard negative mining

Dung model da train de quet cac anh background khong co bien so, lay nhung window bi
model nham thanh bien so, luu thanh crop negative va CSV label.

```powershell
python mine_hard_negatives.py --negatives images/background --model models/plate_svm.joblib --output outputs/hard_negatives --score-threshold 0.5
```

Ket qua:

- `outputs/hard_negatives/crops`: crop hard negatives.
- `outputs/hard_negatives/labels.csv`: CSV co `label=0`, dung lai de retrain.
- `outputs/hard_negatives/hard_neg_features.npy`: backup HOG features.

Retrain voi hard negatives:

```powershell
python train_plate_classifier.py --labels outputs/candidates/labels.csv --hard-negatives outputs/hard_negatives/labels.csv --model models/plate_svm.joblib
```

Ghi chu: folder dua vao `--negatives` nen la anh khong chua bien so. Neu trong do co
bien so that, mining co the them nham bien so vao class negative.

## Class 2: sliding window detect + NMS

```powershell
python detect_plates.py --input images --model models/plate_svm.joblib --output outputs/plates --debug-output outputs/debug
```

Ket qua:

- `outputs/plates`: crop window bien so tot nhat.
- `outputs/debug`: anh goc co bbox debug.
- `outputs/detections.csv`: bbox, SVM score, confidence, so window da quet.
- `outputs/report.html`: trang HTML xem nhanh ket qua.

Chay nhanh 20 anh:

```powershell
python detect_plates.py --input images\val --limit 20 --model models/plate_svm.joblib
```

Tinh chinh detection:

```powershell
python detect_plates.py --input images --score-threshold 0.3 --iou-threshold 0.3 --top-k 1
```

Ghi chu:

- `--score-threshold`: score SVM toi thieu de giu candidate.
- `--iou-threshold`: nguong NMS de loai box trung nhau.
- `--top-k`: so detection toi da tra ve moi anh.
- Giam `--stride-ratio` se quet day hon nhung cham hon.
