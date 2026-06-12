# machine_learning_license_plate_detect

Phat hien bien so xe bang pipeline co dien HOG + LinearSVM.

Pipeline gom 2 buoc chinh:

- Class 1: tien xu ly patch anh, xam hoa, resize ve cung kich thuoc, trich xuat HOG,
  chuan hoa feature bang `StandardScaler`, tune tham so `C`, train `LinearSVM`.
- Class 2: dung sliding window quet anh tu tren xuong duoi, trai qua phai. Moi window
  duoc trich HOG va dua vao sklearn Pipeline de lay `decision_function`. Cac window
  vuot threshold duoc loc bang NMS, roi chon bbox co score cao nhat.

## Cau truc code

```text
plate_hog_svm/
  config.py          # cau hinh HOG, train LinearSVM, NMS, detection
  features.py        # tien xu ly + HOG extractor
  dataset.py         # doc CSV train plate/non-plate
  training.py        # StandardScaler + LinearSVM + RandomizedSearchCV + metrics
  svm_utils.py       # load/save sklearn Pipeline, decision scores
  windows.py         # sinh sliding window
  nms.py             # Non-Maximum Suppression
  detector.py        # sliding window + HOG + LinearSVM + NMS
  hard_negatives.py  # mining hard negatives theo batch
  io_utils.py        # doc/ghi anh, duyet thu muc anh
  report.py          # CSV/HTML report
  visualization.py   # HOG visualize=True va PCA/t-SNE scatter plot

train_plate_classifier.py  # CLI train Class 1
prepare_auto_training_data.py # tu sinh weak labels tu anh train co san
run_demo_pipeline.py       # chay demo end-to-end: labels -> train -> detect
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

## Class 1: train HOG + LinearSVM

Neu repo chi co san anh xe trong `images/train` va chua co labels/model, co the
chay demo end-to-end:

```powershell
python run_demo_pipeline.py --train-sample-size 200 --detect-sample-size 20
```

Script nay se:

1. Tu sinh weak labels tu `images/train` bang OpenCV plate proposal.
2. Train model HOG + `StandardScaler` + `LinearSVM`.
3. Sinh visualize HOG va scatter PCA cua feature HOG.
4. Dung model vua train de scan/crop bien so trong `images/val`.

Ket qua demo:

- `outputs/labels.csv`: weak labels de train LinearSVM.
- `models/plate_svm.joblib`: model ML da train.
- `outputs/visualize/hog_visualization.png`: anh HOG visual.
- `outputs/visualize/hog_feature_embedding.png`: scatter PCA 2D cua HOG feature.
- `outputs/plates`: crop bien so detect duoc.
- `outputs/output`: anh goc da ke bbox `LinearSVM` de kiem tra crop co trung bien so khong.
- `outputs/detections.csv`, `outputs/report.html`: bao cao ket qua.

Mac dinh demo moi lan chay se lay ngau nhien tap anh moi va xoa output cu trong
`outputs`: 100 anh de tao training labels, va 20 anh de detect/crop demo.
Neu muon co dinh dung cung tap anh de thuyet trinh:

```powershell
python run_demo_pipeline.py --seed 42
```

Tang so anh train de model hoc nhieu hon nhung van chi output 20 anh demo:

```powershell
python run_demo_pipeline.py --train-sample-size 200 --detect-sample-size 20
```

Neu muon ca train va detect cung mot so anh:

```powershell
python run_demo_pipeline.py --sample-size 50
```

Neu muon giu output cu:

```powershell
python run_demo_pipeline.py --keep-output
```

Neu muon chay nhanh va khong sinh visualize:

```powershell
python run_demo_pipeline.py --train-sample-size 200 --detect-sample-size 20 --no-visualize
```

Neu muon chay tung buoc de thuyet trinh:

```powershell
python prepare_auto_training_data.py --input images\train --output outputs\labels.csv
python train_plate_classifier.py --labels outputs\labels.csv --model models\plate_svm.joblib
python detect_plates.py --input images\val --limit 20 --model models\plate_svm.joblib --plates-output outputs\plates --output outputs\output
```

```powershell
python train_plate_classifier.py --labels outputs/labels.csv --model models/plate_svm.joblib
```

Ket qua:

- `models/plate_svm.joblib`: sklearn Pipeline gom `StandardScaler` + `LinearSVM`.
- `models/plate_svm.json`: metadata gom HOG config, best C, classes, metrics.
- `outputs/visualize/hog_visualization.png`: anh visualize HOG bang `skimage.feature.hog(..., visualize=True)`.
- `outputs/visualize/hog_feature_embedding.png`: scatter plot HOG features sau khi giam chieu bang PCA.

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
python train_plate_classifier.py --labels outputs/labels.csv --c-values 0.01,0.1,1,10 --class-weight balanced
```

Neu ban da can bang dataset thu cong, co the dung:

```powershell
python train_plate_classifier.py --labels outputs/labels.csv --class-weight none
```

### Visualize: HOG va PCA/t-SNE scatter

Mac dinh truoc khi train LinearSVM, script se sinh 2 anh visualize:

- HOG visualization: lay mot so crop label, tien xu ly giong pipeline, dung
  `skimage.feature.hog(..., visualize=True)` de ve anh HOG bang matplotlib.
- Feature scatter: chuan hoa vector HOG, giam chieu xuong 2D bang PCA, roi ve
  scatter plot mau xanh cho `plate` va mau do cho `non_plate`.

Chay voi mac dinh:

```powershell
python train_plate_classifier.py --labels outputs/labels.csv
```

Dung t-SNE hoac ve 3D:

```powershell
python train_plate_classifier.py --labels outputs/labels.csv --feature-plot-method tsne --feature-plot-dims 2
python train_plate_classifier.py --labels outputs/labels.csv --feature-plot-method pca --feature-plot-dims 3
```

Tuy chinh/tat visualize:

```powershell
python train_plate_classifier.py --labels outputs/labels.csv --hog-visualization-limit 24
python train_plate_classifier.py --labels outputs/labels.csv --hog-visualization "" --feature-plot ""
```

Neu scatter plot co 2 cum xanh/do tach nhau ro, dac trung HOG dang phan biet
tot hai class va LinearSVM se de train hon. Neu 2 cum chong lap nhieu, can xem lai crop,
label, tham so HOG, hoac bo sung hard negatives.

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
python train_plate_classifier.py --labels outputs/labels.csv --hard-negatives outputs/hard_negatives/labels.csv --model models/plate_svm.joblib
```

Ghi chu: folder dua vao `--negatives` nen la anh khong chua bien so. Neu trong do co
bien so that, mining co the them nham bien so vao class negative.

## Class 2: sliding window detect + NMS

```powershell
python detect_plates.py --input images --model models/plate_svm.joblib --plates-output outputs/plates --output outputs/output
```

Ket qua:

- `outputs/plates`: crop window bien so tot nhat.
- `outputs/output`: anh goc da ke bbox `LinearSVM`.
- `outputs/detections.csv`: bbox, LinearSVM score, confidence, so window da quet.
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

- `--score-threshold`: score LinearSVM toi thieu de giu candidate.
- `--iou-threshold`: nguong NMS de loai box trung nhau.
- `--top-k`: so detection toi da tra ve moi anh.
- Giam `--stride-ratio` se quet day hon nhung cham hon.
