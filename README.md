# machine_learning_license_plate_detect

Phat hien bien so xe bang pipeline co dien HOG + SVM.

Repo nay duoc tach thanh 2 buoc dung theo yeu cau:

- Class 1: tien xu ly patch anh, xam hoa, resize ve cung kich thuoc, trich xuat HOG,
  train Linear SVM de phan loai `plate` / `non_plate`.
- Class 2: dung cua so truot quet anh tu tren xuong duoi, trai qua phai. Moi window
  duoc trich HOG va dua vao SVM. Window co SVM confidence cao nhat se duoc chon lam
  bbox bien so.

## Cau truc code

```text
plate_hog_svm/
  config.py      # cau hinh HOG va sliding window
  features.py    # tien xu ly + HOG extractor
  dataset.py     # doc CSV train plate/non-plate
  training.py    # split data, train/evaluate Linear SVM
  svm_utils.py   # load/save SVM va metadata margin
  windows.py     # sinh sliding window va IoU
  detector.py    # Class 2: quet window bang SVM
  io_utils.py    # doc/ghi anh, duyet thu muc anh
  report.py      # CSV/HTML report

train_plate_classifier.py  # CLI train Class 1
detect_plates.py           # CLI detect Class 2
```

## Cai dat

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chuan bi du lieu train

CSV train can co cot `label` va mot cot duong dan anh, vi du `candidate_path`.

Vi du:

```csv
candidate_path,label
data/plates/plate_001.png,1
data/background/bg_001.png,0
```

Chap nhan nhan duong: `1`, `plate`, `positive`, `bien_so`.
Chap nhan nhan am: `0`, `-1`, `non_plate`, `background`, `negative`.

Neu CSV dung anh goc kem bbox thi co the dung cac cot:

```csv
image,x,y,w,h,label
images/train/car_001.png,120,220,180,55,1
images/train/car_001.png,20,30,180,55,0
```

## Class 1: train HOG + SVM

```powershell
python train_plate_classifier.py --labels outputs/candidates/labels.csv --model models/plate_svm.yml
```

Ket qua:

- `models/plate_svm.yml`: OpenCV Linear SVM.
- `models/plate_svm.json`: metadata HOG va huong raw margin de confidence cao hon nghia la giong bien so hon.

## Class 2: sliding window detect

```powershell
python detect_plates.py --input images --model models/plate_svm.yml --output outputs/plates --debug-output outputs/debug
```

Ket qua:

- `outputs/plates`: crop window bien so duoc SVM chon.
- `outputs/debug`: anh goc co bbox.
- `outputs/detections.csv`: bbox, SVM margin, confidence va so window da quet.
- `outputs/report.html`: trang HTML xem nhanh ket qua.

Chay nhanh 20 anh:

```powershell
python detect_plates.py --input images\val --limit 20
```

Tinh chinh sliding window:

```powershell
python detect_plates.py --input images --stride-ratio 0.2 --window-heights 32,48,64,96,128 --aspect-ratios 1.3,1.6,2.4,3.6,4.5
```

Ghi chu:

- `--score-threshold 0.0` nghia la chi chap nhan window nam phia positive cua SVM.
- Giam `--stride-ratio` se quet day hon nhung cham hon.
- Tang danh sach `--window-heights` hoac `--aspect-ratios` se bao phu nhieu dang bien so hon nhung cham hon.
