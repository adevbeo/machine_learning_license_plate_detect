# machine_learning_license_plate_detect

Tach bien so xe tu anh bang xu ly anh co dien voi OpenCV. Pipeline nay khong dung
model deep learning co san: no tim vung co dac trung giong bien so bang tang tuong
phan, blackhat morphology, bien canny/sobel, contour va bo loc hinh hoc + mat do
ky tu.

## Cai dat

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chay thu

Xu ly toan bo thu muc `images`:

```powershell
python detect_plates.py --input images --output outputs/plates --debug-output outputs/debug
```

Mac dinh OCR chay che do `hybrid`: template/HOG cuc bo + EasyOCR neu EasyOCR cho
chuoi hop format bien so hon. Lan chay EasyOCR dau tien se tai model nhan dang.

Chay nhanh 20 anh dau tien:

```powershell
python detect_plates.py --input images\val --limit 20
```

Chi dung OCR template co dien, khong dung EasyOCR:

```powershell
python detect_plates.py --input images --ocr-engine template
```

Ket qua:

- `outputs/plates`: anh crop bien so.
- `outputs/debug`: anh goc co ve bbox debug.
- `outputs/detections.csv`: toa do bbox, diem tin cay, `plate_text` va duong dan crop.
- `outputs/chars`: crop tung ky tu da tach ra.
- `outputs/characters.csv`: metadata tung ky tu, ky tu du doan va cot `label` de sua.

## Xuat du lieu de train buoc sau

Detector co the luu nhieu vung ung vien moi anh, kem feature va cot nhan trong
CSV de review thu cong:

```powershell
python detect_plates.py --input images --output outputs/plates --debug-output outputs/debug --report outputs/detections.csv --export-candidates outputs/candidates --candidates-per-image 8
```

Ket qua:

- `outputs/candidates`: crop cac vung ung vien.
- `outputs/candidates/labels.csv`: metadata + feature cua tung crop.
- Cot `suggested_label` la goi y tu score hien tai; can tu dien cot `label`.
- Dien `label=1` neu crop dung la bien so, `label=0` neu khong phai bien so.

Neu muon lay tat ca ung vien thay vi top 8:

```powershell
python detect_plates.py --input images --export-candidates outputs/candidates --candidates-per-image 0
```

Sau khi da gan nhan, train classifier bien-so/khong-bien-so bang HOG + OpenCV SVM:

```powershell
python train_plate_classifier.py --labels outputs/candidates/labels.csv --model models/plate_svm.yml
```

## Train OCR ky tu

OCR mac dinh dung template/HOG co dien nen chi la baseline. De doc dung tren bo anh
cua ban, hay sua cot `label` trong `outputs/characters.csv`:

- `char`: ky tu du doan hien tai.
- `label`: ky tu dung do ban dien lai, vi du `3`, `0`, `F`.

Train model ky tu:

```powershell
python train_char_classifier.py --labels outputs/characters.csv --model models/char_knn.npz
```

Chay lai detector voi model OCR da train:

```powershell
python detect_plates.py --input images --output outputs/plates --debug-output outputs/debug --report outputs/detections.csv --chars-output outputs/chars --char-report outputs/characters.csv --char-model models/char_knn.npz
```

Vi repo hien chua co annotation bbox/chu so that, detector va OCR template chi la
baseline. Phan export candidate + character CSV tao du lieu de train model loc vung
bien so va OCR ky tu cho buoc sau.

.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python detect_plates.py --input images --output outputs\plates --debug-output outputs\debug --report outputs\detections.csv --chars-output outputs\chars --char-report outputs\characters.csv --ocr-engine hybrid