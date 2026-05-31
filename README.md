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

Chay nhanh 20 anh dau tien:

```powershell
python detect_plates.py --input images\val --limit 20
```

Ket qua:

- `outputs/plates`: anh crop bien so.
- `outputs/debug`: anh goc co ve bbox debug.
- `outputs/detections.csv`: toa do bbox, diem tin cay va duong dan crop.

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

Vi repo hien chua co annotation bbox/chu so, day la detector xu ly anh khong
supervised. Phan export candidate o tren tao du lieu de train model loc vung bien
so cho buoc sau; neu can doc thanh chuoi bien so, can them buoc OCR hoac bo du
lieu ky tu de train classifier rieng.
