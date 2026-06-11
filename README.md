# 영수증 → 엑셀 변환기

Tesseract `kor_best` 모델 기반 영수증 OCR → CSV 변환 도구.  
Flask 백엔드 + 순수 HTML 프론트엔드. **완전 무료, 로컬 처리.**

---

## 구조

```
receipt_ocr/
├── app.py               # Flask 백엔드 (OCR + 파싱)
├── templates/
│   └── index.html       # 프론트엔드 UI
└── README.md
```

---

## 설치

### 1. Tesseract 설치

**Ubuntu / Debian**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-kor
```

**macOS**
```bash
brew install tesseract
brew install tesseract-lang   # 한국어 포함 전체 언어팩
```

**Windows**
- https://github.com/UB-Mannheim/tesseract/wiki 에서 설치
- 설치 시 "Korean" 언어팩 체크
- `app.py` 상단 주석 처리된 경로 설정 해제:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```

### 2. kor_best 모델 다운로드 (인식률 핵심)

기본 `kor.traineddata`보다 정확도가 훨씬 높습니다.

```bash
# Ubuntu 기준 tessdata 경로
TESSDATA=/usr/share/tesseract-ocr/5/tessdata

sudo curl -L \
  "https://github.com/tesseract-ocr/tessdata_best/raw/main/kor.traineddata" \
  -o $TESSDATA/kor_best.traineddata

sudo curl -L \
  "https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata" \
  -o $TESSDATA/eng_best.traineddata
```

> macOS (Homebrew) tessdata 경로: `/usr/local/share/tessdata/`  
> Windows 경로: `C:\Program Files\Tesseract-OCR\tessdata\`

### 3. Python 패키지 설치

```bash
pip install flask flask-cors opencv-python pytesseract pillow
```

---

## 실행

```bash
cd receipt_ocr
python app.py
```

브라우저에서 `http://localhost:5000` 접속

---

## 전처리 파이프라인

벤치마크를 통해 선정한 최적 조합:

| 단계 | 처리 | 이유 |
|------|------|------|
| 1 | 2배 업스케일 (INTER_CUBIC) | Tesseract는 작은 텍스트에 취약 |
| 2 | Grayscale 변환 | 색상 정보 불필요, 처리 단순화 |
| 3 | Unsharp Mask | 획 선명도 향상 |
| 4 | `kor_best` 모델 | 기본 `kor`보다 정확도 대폭 향상 |
| 5 | PSM 4 + OEM 1 | 영수증 레이아웃에 최적 |

---

## 지출내용 자동 분류

영수증에서 결제 시간이 인식되면 자동으로 분류합니다:

| 시간대 | 분류 |
|--------|------|
| 06:00 ~ 10:59 | 조식 |
| 11:00 ~ 15:59 | 중식 |
| 16:00 ~ 23:59 | 석식 |

시간이 인식되지 않으면 품목명을 지출내용으로 사용합니다.

---

## 알려진 한계

- 손글씨, 흐릿한 사진, 극심한 기울기는 인식률이 낮습니다
- 인식 실패 항목은 UI에서 직접 수정하거나 수동 추가 기능을 사용하세요
- 📄 버튼으로 OCR 원문을 확인하고 직접 참고할 수 있습니다
