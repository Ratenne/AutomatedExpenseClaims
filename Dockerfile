# 1. 파이썬 베이스 이미지 지정
FROM python:3.11-slim

# 2. Tesseract OCR 및 한국어/영어 패키지 설치
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-kor \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 파이썬 라이브러리 설치 (가상환경 대신 서버 자체에 설치)
# 만약 requirements.txt가 아직 없다면 pip install flask pytesseract 등 직접 명시해도 됩니다.
# RUN pip install --no-cache-dir flask pytesseract Pillow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 프로젝트 소스 코드 복사
COPY . .

# 6. 허깅페이스 전용 포트 설정 (7860으로 고정)
EXPOSE 7860

# 7. 서버 실행 명령어 (포트를 7860으로 강제 주입)
CMD ["python", "app.py"]