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
RUN pip install --no-cache-dir flask pytesseract Pillow

# 5. 프로젝트 소스 코드 복사
COPY . .

# 6. 포트 설정 (Render는 기본적으로 10000 포트를 많이 씁니다)
EXPOSE 10000

# 7. 서버 실행 명령어 (app.py 내부에서 host='0.0.0.0', port=10000으로 구동되게 설정 필요)
CMD ["python", "app.py"]