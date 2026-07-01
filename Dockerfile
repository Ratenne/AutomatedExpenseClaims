# 1. 파이썬 베이스 이미지 지정
FROM python:3.11-slim

# 2. Tesseract OCR 엔진 및 다운로드를 위한 wget 설치 (언어 팩은 apt로 안 받음)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-kor \
    tesseract-ocr-eng \
    wget \
    && rm -rf /var/lib/apt/lists/*
    

# 3. 작업 디렉토리 설정
WORKDIR /app

# 🌟 [핵심 추가] 내 작업 공간에 tessdata 폴더를 만들고 진짜 언어 팩 직접 때려박기
RUN mkdir -p /app/tessdata
RUN wget -O /app/tessdata/kor.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/kor.traineddata
RUN wget -O /app/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
RUN wget -O /app/tessdata/osd.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/osd.traineddata

# 4. 파이썬 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 프로젝트 소스 코드 복사
COPY . .

# 🌟 [안전장치] 허깅페이스 컨테이너 권한 문제 방지를 위해 폴더 권한 전면 개방
RUN chmod -R 777 /app/tessdata

# 6. 허깅페이스 전용 포트 설정
EXPOSE 7860

# 7. 서버 실행 명령어 (포트를 7860으로 강제 주입)
CMD ["python", "app.py"]