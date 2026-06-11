import os
import re
import json
import shutil
import cv2
import numpy as np
import pytesseract
import platform
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ── Tesseract 경로 설정 ──
def _find_tesseract_cmd() -> str:
    candidates = []
    found = shutil.which('tesseract')
    if found:
        candidates.append(found)

    if platform.system() == 'Windows':
        candidates.extend([
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ])

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ''


if platform.system() == 'Windows':
    local_tessdata = os.path.join(os.path.dirname(__file__), 'tessdata')
    TESSDATA_DIR = local_tessdata if os.path.exists(local_tessdata) else r'C:\Program Files\Tesseract-OCR\tessdata'
else:
    TESSDATA_DIR = '/usr/share/tesseract-ocr/5/tessdata'

TESSERACT_CMD = _find_tesseract_cmd()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR

def _pick_lang(*names: str) -> str:
    for name in names:
        if os.path.exists(os.path.join(TESSDATA_DIR, f'{name}.traineddata')):
            return name
    return ''


kor_lang = _pick_lang('kor_best', 'kor')
eng_lang = _pick_lang('eng_best', 'eng')
if kor_lang and eng_lang:
    TESS_LANG = f'{kor_lang}+{eng_lang}'
elif kor_lang:
    TESS_LANG = kor_lang
elif eng_lang:
    TESS_LANG = eng_lang
else:
    TESS_LANG = ''

# PSM별 config (멀티패스에서 사용)
def _cfg(psm: int) -> str:
    return f'--psm {psm} --oem 1 --tessdata-dir {TESSDATA_DIR}'


# ──────────────────────────────────────────
#  이미지 전처리
#  벤치마크 결과: 2x upscale → grayscale → unsharp mask
# ──────────────────────────────────────────
def preprocess(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")

    # 1) 2배 업스케일 (최대 4000px 제한)
    h, w = img.shape[:2]
    scale = min(2.0, 4000 / max(h, w))
    img = cv2.resize(img, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_CUBIC)

    # 2) Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3) Unsharp mask — 획 선명도 향상
    blurred = cv2.GaussianBlur(gray, (0, 0), 2)
    sharpened = cv2.addWeighted(gray, 1.3, blurred, -0.3, 0)

    return sharpened


# ──────────────────────────────────────────
#  멀티패스 OCR
#  PSM 4 (단일 컬럼 텍스트)
#  PSM 6 (단일 균일 블록)
#  PSM 11 (sparse text - 흩어진 텍스트)
#  → 세 결과를 합산해 누락 문자 보완
# ──────────────────────────────────────────
def run_ocr(img: np.ndarray) -> str:
    if not TESSERACT_CMD:
        raise RuntimeError('Tesseract 실행 파일을 찾을 수 없습니다')
    if not TESS_LANG:
        raise RuntimeError('설치된 Tesseract 언어팩을 찾을 수 없습니다')
    t4  = pytesseract.image_to_string(img, lang=TESS_LANG, config=_cfg(4))
    t6  = pytesseract.image_to_string(img, lang=TESS_LANG, config=_cfg(6))
    t11 = pytesseract.image_to_string(img, lang=TESS_LANG, config=_cfg(11))
    # 세 결과를 구분선으로 합산 (파싱 시 전체 텍스트에서 검색)
    return t4 + '\n' + t6 + '\n' + t11


# ──────────────────────────────────────────
#  파싱 유틸
# ──────────────────────────────────────────
def extract_date(lines):
    for line in lines:
        m = re.search(r'(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})', line)
        if m:
            return m.group(2).lstrip('0') or '0', m.group(3).lstrip('0') or '0'
        m = re.search(r'\b(\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})\b', line)
        if m:
            return m.group(2).lstrip('0') or '0', m.group(3).lstrip('0') or '0'
        m = re.search(r'(\d{1,2})월\s*(\d{1,2})일', line)
        if m:
            return m.group(2).lstrip('0') or '0', m.group(3).lstrip('0') or '0'
        m = re.search(r'(\d{1,2})년\s*(\d{1,2})월\s*(\d{1,2})일', line)        
        if m:
            return m.group(2), m.group(3)
    return '', ''


def extract_time_meal(lines):
    """결제 시간 → 조식/중식/석식"""
    for line in lines:
        m = re.search(r'오전\s*(\d{1,2})[:\.](\d{2})', line)
        if m:
            h = int(m.group(1))
            return '조식' if h < 10 else '중식'
        m = re.search(r'오후\s*(\d{1,2})[:\.](\d{2})', line)
        if m:
            h = int(m.group(1)) + 12
            return '중식' if h < 17 else '석식'
        m = re.search(r'\b(AM|am)\s*(\d{1,2})[:\.](\d{2})', line)
        if m:
            h = int(m.group(2))
            return '조식' if h < 10 else '중식'
        m = re.search(r'\b(PM|pm)\s*(\d{1,2})[:\.](\d{2})', line)
        if m:
            h = int(m.group(2)) + 12
            return '중식' if h < 17 else '석식'
        # 24시간 HH:MM(:SS)
        m = re.search(r'\b(\d{1,2}):(\d{2})(?::\d{2})?\b', line)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                if  6 <= h < 11: return '조식'
                if 11 <= h < 16: return '중식'
                if 16 <= h <= 23: return '석식'
    return ''


def extract_amount(lines, full_text):
    """합계/총액 키워드 우선, fallback은 전체 텍스트 최댓값"""
    keywords = ['합계', '총액', '결제금액', '받을금액', '청구금액', '결제액',
                'total', 'TOTAL', '지불금액', '합    계', '합     계',
                '합  계', '합   계', '원']
    for line in lines:
        if any(k in line for k in keywords):
            nums = re.findall(r'[\d,]+', line)
            candidates = [int(n.replace(',', '')) for n in nums
                          if len(n.replace(',', '')) >= 3]
            if candidates:
                return str(max(candidates))
    nums = re.findall(r'[\d,]+', full_text)
    candidates = [int(n.replace(',', '')) for n in nums
                  if 100 <= int(n.replace(',', '')) <= 9_999_999]
    return str(max(candidates)) if candidates else '0'


def extract_vendor(lines):
    """상호명 추출"""
    vendor_keys = ['상호', '가맹점', '점명', '상점명', '점 명', '입금처', '업체', '매장']
    for line in lines:
        if any(k in line for k in vendor_keys):
            parts = re.split(r'[:：]\s*', line, maxsplit=1)
            if len(parts) == 2:
                candidate = parts[1].strip()
                if re.search(r'[\uAC00-\uD7A3a-zA-Z]{2,}', candidate):
                    return candidate[:20]

    # fallback 1: 지점명 패턴 (○○점, ○○마트 등)
    store_pat = re.compile(
        r'[\uAC00-\uD7A3a-zA-Z0-9]{2,}'
        r'(?:점|마트|편의점|슈퍼|식당|카페|베이커리|약국|병원|주유소|센터|치킨|피자|버거)'
    )
    for line in lines:
        m = store_pat.search(line)
        if m and len(line) <= 25:
            return m.group(0)

    # fallback 2: 한글 2자 이상 + 짧고 깔끔한 라인
    skip = {'사업자', '대표자', '대표', '주소', '전화', 'TEL', 'FAX',
            '영수증', '감사', '날짜', '시간', '결제', '합계', '총', '일시'}
    for line in lines:
        if (re.search(r'[\uAC00-\uD7A3]{2,}', line)
                and 2 <= len(line) <= 20
                and not re.match(r'^\d', line)
                and ':' not in line and '：' not in line
                and not any(k in line for k in skip)):
            return line.strip()
    return ''


def extract_items(lines, vendor):
    """구매 품목 추출 (금액·잡라인 제외)"""
    skip_kw = ['날짜', '시간', '주소', '사업자', '대표', '전화', '영수증',
               '결제', '합계', '총', '번호', '단말', 'TEL', 'FAX', '등록',
               '승인', '카드', '현금', '거스름', '잔액', '받은', '감사',
               '과세', '부가세', '공급가', '일시']
    result = []
    seen = set()
    for line in lines:
        if not line or re.match(r'^[\d,\s\-─=]+$', line):
            continue
        if any(k in line for k in skip_kw):
            continue
        if line == vendor:
            continue
        if re.match(r'\d{4}[.\-/]\d', line):
            continue
        if not re.search(r'[\uAC00-\uD7A3a-zA-Z]', line):
            continue
        name = re.sub(r'\s+[\d,]+\s*원?\s*$', '', line).strip()
        name = re.sub(r'^\d+[.)\s]+', '', name).strip()
        if name and len(name) >= 2 and name not in seen:
            seen.add(name)
            result.append(name)
    return result[:3]


# ──────────────────────────────────────────
#  메인 파싱
# ──────────────────────────────────────────
def parse_receipt(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    month, day  = extract_date(lines)
    meal        = extract_time_meal(lines)
    amount      = extract_amount(lines, text)
    vendor      = extract_vendor(lines)
    items       = extract_items(lines, vendor)
    description = meal if meal else (', '.join(items) if items else '')

    return {
        'month':       month,
        'day':         day,
        'vendor':      vendor or '인식 실패',
        'description': description or '수동 입력 필요',
        'amount':      amount,
        'raw_text':    text,
    }


# ──────────────────────────────────────────
#  라우트
# ──────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 없습니다'}), 400

    if not TESSERACT_CMD:
        return jsonify({
            'error': 'Tesseract를 찾을 수 없습니다',
            'detail': 'Windows에서는 C:\\Program Files\\Tesseract-OCR\\tesseract.exe 설치 또는 PATH 등록이 필요합니다.'
        }), 503

    img_bytes = request.files['image'].read()
    try:
        if not img_bytes:
            return jsonify({'error': '이미지 데이터가 비어 있습니다'}), 400
        processed = preprocess(img_bytes)
        text      = run_ocr(processed)          # ← 멀티패스
        result    = parse_receipt(text)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except pytesseract.TesseractNotFoundError as e:
        return jsonify({'error': 'Tesseract 실행 파일을 찾을 수 없습니다', 'detail': str(e)}), 503
    except pytesseract.TesseractError as e:
        return jsonify({'error': 'OCR 처리 실패', 'detail': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
