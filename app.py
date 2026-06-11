import os
import re
import json
import logging
import cv2
import numpy as np
import pytesseract
import platform
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.logger.setLevel(logging.INFO)

# 프로젝트 루트의 tessdata를 우선 사용한다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESSDATA_DIR = os.path.join(BASE_DIR, 'tessdata')

# ── Tesseract 경로 설정 ──
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

os.environ.setdefault('TESSDATA_PREFIX', TESSDATA_DIR)

# 설치된 언어 팩 자동 탐색 및 선택
def _resolve_tess_lang(tessdata_dir: str) -> str:
    """사용 가능한 Tesseract 언어 팩을 우선순위별로 선택"""
    available = {
        os.path.splitext(name)[0]
        for name in os.listdir(tessdata_dir)
        if name.endswith('.traineddata')
    }
    
    for candidate in (
        ('kor_best', 'eng_best'),
        ('kor', 'eng'),
        ('kor_best',),
        ('kor',),
        ('eng_best',),
        ('eng',),
    ):
        if all(lang in available for lang in candidate):
            return '+'.join(candidate)
    
    raise RuntimeError(
        f'사용 가능한 Tesseract 언어 팩을 찾을 수 없습니다: {tessdata_dir}'
    )

TESS_LANG = _resolve_tess_lang(TESSDATA_DIR)

# PSM별 config (멀티패스에서 사용)
def _cfg(psm: int) -> str:
    return f'--psm {psm} --oem 1 --tessdata-dir {TESSDATA_DIR}'


# ──────────────────────────────────────────
#  이미지 전처리
#  벤치마크 결과: 2x upscale → grayscale → unsharp mask
# ──────────────────────────────────────────
def preprocess(img_bytes: bytes):
    """
    반환값: (raw_gray, sharpened)
      - raw_gray  : unsharp 미적용 grayscale (vendor 크롭 재OCR용)
      - sharpened : unsharp 적용 (멀티패스 OCR + bounding box 추출용)
    """
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

    # 3) Dark mode 자동 감지 → 반전
    if np.mean(gray) < 128:
        gray = cv2.bitwise_not(gray)

    # 4) Unsharp mask
    blurred   = cv2.GaussianBlur(gray, (0, 0), 2)
    sharpened = cv2.addWeighted(gray, 1.3, blurred, -0.3, 0)

    return gray, sharpened   # (raw, sharp) 튜플 반환


# ──────────────────────────────────────────
#  vendor 영역 크롭 재OCR
#  bounding box로 vendor 키워드 라인을 찾아
#  오른쪽 값 영역만 raw_gray 3x + PSM7 재인식
#  → 토스앱처럼 얇은 UI 폰트에서 한글 인식률 향상
# ──────────────────────────────────────────
def _reocr_vendor_crop(raw_gray: np.ndarray, sharp: np.ndarray) -> list:
    """vendor 키워드 라인의 값 영역을 3x+PSM7로 재OCR, 후보 리스트 반환"""
    vendor_keys = ['적요', '입금처', '가맹점', '상호', '점명']
    h, w = raw_gray.shape

    try:
        data = pytesseract.image_to_data(
            sharp, lang=TESS_LANG, config=_cfg(6),
            output_type=pytesseract.Output.DATAFRAME
        )
        # text 컬럼을 string 타입으로 변환 (NaN 값은 빈 문자열로)
        data['text'] = data['text'].fillna('').astype(str)
    except Exception:
        app.logger.exception('[OCR] vendor crop re-ocr failed')
        return []

    candidates = []
    seen_lines  = set()

    for key in vendor_keys:
        key_rows = data[data['text'].str.contains(key, na=False)]
        for _, krow in key_rows.iterrows():
            lid = (int(krow['block_num']), int(krow['line_num']))
            if lid in seen_lines:
                continue
            seen_lines.add(lid)

            key_right  = int(krow['left']) + int(krow['width'])
            line_top   = int(krow['top'])
            line_h     = int(krow['height'])

            same_line = data[
                (data['block_num'] == krow['block_num']) &
                (data['line_num']  == krow['line_num'])  &
                (data['left'] > key_right)
            ]
            if same_line.empty:
                continue

            x1 = int(same_line['left'].min())
            x2 = min(w, int((same_line['left'] + same_line['width']).max()) + 20)
            y1 = max(0, line_top - 10)
            y2 = min(h, line_top + line_h + 20)

            if x2 - x1 < 30 or y2 - y1 < 10:
                continue

            # raw_gray에서 크롭 → 3x → PSM7 재인식
            crop = raw_gray[y1:y2, x1:x2]
            up3  = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                              interpolation=cv2.INTER_CUBIC)
            t = pytesseract.image_to_string(
                up3, lang=TESS_LANG, config=_cfg(7)
            ).strip()
            # 특수문자 정리
            t = re.sub(r'[|\\[\]{}<>()@#$%^&*]', '', t)
            t = re.sub(r'\s+', ' ', t).strip()

            if re.search(r'[\uAC00-\uD7A3]{2,}', t):
                candidates.append(t[:25])

    return candidates


# ──────────────────────────────────────────
#  멀티패스 OCR
# ──────────────────────────────────────────
def run_ocr(raw_gray: np.ndarray, sharp: np.ndarray) -> str:
    t4  = pytesseract.image_to_string(sharp, lang=TESS_LANG, config=_cfg(4))
    t6  = pytesseract.image_to_string(sharp, lang=TESS_LANG, config=_cfg(6))
    t11 = pytesseract.image_to_string(sharp, lang=TESS_LANG, config=_cfg(11))
    return t4 + '\n' + t6 + '\n' + t11


# ──────────────────────────────────────────
#  파싱 유틸
# ──────────────────────────────────────────
def extract_date(lines):
    for line in lines:
        # YYYY-MM-DD / YYYY.MM.DD / YYYY/MM/DD
        m = re.search(r'(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})', line)
        if m:
            return m.group(2).lstrip('0') or '0', m.group(3).lstrip('0') or '0'
        # YY-MM-DD
        m = re.search(r'\b(\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})\b', line)
        if m:
            return m.group(2).lstrip('0') or '0', m.group(3).lstrip('0') or '0'
        # YYYY년 MM월 DD일 (정상 인식)
        m = re.search(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', line)
        if m:
            return m.group(2).lstrip('0') or '0', m.group(3).lstrip('0') or '0'
        # MM월 DD일
        m = re.search(r'(\d{1,2})월\s*(\d{1,2})일', line)
        if m:
            return m.group(1), m.group(2)
        # OCR 오인식 보완: 'YYYY M D' 공백구분 (토스앱 '20264 53 72' 패턴 등)
        m = re.search(r'\b(20\d{2})\s+(\d{1,2})\s+(\d{1,2})\b', line)
        if m:
            mo, d = int(m.group(2)), int(m.group(3))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return str(mo), str(d)
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
    """합계/총액 키워드 우선, fallback은 전체 텍스트 최댓값
    음수 금액(-9,900원)도 절댓값으로 처리"""
    keywords = ['합계', '총액', '결제금액', '받을금액', '청구금액', '결제액',
                'total', 'TOTAL', '지불금액', '합    계', '합     계',
                '합  계', '합   계']

    def parse_nums_from_line(line):
        # 하이픈 연결 숫자(계좌번호, 전화번호 등) 제거 후 파싱
        cleaned = re.sub(r'\d[\d\-]{5,}\d', '', line)  # XXX-XXXX-XXXX 패턴 제거
        nums = re.findall(r'-?[\d,]+', cleaned)
        result = []
        for n in nums:
            try:
                val = abs(int(n.replace(',', '')))
                if val >= 100:
                    result.append(val)
            except ValueError:
                pass
        return result

    # 1순위: 합계 키워드 라인
    for line in lines:
        if any(k in line for k in keywords):
            candidates = parse_nums_from_line(line)
            if candidates:
                return str(max(candidates))

    # 2순위: '원' 접미사가 붙은 금액 라인 (토스앱: '-9,900원')
    for line in lines:
        if re.search(r'-?[\d,]+원', line):
            m = re.search(r'-?([\d,]+)원', line)
            if m:
                val = int(m.group(1).replace(',', ''))
                if 100 <= val <= 9_999_999:
                    return str(val)

    # 3순위: 전체 텍스트 fallback (하이픈 연결 숫자 제외)
    cleaned_text = re.sub(r'\d[\d\-]{5,}\d', '', full_text)
    all_nums = re.findall(r'-?[\d,]+', cleaned_text)
    candidates = []
    for n in all_nums:
        try:
            val = abs(int(n.replace(',', '')))
            if 100 <= val <= 9_999_999:
                candidates.append(val)
        except ValueError:
            pass
    return str(max(candidates)) if candidates else '0'


def _kor_ratio(s: str) -> float:
    """문자열에서 한글 비율 반환 (0~1)"""
    if not s:
        return 0.0
    kor = sum(1 for c in s if '\uAC00' <= c <= '\uD7A3')
    return kor / len(s)


def _best_candidate(candidates: list) -> str:
    """여러 후보 중 가장 올바르게 인식된 것을 선택
    기준: ① 한글 연속 최대 길이 ② 한글 총 글자수 ③ 전체 길이"""
    if not candidates:
        return ''

    def score(s):
        kor_runs = re.findall(r'[\uAC00-\uD7A3]+', s)
        max_run   = max((len(r) for r in kor_runs), default=0)
        total_kor = sum(len(r) for r in kor_runs)
        return (max_run, total_kor, len(s))

    # 한글이 하나라도 있는 후보 우선
    valid = [c for c in candidates if re.search(r'[\uAC00-\uD7A3]', c)]
    pool  = valid if valid else candidates
    return max(pool, key=score)


def _parse_label_value(line: str):
    """'키워드   값' (공백 2개 이상) 또는 '키워드: 값' 형태 파싱
    → (key, value) 반환. 값이 없으면 ('', '')"""
    # 공백 2개 이상으로 분리 (탭 포함) — 카드영수증/토스앱 레이아웃
    m = re.split(r'\s{2,}|\t', line.strip(), maxsplit=1)
    if len(m) == 2 and m[1].strip():
        return m[0].strip(), m[1].strip()
    # 콜론 분리
    m2 = re.split(r'[:：]\s*', line.strip(), maxsplit=1)
    if len(m2) == 2 and m2[1].strip():
        return m2[0].strip(), m2[1].strip()
    return line.strip(), ''


def extract_vendor(lines):
    """상호명 추출
    멀티패스(PSM 4/6/11) 결과에서 vendor_keys 라인의 값을 모두 수집 →
    한글 비율이 가장 높은 후보를 선택 (깨진 OCR 자동 보정)"""
    vendor_keys = ['상호', '가맹점', '점명', '상점명', '점 명', '입금처', '적요', '업체', '매장']
    candidates = []

    for i, line in enumerate(lines):
        key, val = _parse_label_value(line)

        if any(k in key for k in vendor_keys):
            if val and re.search(r'[\uAC00-\uD7A3a-zA-Z]{2,}', val):
                # 숫자만이거나 전화번호면 제외
                if not re.match(r'^[\d\-\s]+$', val):
                    candidates.append(val[:25])

            # 패턴 2: 키워드 단독 줄 → 바로 다음 줄이 가맹점명
            elif not val:
                for j in range(i + 1, min(i + 4, len(lines))):
                    nxt = lines[j].strip()
                    if not nxt:
                        continue
                    if re.match(r'^[\d\-\s]+$', nxt):
                        break
                    if any(k in nxt for k in ['사업자', '대표', 'TEL', '전화', '주소']):
                        break
                    if re.search(r'[\uAC00-\uD7A3a-zA-Z]{2,}', nxt):
                        candidates.append(nxt[:25])
                    break

    if candidates:
        return _best_candidate(candidates)

    # ── fallback 1: 지점명 패턴 (○○점, ○○마트 등) ──
    store_pat = re.compile(
        r'[\uAC00-\uD7A3a-zA-Z0-9]{2,}'
        r'(?:점|마트|편의점|슈퍼|식당|카페|베이커리|약국|병원|주유소|센터|치킨|피자|버거)'
    )
    for line in lines:
        m = store_pat.search(line)
        if m and len(line) <= 30:
            return m.group(0)

    # ── fallback 2: 한글 2자 이상 + 짧고 깔끔한 라인 ──
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
               '과세', '부가세', '공급가']
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
def parse_receipt(text: str, crop_candidates: list = None) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    month, day  = extract_date(lines)
    meal        = extract_time_meal(lines)
    amount      = extract_amount(lines, text)

    # vendor: 크롭 재OCR 후보가 있으면 우선 사용, 없으면 텍스트 파싱
    if crop_candidates:
        vendor = _best_candidate(crop_candidates)
    else:
        vendor = extract_vendor(lines)

    # 크롭 결과가 깨진 경우 텍스트 파싱 결과와 비교해 더 나은 것 선택
    text_vendor = extract_vendor(lines)
    if text_vendor:
        vendor = _best_candidate([vendor, text_vendor]) if vendor else text_vendor

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

    image_file = request.files['image']
    filename = image_file.filename or 'unknown'
    content_type = image_file.content_type or 'unknown'
    img_bytes = image_file.read()
    app.logger.info(
        '[OCR] request received filename=%s content_type=%s size=%d',
        filename,
        content_type,
        len(img_bytes),
    )

    try:
        raw_gray, sharp = preprocess(img_bytes)
        text            = run_ocr(raw_gray, sharp)
        crop_candidates = _reocr_vendor_crop(raw_gray, sharp)
        result          = parse_receipt(text, crop_candidates)
        app.logger.info('[OCR] success filename=%s parsed_amount=%s', filename, result.get('amount'))
        return jsonify(result)
    except Exception as e:
        app.logger.exception('[OCR] failed filename=%s error=%s', filename, str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Render 환경변수에 PORT가 있으면 쓰고, 없으면 10000번을 기본값으로 사용
    port = int(os.environ.get('PORT', 10000))
    
    # host를 '0.0.0.0'으로 해야 외부에서 접속이 가능합니다!
    app.run(host='0.0.0.0', port=port, debug=True)