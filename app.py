import os
import re
import json
import threading
from typing import Optional
import cv2
import numpy as np
import pytesseract
import platform
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ── 오타 보정 사전 경로 ──
CORRECTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corrections.json')
_corrections_lock = threading.Lock()
_corrections_cache = None

# ── Tesseract 경로 설정 ──
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSDATA_DIR = r'C:\Program Files\Tesseract-OCR\tessdata'
else:
    TESSDATA_DIR = '/usr/share/tesseract-ocr/5/tessdata'

os.environ.setdefault('TESSDATA_PREFIX', TESSDATA_DIR)

# kor_best 모델 존재 여부에 따라 자동 선택
best_kor = os.path.join(TESSDATA_DIR, 'kor_best.traineddata')
best_eng = os.path.join(TESSDATA_DIR, 'eng_best.traineddata')
TESS_LANG = 'kor_best+eng_best' if (os.path.exists(best_kor) and os.path.exists(best_eng)) else 'kor+eng'

# 가맹점명 크롭 재OCR 전용: 한글 단독 모델
# eng_best를 함께 쓰면 영문 글자에 끌려 한글 뒷부분(센텀점 등)을 영문으로
# 오인식하는 경향이 강해서, 가맹점명 영역은 한글 모델만 사용한다.
KOR_ONLY_LANG = 'kor_best' if os.path.exists(best_kor) else 'kor'

# PSM별 config (멀티패스에서 사용)
def _cfg(psm: int) -> str:
    return f'--psm {psm} --oem 1 --tessdata-dir {TESSDATA_DIR}'


# ──────────────────────────────────────────
#  오타 보정 사전 (corrections.json)
#
#  - apply_corrections(text)     : OCR 결과 텍스트에서 typo_variants를
#                                   correct_word로 치환
#  - learn_correction(ocr, user) : 사용자가 수동으로 값을 수정했을 때
#                                   OCR 결과와 수정값을 비교해 새로운
#                                   typo_variant를 사전에 자동 추가
# ──────────────────────────────────────────
_DEFAULT_CORRECTIONS = {
    "version": "1.0.0",
    "description": "영수증 OCR 한글 폰트 오인식 및 노이즈 보정용 사전",
    "corrections": [],
}


def _load_corrections() -> dict:
    """corrections.json 로드 (메모리 캐시)"""
    global _corrections_cache
    if _corrections_cache is not None:
        return _corrections_cache
    try:
        with open(CORRECTIONS_PATH, 'r', encoding='utf-8') as f:
            _corrections_cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _corrections_cache = json.loads(json.dumps(_DEFAULT_CORRECTIONS))
    _corrections_cache.setdefault('corrections', [])
    return _corrections_cache


def _save_corrections(data: dict) -> None:
    """corrections.json 저장 + 캐시 갱신"""
    global _corrections_cache
    with open(CORRECTIONS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    _corrections_cache = data


def _build_typo_pairs(data: dict) -> list:
    """(typo, correct_word) 쌍 리스트, typo 길이 내림차순
    (긴 패턴을 먼저 치환해야 짧은 패턴에 의한 부분 치환 오류를 피할 수 있음)"""
    pairs = []
    for entry in data.get('corrections', []):
        correct = entry.get('correct_word', '')
        if not correct:
            continue
        for typo in entry.get('typo_variants', []):
            if typo and typo != correct:
                pairs.append((typo, correct))
    pairs.sort(key=lambda p: -len(p[0]))
    return pairs


def apply_corrections(text: str) -> str:
    """OCR 결과 문자열에서 corrections.json의 typo_variants를
    correct_word로 치환해 반환"""
    if not text:
        return text
    data = _load_corrections()
    for typo, correct in _build_typo_pairs(data):
        if typo in text:
            text = text.replace(typo, correct)
    return text


def _diff_strings(a: str, b: str):
    """a, b의 공통 접두/접미를 제거하고 차이나는 가운데 부분만 반환
    예) ('센턴점', '센텀점') → ('턴', '텀')
        ('이마트24 MEIAIO|AALIEH', '이마트24 센텀사이언스파트점')
            → ('MEIAIO|AALIEH', '센텀사이언스파트점')"""
    la, lb = len(a), len(b)
    p = 0
    while p < la and p < lb and a[p] == b[p]:
        p += 1
    s = 0
    while s < (la - p) and s < (lb - p) and a[la - 1 - s] == b[lb - 1 - s]:
        s += 1
    return a[p:la - s], b[p:lb - s]


def _guess_category(correct_word: str) -> str:
    """새 보정 항목의 category 추정"""
    if correct_word.endswith('점'):
        return 'branch_suffix'
    if correct_word.endswith(('카드', '결제', '불')):
        return 'payment_method'
    if any(k in correct_word for k in ('일시', '날짜')):
        return 'date_time'
    return 'user_learned'


def _char_similarity(a: str, b: str) -> float:
    """두 문자열의 글자 유사도 (0~1). 길이 대비 공통 접두/접미 비율."""
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    # 공통 접두 길이
    p = 0
    while p < la and p < lb and a[p] == b[p]:
        p += 1
    # 공통 접미 길이
    s = 0
    while s < (la - p) and s < (lb - p) and a[la - 1 - s] == b[lb - 1 - s]:
        s += 1
    common = p + s
    return common / max(la, lb)


def learn_correction(ocr_value: str, corrected_value: str) -> Optional[dict]:
    """OCR 인식값과 사용자가 수동으로 고친 값을 비교해
    corrections.json에 새 typo_variant / correct_word를 학습한다.

    조건:
      - 두 값이 다를 것
      - 유사도(공통 접두/접미 비율) ≥ 0.4 → 완전히 다른 가게명 전체 교체는 학습하지 않음
      - 차이나는 부분(typo, correct)이 각각 1~8자일 것
    """
    ocr_value = (ocr_value or '').strip()
    corrected_value = (corrected_value or '').strip()
    if not ocr_value or not corrected_value or ocr_value == corrected_value:
        return None

    # 유사도 체크: 너무 다른 문자열은 학습하지 않음
    if _char_similarity(ocr_value, corrected_value) < 0.4:
        return None

    typo, correct = _diff_strings(ocr_value, corrected_value)
    if not typo or not correct or typo == correct:
        return None
    if len(typo) > 8 or len(correct) > 8:
        return None

    with _corrections_lock:
        data = _load_corrections()
        entries = data.setdefault('corrections', [])

        for entry in entries:
            if entry.get('correct_word') == correct:
                variants = entry.setdefault('typo_variants', [])
                if typo not in variants:
                    variants.append(typo)
                    _save_corrections(data)
                    return {'action': 'updated', 'category': entry.get('category'),
                            'correct_word': correct, 'typo_variants': variants}
                return {'action': 'noop', 'category': entry.get('category'),
                        'correct_word': correct, 'typo_variants': variants}

        new_entry = {
            'category': _guess_category(correct),
            'correct_word': correct,
            'typo_variants': sorted(set([typo, correct])),
        }
        entries.append(new_entry)
        _save_corrections(data)
        return {'action': 'created', **new_entry}


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
#  오른쪽 값 영역만 raw_gray 멀티스케일(3x·4x) + PSM7 재인식
#  → 토스앱/카드영수증 모두 커버
# ──────────────────────────────────────────
def _vendor_score(s: str):
    """vendor 후보 문자열 점수: (한글 연속최대길이, 한글 총길이, -잡음문자수)
    값이 클수록 더 정확한 인식 결과로 판단"""
    runs = re.findall(r'[\uAC00-\uD7A3]+', s)
    max_run   = max((len(r) for r in runs), default=0)
    total_kor = sum(len(r) for r in runs)
    junk      = len(re.findall(r'[^\uAC00-\uD7A3a-zA-Z0-9()\s]', s))
    return (max_run, total_kor, -junk)


def _ocr_region(raw_gray, x1, y1, x2, y2, scales=(3, 4)):
    """지정 영역을 여러 배율로 PSM7 재OCR, 최고 점수 결과 반환"""
    h, w = raw_gray.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 5:
        return ''
    crop = raw_gray[y1:y2, x1:x2]
    best, best_score = '', (-1, -1, -999)
    for scale in scales:
        up = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                         interpolation=cv2.INTER_CUBIC)
        t = pytesseract.image_to_string(up, lang=KOR_ONLY_LANG, config=_cfg(7)).strip()
        t = re.sub(r'\s+', ' ', t).strip()
        if not t:
            continue
        score = _vendor_score(t)
        if score > best_score:
            best_score, best = score, t
    return best


def _reocr_vendor_crop(raw_gray: np.ndarray, sharp: np.ndarray) -> list:
    """vendor 키워드 라인의 값 영역을 재OCR, 후보 리스트 반환

    두 전략을 모두 시도해 더 나은 쪽을 선택:
    A) 전체 라인 영역, left_pad 0/20/40, 3x·4x, PSM7
    B) 마지막 단어 영역만 left_pad 40/50/60 + y_pad 20/25/30 (앞단어 컨텍스트 포함),
       3x·4x, PSM7 → 앞쪽 고신뢰 단어들과 결합
    B는 마지막 단어가 짧은 UI 폰트로 깨지는 경우(예: '센텀점')에 효과적
    """
    vendor_keys = ['적요', '입금처', '가맹점', '상호', '점명', '사용처']
    h, w = raw_gray.shape

    try:
        data = pytesseract.image_to_data(
            sharp, lang=TESS_LANG, config=_cfg(6),
            output_type=pytesseract.Output.DATAFRAME
        )
    except Exception:
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

            key_right = int(krow['left']) + int(krow['width'])
            same_line = data[
                (data['block_num'] == krow['block_num']) &
                (data['line_num']  == krow['line_num'])  &
                (data['left'] > key_right)
            ].reset_index(drop=True)
            if same_line.empty:
                continue

            # ── 전략 A: 전체 라인, left_pad 0/20/40 ──
            x_start = int(same_line['left'].min())
            y1 = int(same_line['top'].min()) - 5
            y2 = int((same_line['top'] + same_line['height']).max()) + 10
            a_best, a_score = '', (-1, -1, -999)
            for left_pad in (0, 20, 40):
                t = _ocr_region(raw_gray, x_start - left_pad, y1, w, y2)
                score = _vendor_score(t)
                if score > a_score:
                    a_score, a_best = score, t

            # ── 전략 B: 마지막 단어만 큰 패딩으로 재OCR + 앞단어 결합 ──
            last = same_line.iloc[-1]
            head = ''.join(same_line.iloc[:-1]['text'].fillna('').tolist())
            wl, wt = int(last['left']), int(last['top'])
            ww, wh = int(last['width']), int(last['height'])
            b_best, b_score = '', (-1, -1, -999)
            for left_pad in (40, 50, 60):
                for y_pad in (20, 25, 30):
                    t = _ocr_region(raw_gray, wl - left_pad, wt - y_pad,
                                     w, wt + wh + y_pad)
                    if not t:
                        continue
                    # 앞쪽 쓰레기 문자 제거 (단어 경계 잘림 보정용 문자)
                    m = re.search(r'[\uAC00-\uD7A3a-zA-Z0-9]', t)
                    tail = t[m.start():] if m else t
                    combined = (head + ' ' + tail).strip() if head else tail
                    score = _vendor_score(combined)
                    if score > b_score:
                        b_score, b_best = score, combined

            best = a_best if a_score >= b_score else b_best
            if re.search(r'[\uAC00-\uD7A3]{2,}', best):
                candidates.append(_clean_vendor(best)[:25])

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
        # YYYY-MM-DD / YYYY.MM.DD / YYYY/MM/DD  (거래일시, 결제일시 등 포함)
        m = re.search(r'(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})', line)
        if m:
            return m.group(2).lstrip('0') or '0', m.group(3).lstrip('0') or '0'
        # YY-MM-DD
        m = re.search(r'\b(\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})\b', line)
        if m:
            return m.group(2).lstrip('0') or '0', m.group(3).lstrip('0') or '0'
        # YYYY년 MM월 DD일 (결제일시, 일시 등)
        m = re.search(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', line)
        if m:
            return m.group(2).lstrip('0') or '0', m.group(3).lstrip('0') or '0'
        # MM월 DD일
        m = re.search(r'(\d{1,2})월\s*(\d{1,2})일', line)
        if m:
            return m.group(1), m.group(2)
        # OCR 오인식 보완: 'YYYY M D' 공백구분
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


def _clean_vendor(s: str) -> str:
    """크롭 재OCR 후 한글 뒤에 오는 영대문자 쓰레기 제거
    '어오케이커피 MIE' → '어오케이커피'
    '이마트24 센팀사이언스파트점' → 그대로 (한글이 뒤에도 있음)"""
    if not s:
        return s
    # 마지막 한글 단어 이후 순수 영대문자/특수문자만 연속되면 제거
    cleaned = re.sub(
        r'([\uAC00-\uD7A3\d])\s+[A-Z]{2,}[\w\[\]|\\{}<>@#]*'
        r'(\s+[A-Z]{2,}[\w\[\]|\\{}<>@#]*)*\s*$',
        r'\1', s
    )
    return cleaned.strip()


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
    vendor_keys = ['상호', '가맹점', '점명', '상점명', '점 명', '입금처', '적요', '업체', '매장', '사용처']
    candidates = []

    for i, line in enumerate(lines):
        key, val = _parse_label_value(line)

        if any(k in key for k in vendor_keys):
            if val and re.search(r'[\uAC00-\uD7A3a-zA-Z]{2,}', val):
                # 숫자만이거나 전화번호면 제외
                if not re.match(r'^[\d\-\s]+$', val):
                    candidates.append(_clean_vendor(val)[:25])

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
    # ── 오타 보정 사전 적용 (OCR 결과 전체) ──
    text = apply_corrections(text)
    if crop_candidates:
        crop_candidates = [apply_corrections(c) for c in crop_candidates]

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

    # 최종 vendor에도 보정 사전 한 번 더 적용 (fallback 경로 보호)
    vendor = apply_corrections(vendor)

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

    img_bytes = request.files['image'].read()
    try:
        raw_gray, sharp = preprocess(img_bytes)
        text            = run_ocr(raw_gray, sharp)
        crop_candidates = _reocr_vendor_crop(raw_gray, sharp)
        result          = parse_receipt(text, crop_candidates)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    """사용자가 수동으로 값을 수정했을 때 호출.
    body: {"ocr_value": "<OCR이 인식한 값>", "corrected_value": "<사용자가 고친 값>"}

    OCR 값과 수정값을 비교해 corrections.json에 새 typo_variant /
    correct_word를 자동으로 학습한다."""
    data = request.get_json(silent=True) or {}
    ocr_value       = data.get('ocr_value', '')
    corrected_value = data.get('corrected_value', '')

    result = learn_correction(ocr_value, corrected_value)
    if result is None:
        return jsonify({'learned': False})
    return jsonify({'learned': True, **result})


@app.route('/corrections', methods=['GET'])
def get_corrections():
    """현재 오타 보정 사전 조회 (디버깅/확인용)"""
    return jsonify(_load_corrections())


if __name__ == '__main__':
    # Render 환경변수에 PORT가 있으면 쓰고, 없으면 7860번을 기본값으로 사용
    port = int(os.environ.get('PORT', 7860))
    
    # host를 '0.0.0.0'으로 해야 외부에서 접속이 가능합니다!
    app.run(host='0.0.0.0', port=port, debug=True)