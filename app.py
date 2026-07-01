import os
import re
import json
import logging
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

logger = logging.getLogger(__name__)


def _img_debug(stage: str, **fields) -> None:
    parts = ' '.join(f'{key}={value}' for key, value in fields.items())
    logger.info('[IMG_DEBUG] %s %s', stage, parts)

# ── 오타 보정 사전 경로 ──
CORRECTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corrections.json')
_corrections_lock = threading.Lock()
_corrections_cache = None


# ── Tesseract 경로 설정 ──
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_TESSDATA_DIR = os.path.join(_ROOT_DIR, 'tessdata')

if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    _SYSTEM_TESSDATA_DIR = r'C:\Program Files\Tesseract-OCR\tessdata'
else:
    _SYSTEM_TESSDATA_DIR = '/usr/share/tesseract-ocr/5/tessdata'

TESSDATA_DIR = _LOCAL_TESSDATA_DIR if os.path.isdir(_LOCAL_TESSDATA_DIR) else _SYSTEM_TESSDATA_DIR


def _traineddata_exists(name: str) -> bool:
    return os.path.exists(os.path.join(TESSDATA_DIR, f'{name}.traineddata'))


def _pick_tess_lang(candidates: tuple[str, ...]) -> str:
    for lang in candidates:
        parts = lang.split('+')
        if all(_traineddata_exists(part) for part in parts):
            return lang
    for lang in ('kor_best', 'kor', 'eng_best', 'eng'):
        if _traineddata_exists(lang):
            return lang
    return candidates[-1]


os.environ['TESSDATA_PREFIX'] = os.path.dirname(TESSDATA_DIR)

# 설치된 tessdata에 맞춰 자동 선택
TESS_LANG = _pick_tess_lang(('kor_best+eng_best', 'kor+eng'))

# 가맹점명 크롭 재OCR 전용: 한글 단독 모델
# eng_best를 함께 쓰면 영문 글자에 끌려 한글 뒷부분(센텀점 등)을 영문으로
# 오인식하는 경향이 강해서, 가맹점명 영역은 한글 모델만 사용한다.
KOR_ONLY_LANG = _pick_tess_lang(('kor_best', 'kor', TESS_LANG))


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


def _diff_with_context(ocr: str, correct: str):
    """두 문자열에서 달라진 부분을 찾고,
    그 부분이 속한 단어(공백 기준) 전체를 (typo_word, correct_word)로 반환.

    예) ('어오케이커피 센팀점', '어오케이커피 센텀점') → ('센팀점', '센텀점')
        ('신용카듲',             '신용카드')           → ('신용카듲', '신용카드')
    """
    la, lb = len(ocr), len(correct)
    # 공통 접두 길이
    p = 0
    while p < la and p < lb and ocr[p] == correct[p]:
        p += 1
    # 공통 접미 길이
    s = 0
    while s < (la - p) and s < (lb - p) and ocr[la-1-s] == correct[lb-1-s]:
        s += 1

    ocr_diff  = ocr[p: la - s] if s else ocr[p:]
    corr_diff = correct[p: lb - s] if s else correct[p:]

    if not ocr_diff or not corr_diff or ocr_diff == corr_diff:
        return None, None

    def _word_at(full: str, start: int, end: int) -> str:
        """full 문자열에서 start~end 구간을 포함하는 단어 반환"""
        tokens = re.split(r'(\s+)', full)
        pos = 0
        for tok in tokens:
            tok_end = pos + len(tok)
            if not tok.strip():
                pos = tok_end
                continue
            if pos <= start < tok_end or pos < end <= tok_end:
                return tok
            pos = tok_end
        return full[start:end]

    typo_word    = _word_at(ocr,     p, la - s)
    correct_word = _word_at(correct, p, lb - s)
    return typo_word, correct_word


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

    typo, correct = _diff_with_context(ocr_value, corrected_value)
    if not typo or not correct or typo == correct:
        return None
    if len(typo) > 12 or len(correct) > 12:
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


def extract_payment_time(lines):
    """결제 시각 추출. 반환 형식은 HH:MM 또는 ''"""
    for line in lines:
        m = re.search(r'오전\s*(\d{1,2})[:\.]?(\d{2})', line)
        if m:
            h = int(m.group(1))
            if 1 <= h <= 12:
                return f'{h:02d}:{m.group(2)}'
        m = re.search(r'오후\s*(\d{1,2})[:\.]?(\d{2})', line)
        if m:
            h = int(m.group(1)) + 12
            if h == 24:
                h = 12
            return f'{h:02d}:{m.group(2)}'
        m = re.search(r'\b(AM|am)\s*(\d{1,2})[:\.]?(\d{2})', line)
        if m:
            h = int(m.group(2))
            if 1 <= h <= 12:
                return f'{h:02d}:{m.group(3)}'
        m = re.search(r'\b(PM|pm)\s*(\d{1,2})[:\.]?(\d{2})', line)
        if m:
            h = int(m.group(2)) + 12
            if h == 24:
                h = 12
            return f'{h:02d}:{m.group(3)}'
        m = re.search(r'\b(\d{1,2}):(\d{2})(?::\d{2})?\b', line)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return f'{h:02d}:{m.group(2)}'
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
    payment_time = extract_payment_time(lines)
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
        'payment_time': payment_time,
        'raw_text':    text,
    }


import uuid
import io
import tempfile
from pathlib import Path
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.page import PageMargins   # ← 추가
from openpyxl.utils import get_column_letter
from PIL import Image as PILImage
from flask import send_file

# # 업로드된 이미지를 메모리에 임시 보관 {image_id: bytes}
# _image_store: dict = {}
# _image_store_lock = threading.Lock()

# 업로드 이미지를 임시 디렉토리에 파일로 저장
# (메모리 저장은 서버 재시작 시 소실되므로 파일로 유지)
_IMAGE_DIR = Path(tempfile.gettempdir()) / 'receipt_ocr_images'
_IMAGE_DIR.mkdir(exist_ok=True)
_image_store_lock = threading.Lock()


import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True   # 잘린 JPEG/PNG도 부분 복원

# HEIC 지원 (pillow-heif 설치 시)
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

def _save_image(image_id: str, img_bytes: bytes) -> None:
    """이미지를 RGB PNG로 정규화 후 저장.
    3단계 폴백 전략:
      1) PIL (LOAD_TRUNCATED_IMAGES=True) — 잘린 JPEG/PNG/HEIC 등 대부분 커버
      2) cv2.imdecode                      — PIL이 포맷 식별 실패 시
      3) 원본 바이트 그대로 저장            — 모두 실패 시 (export 시 오류 메시지 표시)
    """
    dest_png  = _IMAGE_DIR / f'{image_id}.png'
    dest_raw  = _IMAGE_DIR / f'{image_id}.raw'
    dest_err  = _IMAGE_DIR / f'{image_id}.err'

    if not img_bytes or len(img_bytes) < 4:
        dest_err.write_text('빈 이미지 데이터', encoding='utf-8')
        return

    # ── 전략 1: PIL ──
    try:
        buf = io.BytesIO(img_bytes)
        buf.seek(0)
        pil = PILImage.open(buf)
        pil.load()
        if pil.mode in ('RGBA', 'P', 'LA'):
            bg = PILImage.new('RGB', pil.size, (255, 255, 255))
            mask = pil.split()[3] if pil.mode in ('RGBA', 'LA') else None
            bg.paste(pil, mask=mask)
            pil = bg
        elif pil.mode != 'RGB':
            pil = pil.convert('RGB')
        pil.save(dest_png, format='PNG')
        return
    except Exception as e1:
        pass

    # ── 전략 2: cv2 ──
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_cv is not None:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(img_rgb)
            pil.save(dest_png, format='PNG')
            return
    except Exception as e2:
        pass

    # ── 전략 3: 원본 저장 (export 시 재시도) ──
    dest_raw.write_bytes(img_bytes)
    dest_err.write_text(
        f'PIL 실패: 포맷 미인식\ncv2 실패: 디코딩 불가\n원본 바이트 저장됨 ({len(img_bytes):,}B)',
        encoding='utf-8'
    )

def _load_image(image_id: str) -> tuple:
    """(img_bytes_or_None, err_msg_or_None) 반환"""
    png = _IMAGE_DIR / f'{image_id}.png'
    raw = _IMAGE_DIR / f'{image_id}.raw'
    err = _IMAGE_DIR / f'{image_id}.err'

    err_msg = err.read_text(encoding='utf-8') if err.exists() else None

    if png.exists():
        return png.read_bytes(), None       # 정상 PNG
    if raw.exists():
        return raw.read_bytes(), err_msg    # 원본 + 오류 메시지
    return None, err_msg or '이미지 파일 없음'

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
    _img_debug('ocr-received', filename=getattr(request.files['image'], 'filename', ''), size=len(img_bytes))

    # # 이미지 메모리 저장
    # image_id = str(uuid.uuid4())
    # with _image_store_lock:
    #     _image_store[image_id] = img_bytes

    # 이미지 파일로 저장
    image_id = str(uuid.uuid4())
    _save_image(image_id, img_bytes)
    
    try:
        _img_debug('ocr-preprocess-start', image_id=image_id)
        raw_gray, sharp = preprocess(img_bytes)
        _img_debug('ocr-preprocess-done', image_id=image_id, raw_shape=getattr(raw_gray, 'shape', None), sharp_shape=getattr(sharp, 'shape', None))
        text            = run_ocr(raw_gray, sharp)
        _img_debug('ocr-run-done', image_id=image_id, text_len=len(text or ''))
        crop_candidates = _reocr_vendor_crop(raw_gray, sharp)
        _img_debug('ocr-crop-done', image_id=image_id, crop_count=len(crop_candidates or []))
        result          = parse_receipt(text, crop_candidates)
        result['image_id'] = image_id
        return jsonify(result)
    except Exception as e:
        logger.exception('[IMG_DEBUG] ocr-failed image_id=%s', image_id)
        return jsonify({'error': str(e), 'image_id': image_id}), 500


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
    """현재 오타 보정 사전 조회"""
    return jsonify(_load_corrections())


@app.route('/corrections/<int:idx>', methods=['DELETE'])
def delete_correction(idx):
    """보정 사전 항목 삭제 (인덱스 기준)"""
    with _corrections_lock:
        data = _load_corrections()
        entries = data.get('corrections', [])
        if idx < 0 or idx >= len(entries):
            return jsonify({'ok': False, 'error': '인덱스 범위 초과'}), 404
        removed = entries.pop(idx)
        _save_corrections(data)
        return jsonify({'ok': True, 'removed': removed})


@app.route('/export', methods=['POST'])
def export():
    """지출내역 시트 + 영수증 이미지 시트가 포함된 xlsx 반환.

    body (JSON):
    {
      "rows": [{"month","day","vendor","description","amount","image_id"?}, ...],
      "order": ["image_id", ...]   // 시간순 image_id 목록
    }
    이미지 시트: 5열씩, order 순서대로 배치. image_id 없는 row는 제외.
    """
    data   = request.get_json(silent=True) or {}
    rows   = data.get('rows', [])
    order  = data.get('order', [])   # 업로드 순서 (시간순)

    wb = Workbook()

    # ── 시트 1: 지출내역 ──
    ws1 = wb.active
    ws1.title = '지출내역'

    header_fill   = PatternFill('solid', start_color='1A1A2E')
    header_font   = Font(bold=True, color='C8F04A', name='Arial', size=10)
    body_font     = Font(name='Arial', size=10)
    center        = Alignment(horizontal='center', vertical='center')
    right         = Alignment(horizontal='right',  vertical='center')
    thin          = Side(style='thin', color='2A2A40')
    border        = Border(left=thin, right=thin, top=thin, bottom=thin)

    headers = ['월', '일', '거래처명', '지출내용', '금액 (원)']
    col_widths = [6, 6, 28, 20, 14]
    for ci, (h, w) in enumerate(zip(headers, col_widths), 1):
        cell = ws1.cell(row=1, column=ci, value=h)
        cell.font      = header_font
        cell.fill      = header_fill
        cell.alignment = center
        cell.border    = border
        ws1.column_dimensions[get_column_letter(ci)].width = w

    for ri, row in enumerate(rows, 2):
        vals = [row.get('month',''), row.get('day',''),
                row.get('vendor',''), row.get('description',''),
                int(row.get('amount', 0) or 0)]
        for ci, v in enumerate(vals, 1):
            cell = ws1.cell(row=ri, column=ci, value=v)
            cell.font      = body_font
            cell.border    = border
            cell.alignment = right if ci == 5 else center if ci <= 2 else Alignment(vertical='center')
        ws1.row_dimensions[ri].height = 18

    # 합계 행
    if rows:
        total_row = len(rows) + 2
        ws1.cell(row=total_row, column=4, value='합계').font = Font(bold=True, name='Arial', size=10)
        ws1.cell(row=total_row, column=5,
                 value=f'=SUM(E2:E{total_row-1})').font = Font(bold=True, name='Arial', size=10, color='4ADE80')
        ws1.cell(row=total_row, column=5).alignment = right

    ws1.freeze_panes = 'A2'
    ws1.row_dimensions[1].height = 22

    # ── 시트 2: 영수증 이미지 ──
    ws2 = wb.create_sheet('영수증 이미지')

    # ── 인쇄 페이지 설정 ──
    _MARGIN_INCH = 0.5 / 2.54          # 0.5cm → 인치
    ws2.page_setup.orientation = 'landscape'
    ws2.page_setup.paperSize   = 9
    ws2.page_setup.fitToPage   = True   # ← 추가
    ws2.page_setup.fitToWidth  = 1      # ← 추가: 가로 1페이지에 맞춤
    ws2.page_setup.fitToHeight = 1      # ← 추가: 세로 1페이지에 맞춤
    ws2.sheet_properties.pageSetUpPr.fitToPage = True   # ← 추가: Excel 인식용
    ws2.page_margins = PageMargins(
        top=_MARGIN_INCH, bottom=_MARGIN_INCH,
        left=_MARGIN_INCH, right=_MARGIN_INCH,
        header=0, footer=0,
    )

    COLS      = 5              # 열 수
    THUMB_W   = 600            # 썸네일 픽셀 너비
    THUMB_H   = 800            # 썸네일 픽셀 높이
    GAP_PX    = 19             # 열 간 공백 (px) ≈ 0.5cm @ 96dpi

    # Excel 단위 변환
    # 열 너비: px / 7  (기본 폰트 Calibri 11pt 기준)
    # 행 높이: px * 0.75  (1pt = 1.333px)
    CELL_W      = round(THUMB_W / 7, 1)   # ≈ 85.7 Excel units
    IMG_ROW_H   = round(THUMB_H * 0.75)   # ≈ 600 pt
    GAP_ROW_H   = round(GAP_PX  * 0.75)   # ≈ 14 pt  (0.5cm 공백)
    LBL_ROW_H   = 14                        # pt
    # 행 그룹 구성: [이미지행, 라벨행, 공백행] → 3행 단위
    ROWS_PER_GROUP = 3

    # 열 너비 설정
    for ci in range(1, COLS + 1):
        ws2.column_dimensions[get_column_letter(ci)].width = CELL_W

    # ── 날짜/시간 기준 정렬 ──
    # month → day → 식사 순서(조식=0, 중식=1, 석식=2, 기타=3) 순으로 정렬
    MEAL_ORDER = {'조식': 0, '중식': 1, '석식': 2}

    def _sort_key(row):
        try:    mo = int(row.get('month') or 0)
        except: mo = 0
        try:    d  = int(row.get('day')   or 0)
        except: d  = 0
        meal = MEAL_ORDER.get(row.get('description', ''), 3)
        return (mo, d, meal)

    id_to_row   = {r.get('image_id'): r for r in rows if r.get('image_id')}
    ordered_ids = [r['image_id'] for r in sorted(id_to_row.values(), key=_sort_key)]

    # with _image_store_lock:
    #     store_snap = dict(_image_store)

    for seq, image_id in enumerate(ordered_ids):
        row_data           = id_to_row[image_id]
        img_bytes, load_err = _load_image(image_id)

        col_idx = (seq % COLS) + 1
        group   = seq // COLS
        img_row = group * ROWS_PER_GROUP + 1
        lbl_row = img_row + 1
        gap_row = img_row + 2

        ws2.row_dimensions[img_row].height = IMG_ROW_H
        ws2.row_dimensions[lbl_row].height = LBL_ROW_H
        ws2.row_dimensions[gap_row].height = GAP_ROW_H

        # 라벨: 월/일 · 거래처명 · 금액
        mo         = row_data.get('month', '')
        day        = row_data.get('day', '')
        date_str   = f'{mo}/{day}' if mo and day else ''
        amount_str = f'{int(row_data.get("amount", 0) or 0):,}원'
        label      = f'{date_str}  {row_data.get("vendor","")}  {amount_str}'
        lbl_cell   = ws2.cell(row=lbl_row, column=col_idx, value=label)
        lbl_cell.font      = Font(name='Arial', size=8, color='555555')
        lbl_cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

        # 이미지 데이터 없음
        if not img_bytes:
            err_text = f'이미지 없음\n{load_err or ""}'
            cell = ws2.cell(row=img_row, column=col_idx, value=err_text)
            cell.font      = Font(name='Arial', size=8, color='FF4444')
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            continue

        try:
            # pil_img = PILImage.open(io.BytesIO(img_bytes))
            buf = io.BytesIO(img_bytes)
            buf.seek(0)
            _img_debug('export-open-start', image_id=image_id, bytes_size=len(img_bytes), head=img_bytes[:16].hex())
            pil_img = PILImage.open(buf)
            pil_img.load()   # 실제 디코딩 강제 → 손상 이미지 즉시 감지            
            _img_debug('export-open-ok', image_id=image_id, format=pil_img.format, mode=pil_img.mode, size=pil_img.size)
            pil_img.thumbnail((THUMB_W, THUMB_H), PILImage.LANCZOS)

            # RGBA / 팔레트 모드 → RGB 변환
            if pil_img.mode in ('RGBA', 'P', 'LA'):
                bg = PILImage.new('RGB', pil_img.size, (255, 255, 255))
                mask = pil_img.split()[3] if pil_img.mode in ('RGBA', 'LA') else None
                bg.paste(pil_img, mask=mask)
                pil_img = bg
            elif pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            img_buf = io.BytesIO()
            pil_img.save(img_buf, format='PNG')
            img_buf.seek(0)

            # 실제 썸네일 크기 기준으로 Excel 이미지 크기 계산
            actual_w, actual_h = pil_img.size
            xl_img        = XLImage(img_buf)
            xl_img.width  = actual_w
            xl_img.height = actual_h

            ws2.add_image(xl_img, f'{get_column_letter(col_idx)}{img_row}')

            # 원본이 손상돼 cv2/raw로 저장된 경우 셀에 경고 텍스트도 표시
            if load_err:
                cell = ws2.cell(row=img_row, column=col_idx,
                                value=f'⚠ 부분 복원됨\n{load_err[:80]}')
                cell.font      = Font(name='Arial', size=7, color='F59E0B')
                cell.alignment = Alignment(horizontal='center', vertical='top', wrap_text=True)

        except Exception as e:
            # 저장·복원 모두 실패한 최종 오류: 오류 내용 전체 표시
            lines = [f'이미지 변환 실패', f'{type(e).__name__}: {e}']
            if load_err:
                lines.append(f'저장 시 오류: {load_err}')
            cell = ws2.cell(row=img_row, column=col_idx, value='\n'.join(lines))
            cell.font      = Font(name='Arial', size=8, color='FF4444')
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            
    # xlsx → BytesIO
    out_buf = io.BytesIO()
    wb.save(out_buf)
    out_buf.seek(0)

    return send_file(
        out_buf,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='영수증_지출내역.xlsx',
    )


if __name__ == '__main__':
    # Render 환경변수에 PORT가 있으면 쓰고, 없으면 7860번을 기본값으로 사용
    port = int(os.environ.get('PORT', 7860))
    
    # host를 '0.0.0.0'으로 해야 외부에서 접속이 가능합니다!
    app.run(host='0.0.0.0', port=port, debug=True)