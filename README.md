# Medicine Name Recognition (تشخیص نام دارو از روی عکس)

سیستم **پیشنهاد نام دارو** از روی عکس جعبه یا برچسب دارو، بر پایه‌ی OCR و تطبیق متنی.

کاربر عکس می‌دهد، سیستم همه‌ی متن انگلیسی روی عکس را می‌خواند، آن را با لیست نام داروهای ثبت‌شده در `labels.csv` مقایسه می‌کند و بهترین نام را **فقط پیشنهاد می‌دهد**. تأیید نهایی همیشه با کاربر است.

> **این سیستم چه چیزی نیست:** تشخیص‌دهنده‌ی تصویری (رنگ یا شکل قرص)، مدل آموزش‌دیده یا fine-tune‌شده، منبع توصیه‌ی پزشکی، یا سیستمی که پیشنهاد را خودکار بپذیرد. امتیاز خروجی «شباهت متنی» است، نه احتمال درست‌بودن تشخیص دارو.

---

## فهرست

1. [خلاصه‌ی قابلیت‌ها](#1-خلاصه‌ی-قابلیت‌ها)
2. [مدل‌ها و کتابخانه‌ها](#2-مدل‌ها-و-کتابخانه‌ها)
3. [نصب](#3-نصب)
4. [راه‌های اجرا](#4-راه‌های-اجرا)
5. [ساختار پروژه](#5-ساختار-پروژه)
6. [دیتاست](#6-دیتاست)
7. [پایپ‌لاین مرحله به مرحله](#7-پایپ‌لاین-مرحله-به-مرحله)
8. [همه‌ی پارامترهای config.py](#8-همه‌ی-پارامترهای-configpy)
9. [ثابت‌ها و قوانین داخلی کد](#9-ثابت‌ها-و-قوانین-داخلی-کد)
10. [خروجی recognize (همه‌ی فیلدها)](#10-خروجی-recognize-همه‌ی-فیلدها)
11. [API برای اپ موبایل](#11-api-برای-اپ-موبایل)
12. [ارزیابی](#12-ارزیابی)
13. [عملکرد و سرعت](#13-عملکرد-و-سرعت)
14. [تست‌ها](#14-تست‌ها)
15. [راهنمای اپ‌سازها و Deploy](#15-راهنمای-اپ‌سازها-و-deploy)
16. [عیب‌یابی](#16-عیب‌یابی)
17. [محدودیت‌ها و نکات مسئولیتی](#17-محدودیت‌ها-و-نکات-مسئولیتی)
18. [مسیر بهبود](#18-مسیر-بهبود)

---

## 1. خلاصه‌ی قابلیت‌ها

| قابلیت | توضیح |
|---|---|
| اعتبارسنجی عکس | فرمت (JPG/JPEG/PNG)، حجم، قابل‌خواندن‌بودن، حداقل ابعاد |
| پیش‌پردازش سبک | تغییر اندازه (همیشه)، بهبود کنتراست و شارپ (اختیاری، پیش‌فرض خاموش) |
| OCR با PaddleOCR | تشخیص و خواندن متن انگلیسی، نگه‌داشتن همه‌ی نواحی با `{text, ocr_confidence, bounding_box}` |
| خروجی خام OCR | قابل‌مشاهده با `--debug` |
| نرمال‌سازی | حروف بزرگ، حذف علائم، حذف کلمات نویز (MG، TABLET، EXP، …) |
| اصلاح کنترل‌شده‌ی خطای OCR | مثلاً `AM0XICILLIN` ← `AMOXICILLIN` (بدون جایگزینی متن اصلی) |
| ساخت Query هوشمند | توکن تکی، جفت مجاور، دنباله‌ی کوتاه، ترکیب نواحی همسایه با مختصات کادر |
| تطبیق دقیق | امتیاز ۱۰۰ |
| تطبیق تقریبی | RapidFuzz (`fuzz.ratio`)، امتیاز ۰ تا ۱۰۰ |
| رتبه‌بندی | یک کاندیدا برای هر دارو، حداکثر ۵ کاندیدا |
| تشخیص ابهام | اگر دو کاندیدای برتر نزدیک هم باشند، `ambiguous=true` با لیست جایگزین‌ها |
| آستانه‌ی قابل‌تنظیم | امتیاز ≥ ۶۰ ← `SUGGESTED`، وگرنه `UNKNOWN` |
| تأیید و ورود دستی | `CONFIRMED` / `REJECTED` / `MANUAL_ACCEPTED`، بدون تغییر دیتاست |
| بک‌اند FastAPI | برای اپ موبایل، بدون رابط وب |
| ابزار دمو | `scripts/demo.py`: انتخاب عکس و دیدن خروجی خوانا |
| ارزیابی | داخل/خارج دیتاست (Negative)، Top-1/3/5، تقسیم tuning/test قفل‌شده |
| بنچمارک | اندازه‌گیری زمان هر مرحله |
| تست خودکار | ۴۴ تست pytest |

---

## 2. مدل‌ها و کتابخانه‌ها

> هیچ مدل یادگیری ماشینی در این پروژه **آموزش داده نشده است**. فقط مدل‌های آماده‌ی OCR استفاده می‌شوند و بقیه‌ی منطق (نرمال‌سازی، تطبیق، تصمیم) الگوریتمی و قابل‌توضیح است.

### مدل‌های OCR (PaddleOCR / PaddleX)

| نقش | مدل | توضیح |
|---|---|---|
| تشخیص محل متن (Detection) | `PP-OCRv5_mobile_det` | مدل سبک CPU. پیدا می‌کند متن کجای عکس است |
| خواندن متن (Recognition) | `en_PP-OCRv3_mobile_rec` | مدل سبک CPU، انگلیسی. کادرهای پیداشده را به متن تبدیل می‌کند |

مدل‌ها بار اول دانلود می‌شوند و در `%USERPROFILE%\.paddlex\official_models` کش می‌شوند.

**جایگزین‌های بررسی‌شده:**

| مدل | نتیجه |
|---|---|
| `PP-OCRv5_mobile_rec` / `PP-OCRv6_medium_rec` | دقیق‌تر، ولی حدود ۱.۵ تا ۲ برابر کندتر. فعلاً لازم نیست |
| `PP-OCRv4_mobile_det` | سریع‌تر، ولی در `test1.jpg` دیگر قطعه‌ی نام دارو با امتیاز بالای آستانه تولید نکرد. رد شد |

برای تعویض مدل فقط `ocr_det_model` یا `ocr_rec_model` را در `config.py` عوض کن.

### الگوریتم تطبیق

| ابزار | کاربرد |
|---|---|
| **RapidFuzz** `fuzz.ratio` | امتیاز شباهت دو رشته (۰ تا ۱۰۰). پیش‌فرض |
| `fuzz.WRatio` | گزینه‌ی جایگزین از طریق `fuzzy_scorer`. شباهت زیررشته‌ای را هم امتیاز می‌دهد و برای متن کوتاه روی نام بلند تطبیق اشتباه می‌سازد، پس پیش‌فرض نیست |

### کتابخانه‌ها (نسخه‌های تست‌شده روی Windows، Python 3.13.9، فقط CPU)

| کتابخانه | نسخه | کاربرد |
|---|---|---|
| `paddlepaddle` | 3.3.1 | موتور اجرای مدل‌ها |
| `paddleocr` | 3.7.0 | رابط OCR |
| `opencv-python` | 5.0.0.93 | خواندن عکس، تغییر اندازه، CLAHE، شارپ |
| `numpy` | 2.3.5 | آرایه‌ی تصویر |
| `rapidfuzz` | 3.14.6 | تطبیق تقریبی |
| `fastapi` | 0.165.0 | API |
| `uvicorn` | 0.54.0 | سرور |
| `python-multipart` | 0.0.34 | دریافت فایل در FastAPI |
| `pytest` | 9.2.0 | تست‌ها |

---

## 3. نصب

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

اختیاری: برای رد کردن بررسی اتصال به سرور مدل‌ها در شروع:

```powershell
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
```

بررسی اینکه OCR روی سیستم کار می‌کند:

```powershell
python scripts/ocr_check.py dataset/images/test1.jpg
```

> روی لینوکس یا مک ممکن است wheel های Paddle متفاوت لازم باشد. ببین [راهنمای نصب Paddle](https://www.paddlepaddle.org.cn/install/quick) و قبل از deploy حتماً یک بار روی همان سیستم تست کن.

---

## 4. راه‌های اجرا

### ۴.۱ دمو تعاملی (پیشنهادی برای نمایش)

```powershell
python scripts/demo.py
```

- Enter خالی: پنجره‌ی انتخاب فایل
- کشیدن فایل داخل ترمینال یا paste مسیر
- `q`: خروج
- مدل فقط یک بار لود می‌شود
- بعد از هر نتیجه، مراحل تأیید/رد/ورود دستی شبیه‌سازی می‌شود

می‌توان مسیر عکس‌ها را هم مستقیم داد: `python scripts/demo.py a.jpg b.jpg`

### ۴.۲ خط فرمان (یک عکس)

```powershell
python main.py dataset/images/test1.jpg
python main.py dataset/images/test1.jpg --debug
```

| آرگومان | پیش‌فرض | توضیح |
|---|---|---|
| `image` | | مسیر عکس JPG/PNG |
| `--debug` | خاموش | خروجی خام OCR، عملیات پیش‌پردازش و زمان مراحل را اضافه می‌کند |
| `--serve` | خاموش | به‌جای تشخیص، سرور API را بالا می‌آورد |
| `--host` | `127.0.0.1` | آدرس سرور. برای دسترسی از دستگاه دیگر `0.0.0.0` |
| `--port` | `8000` | پورت سرور |

کد خروج: `0` موفق، `2` خطا (مثلاً عکس نامعتبر).
هر بار اجرا مدل را از اول لود می‌کند (۳ تا ۱۲ ثانیه).

### ۴.۳ سرور API

```powershell
python main.py --serve --port 8000
# یا
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

تست سریع:

```powershell
curl.exe http://127.0.0.1:8000/api/health
curl.exe -X POST -F "file=@dataset/images/test1.jpg" http://127.0.0.1:8000/api/recognize
```

> در PowerShell حتماً `curl.exe` بنویس. `curl` یک alias برای دستور دیگری است.

### ۴.۴ از Python

```python
from main import recognize_medicine
result = recognize_medicine("dataset/images/test1.jpg", include_debug=False)
print(result["status"], result["suggested_medicine"])
```

مدل و دیتاست یک بار لود و بین فراخوانی‌ها استفاده می‌شوند.

### ۴.۵ اسکریپت‌ها

| اسکریپت | کار |
|---|---|
| `scripts/ocr_check.py <image> [--save-raw path.json]` | فقط OCR: ناحیه‌ها، اطمینان و کادرها. تطبیق انجام نمی‌دهد |
| `scripts/benchmark.py [--images ...] [--runs N] [--report path]` | زمان هر مرحله. پیش‌فرض ۳ اجرا برای هر عکس + یک warm-up اندازه‌گیری‌نشده |
| `scripts/make_split.py` | ساخت split قفل‌شده‌ی tuning/test |
| `scripts/evaluate.py` | ارزیابی دقت (بخش ۱۲) |
| `scripts/demo.py` | دمو تعاملی |

---

## 5. ساختار پروژه

```
Medicine_Recognition/
├── dataset/
│   ├── images/                       # عکس‌ها
│   └── annotations/
│       ├── labels.csv                # داروهای داخل دیتاست (image_name,medicine_label)
│       ├── labels.example.csv        # نمونه‌ی راهنما، نه داده‌ی واقعی
│       ├── negatives.csv             # (اختیاری) عکس داروهای بیرون از دیتاست
│       └── splits.csv                # (ساخته‌شده) تقسیم tuning/test
├── src/
│   ├── preprocessing/image_processor.py   # خواندن، اعتبارسنجی، resize، enhancement
│   ├── ocr/
│   │   ├── text_detector.py               # TextRegion و قوانین هندسی مجاورت
│   │   └── text_extractor.py              # موتور PaddleOCR و خروجی خام
│   ├── text_processing/normalizer.py      # نرمال‌سازی و اصلاح OCR
│   ├── matching/
│   │   ├── matcher.py                     # ساخت Query، تطبیق دقیق و تقریبی
│   │   └── candidate_ranker.py            # حذف تکراری، رتبه‌بندی، ابهام
│   ├── decision/
│   │   ├── decision_engine.py             # آستانه ← SUGGESTED / UNKNOWN
│   │   └── confirmation.py                # تأیید و ورود دستی
│   ├── database/dataset_loader.py         # labels.csv ← برچسب‌های یکتا
│   ├── evaluation/performance.py          # زمان‌سنج مراحل و آمار
│   ├── pipeline.py                        # هماهنگ‌کننده‌ی همه‌ی مراحل
│   └── api.py                             # FastAPI
├── scripts/  ocr_check.py  benchmark.py  evaluate.py  make_split.py  demo.py
├── tests/    test_api.py  test_dataset_loader.py  test_matching.py
│             test_normalizer.py  test_pipeline.py
├── outputs/                          # گزارش‌ها و requests.jsonl (ساخته می‌شود)
├── main.py
├── config.py
└── requirements.txt
```

---

## 6. دیتاست

### labels.csv

دقیقاً دو ستون لازم است:

```csv
image_name,medicine_label
img001.jpg,AMOXICILLIN
img002.jpg,IBUPROFEN
```

- چند عکس می‌توانند به یک دارو اشاره کنند.
- هر برچسب نرمال می‌شود (حروف بزرگ، حذف علائم) و بر اساس کلید «فقط حروف و اعداد» یکتا می‌شود. پس `amoxicillin` و `AMOXICILLIN` و `Amoxicillin` یک برچسب‌اند.
- ردیف‌هایی که `image_name` یا `medicine_label` خالی دارند نادیده گرفته می‌شوند.
- فایل با BOM (`utf-8-sig`) هم درست خوانده می‌شود.
- اگر هدر اشتباه باشد یا فایل نباشد، `DatasetError` می‌آید.
- `duplicate_rows` تعداد ردیف‌های تکراری هر دارو را می‌شمارد. `missing_images` عکس‌های ذکرشده ولی ناموجود را گزارش می‌دهد.
- هیچ پایگاه دارویی خارجی استفاده نمی‌شود. سیستم فقط نام‌های همین فایل را می‌شناسد.

### negatives.csv (اختیاری، برای ارزیابی)

عکس داروهایی که **در `labels.csv` نیستند**. جواب درست سیستم برای آن‌ها `UNKNOWN` است.

```csv
image_name,medicine_label
neg001.jpg,SOMEDRUG
neg002.jpg,
```

ستون `medicine_label` اختیاری است ولی اگر پر باشد برای گروه‌بندی در split استفاده می‌شود. اگر اسمش در `labels.csv` باشد اسکریپت‌ها خطا یا هشدار می‌دهند.

### splits.csv

توسط `make_split.py` ساخته می‌شود: `image_name, medicine_label, kind, split` (`kind`: `in_dataset` یا `negative`، `split`: `tuning` یا `test`).

---

## 7. پایپ‌لاین مرحله به مرحله

```
عکس → اعتبارسنجی → resize → OCR → نرمال‌سازی/Query → تطبیق دقیق
    → تطبیق تقریبی → رتبه‌بندی → تصمیم → SUGGESTED / UNKNOWN
    → تأیید / رد / ورود دستی
```

**۱. اعتبارسنجی** — پسوند مجاز، حجم ≤ `max_upload_mb`، فایل خالی نباشد، عکس رنگی و قابل decode باشد، کوچک‌ترین ضلع ≥ `min_image_side`.

**۲. پیش‌پردازش** — ضلع بلند به `max_image_side` (۹۶۰) کوچک می‌شود (`INTER_AREA`). اگر عکس از قبل کوچک‌تر باشد بزرگ نمی‌شود. اگر `enable_enhancement=True`، به ترتیب `enhancement_ops` اعمال می‌شود: `clahe` (کنتراست روی کانال L در فضای LAB، clipLimit 2.0، شبکه‌ی 8×8) و `sharpen` (unsharp mask سبک).

**۳. OCR** — یک موتور PaddleOCR موقع شروع ساخته و برای همه‌ی عکس‌ها استفاده می‌شود. نواحی با اطمینان < `ocr_min_confidence` از خروجی حذف می‌شوند (در خروجی خام می‌مانند). نواحی به ترتیب خواندن (بالا به پایین، چپ به راست) مرتب می‌شوند.

**۴. نرمال‌سازی** — NFKC، حروف بزرگ، علائم به فاصله، فاصله‌ها یکی. کلمات نویز (`ignored_tokens`)، توکن‌های تک‌حرفی یا بدون حرف و توکن‌هایی که بیش از ۴۰٪ رقم دارند از Query حذف می‌شوند. متن اصلی OCR هیچ‌وقت تغییر نمی‌کند.

**۵. اصلاح کنترل‌شده‌ی OCR** — ارقام `0→O`، `1→I`، `5→S`، `8→B` فقط در توکن‌هایی با شرایط زیر:
- حداقل یک رقم داشته باشد،
- حداقل ۴ حرف داشته باشد،
- حداکثر ۲۵٪ رقم باشد.

پس `B12` و `500` و `MG` دست نمی‌خورند. هم نسخه‌ی اصلی و هم اصلاح‌شده جست‌وجو می‌شوند.

**۶. ساخت Query** — برای هر ناحیه: توکن‌های تکی، جفت‌های مجاور، و کل دنباله (اگر ۲ تا ۶ توکن). اگر `combine_adjacent` روشن باشد، نواحی همسایه (هم‌ردیف یا دقیقاً زیر هم، طبق کادرها) ترکیب می‌شوند و فقط ۶ توکن آخر نگه داشته می‌شود. نواحی بدون کادر فقط وقتی جفت می‌شوند که پشت‌سرهم باشند. Query ها یکتا می‌شوند (کلید فشرده‌شده)، بر اساس اطمینان نزولی مرتب و به `max_queries` محدود می‌شوند.

**۷. تطبیق دقیق** — کلید Query برابر کلید یک برچسب ← امتیاز ۱۰۰، روش `EXACT`.

**۸. تطبیق تقریبی** — Query هایی که دقیق نبودند و کلیدشان ≥ `min_fuzzy_query_length` است با `fuzzy_scorer` روی همه‌ی برچسب‌ها سنجیده می‌شوند. فقط امتیاز ≥ `fuzzy_candidate_cutoff` و حداکثر `top_k_per_query` نتیجه نگه داشته می‌شود. روش `FUZZY`.

**۹. رتبه‌بندی** — برای هر دارو بهترین شاهد نگه داشته می‌شود (امتیاز بالاتر؛ در تساوی EXACT؛ بعد اطمینان OCR بالاتر). مرتب‌سازی: امتیاز نزولی، EXACT قبل از FUZZY، اطمینان OCR، نام. حداکثر `max_candidates` کاندیدا.

**۱۰. تشخیص ابهام** — اگر اختلاف نفر اول و دوم ≤ `ambiguity_margin` باشد، `ambiguous=true` و همه‌ی کاندیداهای در این فاصله در `alternatives` می‌آیند. این در هر دو حالت `SUGGESTED` و `UNKNOWN` محاسبه می‌شود.

**۱۱. تصمیم**
- بهترین امتیاز ≥ `match_threshold` ← `SUGGESTED` (`user_confirmation_required=true`)
- در غیر این صورت ← `UNKNOWN` (`manual_input_required=true`)، ولی `candidates` همچنان برمی‌گردد.

---

## 8. همه‌ی پارامترهای config.py

همه‌ی مقادیر قابل‌تنظیم در کلاس `Config` هستند. هیچ عدد جادویی دیگری در ماژول‌ها نیست.

### ۸.۱ مسیرها

| پارامتر | پیش‌فرض | توضیح |
|---|---|---|
| `root` | پوشه‌ی پروژه | ریشه‌ی مسیرها |
| `labels_path` | `dataset/annotations/labels.csv` | فایل برچسب‌ها |
| `images_dir` | `dataset/images` | پوشه‌ی عکس‌ها |
| `outputs_dir` | `outputs` | گزارش‌ها و `requests.jsonl` |

### ۸.۲ اعتبارسنجی آپلود

| پارامتر | پیش‌فرض | توضیح |
|---|---|---|
| `allowed_extensions` | `.jpg .jpeg .png` | فرمت‌های مجاز |
| `max_upload_mb` | `10.0` | سقف حجم فایل. بیشتر ← خطای ۴۱۳ |
| `min_image_side` | `120` | حداقل پیکسل برای هر ضلع. کمتر ← خطای ۴۰۰ |

### ۸.۳ پیش‌پردازش

| پارامتر | پیش‌فرض | توضیح |
|---|---|---|
| `max_image_side` | `960` | سقف ضلع بلند. کمتر کردن سریع‌تر ولی خوانایی کمتر (در آزمایش، ۸۰۰ و ۷۳۶ باعث گم‌شدن اسم دارو شد) |
| `enable_enhancement` | `False` | روشن‌کردن بهبود تصویر. در شرایط نور بد ممکن است کمک کند، ولی باید با ارزیابی سنجیده شود |
| `enhancement_ops` | `("clahe", "sharpen")` | ترتیب و نوع عملیات بهبود. فقط این دو معتبرند، غیر از این خطا می‌دهد |

### ۸.۴ OCR

| پارامتر | پیش‌فرض | توضیح |
|---|---|---|
| `ocr_language` | `"en"` | زبان. وقتی نام مدل‌ها داده شده، Paddle آن را نادیده می‌گیرد و در لاگ هشدار می‌دهد (بی‌خطر) |
| `ocr_min_confidence` | `0.30` | حداقل اطمینان هر ناحیه. پایین‌تر ← نویز بیشتر، بالاتر ← متن تار حذف می‌شود |
| `ocr_det_model` | `PP-OCRv5_mobile_det` | مدل تشخیص محل متن |
| `ocr_rec_model` | `en_PP-OCRv3_mobile_rec` | مدل خواندن متن |
| `ocr_rec_batch_size` | `16` | تعداد کادر در هر batch خواندن. ۳۲ سودی نداشت |
| `ocr_det_limit_side_len` | `960` | حد اندازه‌ی ورودی تشخیص |
| `ocr_det_limit_type` | `"max"` | `max` یعنی ضلع بلند به این حد محدود می‌شود و ضلع کوتاه ناخواسته بزرگ نمی‌شود |
| `ocr_num_threads` | `4` | تعداد ترد (`OMP_NUM_THREADS`). روی ماشین ۱۶ هسته‌ی تو از ۱۶ ترد سریع‌تر و پایدارتر بود. فقط اگر این متغیر محیطی از قبل تنظیم نشده باشد اعمال می‌شود |
| `ocr_use_mkldnn` | `False` | oneDNN. **روی این نسخه‌ی Paddle کرش می‌کند**، روشنش نکن |

موتور همیشه با این گزینه‌ها ساخته می‌شود: `use_doc_orientation_classify=False`، `use_doc_unwarping=False`، `use_textline_orientation=False` (برای سرعت، چرخش و صاف‌سازی سند خاموش است).

### ۸.۵ تطبیق و تصمیم

| پارامتر | پیش‌فرض | توضیح |
|---|---|---|
| `match_threshold` | `60.0` | آستانه‌ی پیشنهاد (۰ تا ۱۰۰). بالاتر ← پیشنهاد اشتباه کمتر، UNKNOWN بیشتر. روی مجموعه‌ی tuning تنظیم شود، نه test |
| `fuzzy_scorer` | `"ratio"` | هر نام از `rapidfuzz.fuzz`، مثلاً `ratio`، `WRatio`، `partial_ratio`. پیش‌فرض سخت‌گیرتر و قابل‌پیش‌بینی‌تر است |
| `fuzzy_candidate_cutoff` | `50.0` | کمترین امتیازی که وارد فهرست کاندیداها می‌شود |
| `min_query_length` | `3` | Query با کلید کوتاه‌تر از این اصلاً ساخته نمی‌شود |
| `min_fuzzy_query_length` | `4` | Query کوتاه‌تر از این فقط تطبیق دقیق می‌شود، نه تقریبی (برای جلوگیری از تطبیق تصادفی) |
| `top_k_per_query` | `5` | حداکثر نتیجه‌ی تقریبی برای هر Query |
| `max_candidates` | `5` | حداکثر کاندیدای برگشتی (سقف Top-5 در ارزیابی) |
| `max_queries` | `300` | سقف تعداد Query برای هر عکس (جلوگیری از کندی روی عکس‌های پرمتن) |
| `combine_adjacent` | `True` | ترکیب نواحی همسایه برای نام‌های شکسته روی چند خط |
| `ambiguity_margin` | `5.0` | اگر فاصله‌ی امتیاز دو کاندیدای اول از این کمتر یا مساوی باشد، `ambiguous=true` |
| `ignored_tokens` | مجموعه‌ی کلمات نویز | شکل‌ها (TABLET، SYRUP، …)، واحدها (MG، ML، IU، …)، بسته‌بندی و مقررات (EXP، BATCH، LOT، STORE، WARNING، PHARMA، LTD، …) و کلمات رایج (THE، AND، FOR، …) که نباید با نام دارو تطبیق بخورند |

### ۸.۶ سرویس

| پارامتر | پیش‌فرض | توضیح |
|---|---|---|
| `log_requests` | `True` | نوشتن هر تشخیص در `outputs/requests.jsonl` (بدون ذخیره‌ی عکس) |
| `service_title` | `Medicine Name Recognition` | عنوان FastAPI |

### ۸.۷ متد کمکی

`Config.image_files()` لیست مرتب‌شده‌ی عکس‌های `images_dir` با پسوند مجاز را برمی‌گرداند (بنچمارک از آن استفاده می‌کند).

---

## 9. ثابت‌ها و قوانین داخلی کد

این‌ها در `config.py` نیستند و داخل کد ثابت‌اند. اگر لازم شد تغییر دهی، باید کد را عوض کنی.

| مقدار | محل | توضیح |
|---|---|---|
| `MAX_RAW_UPLOAD_BYTES = 25 MB` | `api.py` | بیشینه‌ی بایتی که از آپلود خوانده می‌شود، قبل از بررسی `max_upload_mb` |
| `max_pending = 500` | `confirmation.py` | تعداد `request_id` نگه‌داری‌شده در RAM؛ قدیمی‌ترین حذف می‌شود |
| طول نام دستی | `confirmation.py` | حداقل ۲ و حداکثر ۱۲۰ کاراکتر (بعد از نرمال‌سازی) و دارای حرف انگلیسی |
| `request_id` | `api.py` | ۱۲ کاراکتر اول `uuid4().hex` |
| نگاشت اصلاح OCR | `normalizer.py` | `0→O`، `1→I`، `5→S`، `8→B` |
| شرط اصلاح OCR | `normalizer.py` | ≥ ۴ حرف، ≥ ۱ رقم، ≤ ۲۵٪ رقم |
| حذف توکن عددی | `normalizer.py` | توکن با > ۴۰٪ رقم یا تک‌حرفی یا بدون حرف |
| `same_row` | `text_detector.py` | هم‌پوشانی عمودی ≥ ۵۰٪ ارتفاع کوتاه‌تر و فاصله‌ی افقی ≤ ۱.۵ × ارتفاع بزرگ‌تر |
| `stacked` | `text_detector.py` | فاصله‌ی عمودی بین ‎−۰.۲ تا ۱.۰ برابر ارتفاع و هم‌پوشانی افقی ≥ ۳۰٪ عرض کوتاه‌تر |
| طول ترکیب | `matcher.py` | ترکیب نواحی همسایه: ۶ توکن آخر؛ دنباله‌ی کامل فقط وقتی ۲ تا ۶ توکن |
| قفل inference | `api.py` | `threading.Lock`: Paddle هم‌زمانی را تحمل نمی‌کند، پس درخواست‌ها نوبتی‌اند |
| `under_6_seconds` | `performance.py` | هدف زمانی ۶۰۰۰ms برای هر عکس |
| p95 | `performance.py` | `ordered[min(n-1, int(n*0.95))]` |

---

## 10. خروجی recognize (همه‌ی فیلدها)

| فیلد | نوع | توضیح |
|---|---|---|
| `status` | string | `SUGGESTED` یا `UNKNOWN` |
| `suggested_medicine` | string / null | نام پیشنهادی. در UNKNOWN برابر null |
| `match_score` | float / null | امتیاز شباهت ۰ تا ۱۰۰. در UNKNOWN برابر null |
| `matched_text` | string / null | متنی از عکس که به این دارو تطبیق خورد |
| `detected_texts` | list | همه‌ی متن‌های خوانده‌شده (اصلی، بدون تغییر) |
| `user_confirmation_required` | bool | در SUGGESTED برابر true |
| `manual_input_required` | bool | در UNKNOWN برابر true |
| `threshold` | float | آستانه‌ی استفاده‌شده |
| `ambiguous` | bool | آیا دو کاندیدای اول نزدیک‌اند |
| `alternatives` | list | نام داروهای نزدیک به اولی |
| `ocr_confidence` | float / null | اطمینان OCR برای تطبیق برنده (فقط SUGGESTED) |
| `candidates` | list | تا ۵ کاندیدا: `medicine_label`، `match_score`، `matched_text`، `ocr_confidence`، `matching_method` (`EXACT`/`FUZZY`). در UNKNOWN هم برمی‌گردد |
| `detected_regions` | list | هر ناحیه: `text`، `ocr_confidence`، `bounding_box` (`[x1,y1,x2,y2]`) |
| `processing_time_ms` | float | زمان کل (جمع مراحل، بدون لود مدل) |
| `timings_ms` | dict | زمان هر مرحله: `image_loading`، `preprocessing`، `ocr`، `normalization`، `exact_matching`، `fuzzy_matching`، `ranking`، `total` |
| `ocr_model_load_ms` | float / null | زمان لود مدل (یک بار) |
| `queries_matched` | int | تعداد Query ساخته‌شده |
| `labels_in_dataset` | int | تعداد برچسب یکتا |
| `request_id` | string | فقط در API؛ برای تأیید/ورود دستی |

با `--debug` (یا `include_debug=True`) این‌ها هم اضافه می‌شود: `preprocess_ops`، `raw_ocr`، `raw_ocr_engine_output`.

> زمان `normalization` در واقع زمان ساخت Query هاست.

---

## 11. API برای اپ موبایل

سرور **هیچ رابط وب ندارد**؛ `/docs` و `/redoc` عمداً خاموش‌اند. فقط JSON برمی‌گرداند و جز `request_id` کوتاه‌مدت وضعیتی نگه نمی‌دارد.

### جریان پیشنهادی در اپ

1. ارسال عکس به `POST /api/recognize`
2. نمایش `suggested_medicine`، `match_score` و `detected_texts` و پرسیدن «آیا درست است؟»
3. ارسال تأیید، رد یا نام دستی

### Endpoint ها

#### `GET /api/health`
`status`، `ocr_loaded`، `labels_in_dataset`، `ocr_model_load_ms`، `threshold`.

#### `POST /api/recognize`
فرم multipart با فیلد `file` (JPG/JPEG/PNG).

```bash
curl -X POST -F "file=@photo.jpg" http://HOST:8000/api/recognize
```

نمونه‌ی جواب `SUGGESTED` (خلاصه):

```json
{
  "status": "SUGGESTED",
  "suggested_medicine": "AMOXICILLIN",
  "match_score": 73.7,
  "matched_text": "AMO KCILL",
  "detected_texts": ["Amoxicillin", "500 MG", "CAPSULE"],
  "user_confirmation_required": true,
  "manual_input_required": false,
  "threshold": 60.0,
  "ambiguous": false,
  "alternatives": [],
  "candidates": [
    {"medicine_label": "AMOXICILLIN", "match_score": 73.7, "matched_text": "AMO KCILL",
     "ocr_confidence": 0.94, "matching_method": "FUZZY"}
  ],
  "processing_time_ms": 2766.0,
  "request_id": "b1c2d3e4f5a6"
}
```

جواب `UNKNOWN`: `suggested_medicine` و `match_score` برابر `null`، `user_confirmation_required=false`، `manual_input_required=true`.

#### `POST /api/confirm`

```json
{"request_id": "b1c2d3e4f5a6", "accepted": true}
```

- `accepted: true` ← `CONFIRMED` با نام پیشنهادی (اگر پیشنهادی نبوده باشد، خطای ۴۰۰)
- `accepted: false` ← `REJECTED` و `manual_input_required: true`
- `accepted: false` همراه با `medicine_name` ← مثل ورود دستی

#### `POST /api/manual-input`

```json
{"request_id": "b1c2d3e4f5a6", "medicine_name": "amoxicillin clavulanate"}
```

← `MANUAL_ACCEPTED` با نام نرمال‌شده (`AMOXICILLIN CLAVULANATE`). **`dataset_updated` همیشه `false` است.** نام دستی هرگز در دیتاست ذخیره نمی‌شود.

### وضعیت‌ها

| رویداد | `status` |
|---|---|
| تشخیص موفق | `SUGGESTED` / `UNKNOWN` |
| قبول پیشنهاد | `CONFIRMED` |
| رد پیشنهاد | `REJECTED` |
| ورود دستی | `MANUAL_ACCEPTED` |

### کدهای خطا

| کد | معنی |
|---|---|
| 400 | عکس خراب یا خیلی کوچک؛ نام دستی نامعتبر؛ تأیید بدون پیشنهاد |
| 404 | `request_id` ناشناخته (مثلاً بعد از ری‌استارت سرور) |
| 413 | فایل بزرگ‌تر از `max_upload_mb` |
| 415 | فرمت فایل پشتیبانی نمی‌شود |
| 500 | خطای OCR (سمت سرور) |

### لاگ

هر تشخیص در `outputs/requests.jsonl` یک خط می‌نویسد: `request_id`، `image_name`، `status`، `suggested_medicine`، `match_score`، `detected_texts`، `timings_ms`، `elapsed_ms`. عکس ذخیره نمی‌شود. با `log_requests=False` خاموش می‌شود.

---

## 12. ارزیابی

### ۱۲.۱ دو نوع عکس

| نوع | تعریف | جواب درست |
|---|---|---|
| `in_dataset` | دارو در `labels.csv` هست | پیشنهاد همان دارو |
| `negative` | دارو در `labels.csv` **نیست** | `UNKNOWN` |

Negative ها خطرناک‌ترین رفتار اپ را می‌سنجند: پیشنهاد اشتباه برای دارویی که اصلاً در لیست نیست.

### ۱۲.۲ ساخت split قفل‌شده

```powershell
python scripts/make_split.py
```

| آرگومان | پیش‌فرض | توضیح |
|---|---|---|
| `--labels` | `dataset/annotations/labels.csv` | |
| `--negatives` | `dataset/annotations/negatives.csv` | اگر نباشد، فقط in_dataset |
| `--output` | `dataset/annotations/splits.csv` | |
| `--test-fraction` | `0.3` | سهم test |
| `--seed` | `42` | برای تکرارپذیری |
| `--force` | خاموش | بازنویسی split موجود (نتایج قبلی دیگر قابل‌مقایسه نیستند) |

- همه‌ی عکس‌های یک دارو در یک طرف می‌افتند تا نشت اطلاعات نباشد.
- Negative ها و in_dataset ها جدا تقسیم می‌شوند تا هر دو در هر طرف باشند.
- اگر `splits.csv` موجود باشد، بدون `--force` دوباره ساخته نمی‌شود.
- لیست نام‌ها برای تطبیق همیشه کامل است؛ فقط عکس‌های ارزیابی تقسیم می‌شوند.

### ۱۲.۳ اجرای ارزیابی

```powershell
python scripts/evaluate.py                    # همه (labels + negatives در صورت وجود)
python scripts/evaluate.py --subset tuning    # حین تنظیم
python scripts/evaluate.py --subset test      # نهایی، فقط یک بار
```

| آرگومان | پیش‌فرض | توضیح |
|---|---|---|
| `--labels` | `labels.csv` | |
| `--negatives` | `negatives.csv` | |
| `--splits` | `splits.csv` | |
| `--subset` | `all` | `all` / `tuning` / `test` |
| `--images` | `dataset/images` | |
| `--limit` | بدون | فقط N ردیف اول |
| `--report` | `outputs/evaluation_report.json` (یا `..._<subset>.json`) | |

### ۱۲.۴ معیارها

**in-dataset (`overall`)**

| معیار | معنی |
|---|---|
| `correct_suggestions` | پیشنهاد درست |
| `incorrect_suggestions` | پیشنهاد اشتباه |
| `unknown_results` | UNKNOWN شد |
| `suggestion_rate` | سهم عکس‌هایی که پیشنهاد گرفتند |
| `wrong_suggestion_rate` | سهم پیشنهاد اشتباه از کل |
| `accuracy_on_suggestions` | درستی بین پیشنهادها |
| `errors` | عکس‌هایی که خطا دادند |

**Top-K (`top_k`)** — مستقل از آستانه: آیا دارو اصلاً بین کاندیداها هست؟ `top1_hit_rate`، `top3_hit_rate`، `top5_hit_rate`. اگر Top-1 پایین ولی Top-3 بالا باشد، نمایش چند گزینه در UI مشکل را کم می‌کند.

**Negative ها (`negatives`)** — `correct_unknown_rate` و `false_suggestion_rate`. برای اپ دارویی معمولاً پیشنهاد اشتباه بدتر از UNKNOWN است.

**برای هر عکس در `results`:** `kind`، `outcome`، `truth_rank`، `top1_label`، `top1_score`، `candidates` (برچسب و امتیاز)، `matched_text`، زمان. چون `top1_score` حتی در UNKNOWN ذخیره می‌شود، منحنی آستانه را می‌توان بدون اجرای دوباره‌ی OCR ساخت.

### ۱۲.۵ قواعد درست ارزیابی

- روی مجموعه‌ی tuning آستانه و تنظیمات را تغییر بده، نه روی test.
- test را فقط یک بار در انتهای کار اجرا کن.
- بدون Negative ها دقت واقعی اپ دیده نمی‌شود.
- با ۱ عکس برای هر دارو و عکس‌های تمیز، نتیجه خوش‌بینانه است. عکس گوشی با نور، زاویه و بازتاب متفاوت اضافه کن.
- قبلاً فقط ۲ عکس ارزیابی شده بود (۲ از ۲ درست)، که **برآورد دقت نیست**. دیتاست فعلی ۱۹۲ ردیف دارد و باید دوباره ارزیابی شود.

### ۱۲.۶ نتیجه‌ی ارزیابی فعلی

> این بخش را بعد از اجرای `evaluate.py` روی دیتاست ۱۹۲تایی پر کن.

| معیار | tuning | test |
|---|---|---|
| تعداد عکس | | |
| Top-1 / Top-3 / Top-5 | | |
| دقت بین پیشنهادها | | |
| UNKNOWN | | |
| پیشنهاد اشتباه برای Negative | | |

---

## 13. عملکرد و سرعت

اندازه‌گیری روی ماشین توسعه (`scripts/benchmark.py --runs 5` و `evaluate.py`)، سرور خاموش، لود مدل جدا.

| مرحله | میانگین (ms) |
|---|---|
| خواندن عکس | 17 |
| پیش‌پردازش | 2 |
| **OCR** | **2746** |
| نرمال‌سازی (ساخت Query) | 1 |
| تطبیق دقیق | 0 |
| تطبیق تقریبی | 0.1 |
| رتبه‌بندی | 0.1 |
| **کل (میانگین)** | **2766** |
| میانه | 2739 |
| بیشترین (۶ اجرا) | 3597 |
| لود مدل و دیتاست | ۳ تا ۱۲ ثانیه (یک بار) |

- `test1.jpg` حدود ۲ ثانیه و `test2.jpg` (متراکم، ۱۲ ناحیه) حدود ۳.۵ ثانیه.
- از طریق API کندترین اجرا ۴.۲۸ ثانیه بود. هدف ۶ ثانیه رعایت شد.
- OCR حدود ۹۹٪ زمان را مصرف می‌کند، پس فقط OCR بهینه شد.

### آنچه بهینه‌سازی را ممکن کرد
- ۴ ترد به‌جای همه‌ی هسته‌ها
- سقف صریح تشخیص ۹۶۰ پیکسل
- یک بار لود مدل و یک فراخوانی OCR برای هر عکس
- batch اندازه‌ی ۱۶

### آزمایش‌های ردشده (کاهش خوانایی نام دارو)
- تشخیص روی ۸۰۰ یا ۷۳۶ پیکسل: در `test1.jpg` نام دارو کلاً گم شد (۱ ناحیه به‌جای ۸)
- `text_det_box_thresh` برابر ۰.۷۰ یا ۰.۷۵: ناحیه‌ی `cillin` گم شد و نتیجه UNKNOWN می‌شد
- `PP-OCRv4_mobile_det`: سریع‌تر ولی نام زیر آستانه
- oneDNN: کرش (`ConvertPirAttribute2RuntimeAttribute not supported`)
- batch برابر ۳۲ و عکس کوچک‌تر از ۹۶۰: بی‌فایده یا کاهش خوانایی

---

## 14. تست‌ها

```powershell
python -m pytest tests -q
# 44 passed
```

| فایل | تعداد | چه چیزی را می‌سنجد |
|---|---|---|
| `test_api.py` | 10 | health، قرارداد خروجی، UNKNOWN، کدهای ۴۱۵/۴۰۰/۴۱۳/۴۰۴، تأیید، رد و ورود دستی، نام دستی خالی |
| `test_dataset_loader.py` | 5 | یکتاسازی، نرمال‌سازی، ردیف خالی، فایل ناموجود، هدر اشتباه، عکس گمشده |
| `test_matching.py` | 14 | تطبیق دقیق و تقریبی، اصلاح OCR، ترجیح داروی درست، نام شکسته روی دو خط، عدم چسباندن نواحی نامرتبط، آستانه، ابهام، رتبه‌بندی، کاندیدای تکراری |
| `test_normalizer.py` | 9 | حروف بزرگ، علائم، کلید فشرده، اصلاح OCR (و دست‌نخوردن `B12`/`500`/`MG`)، حذف نویز |
| `test_pipeline.py` | 6 | قرارداد کامل خروجی، UNKNOWN، خروجی خالی OCR، عکس خراب یا کوچک، debug |

تست‌ها از یک `StubExtractor` استفاده می‌کنند؛ یعنی OCR واقعی اجرا نمی‌شود و فقط منطق تطبیق، تصمیم و API سنجیده می‌شود. کیفیت OCR را `evaluate.py` می‌سنجد.

---

## 15. راهنمای اپ‌سازها و Deploy

### رفتار UI پیشنهادی

| حالت | نمایش |
|---|---|
| `SUGGESTED` | «آیا این دارو **X** است؟» با دکمه‌های بله/خیر و نمایش `detected_texts` |
| `SUGGESTED` + `ambiguous=true` | گزینه‌ی اول + `alternatives` برای انتخاب |
| `UNKNOWN` | «شناسایی نشد» + متن‌های خوانده‌شده + فیلد ورود دستی |
| خطای ۴۰۰ / ۴۱۵ / ۴۱۳ | پیام مناسب و درخواست عکس جدید |
| خطای ۴۰۴ در confirm | دوباره recognize کن یا نام را مستقیم نگه دار |

همیشه بنویسید: «این فقط یک **پیشنهاد** است. لطفاً با روی جعبه مطابقت دهید.»

راهنمای عکس‌گرفتن برای کاربر: نزدیک، صاف، نور خوب، بدون بازتاب، اسم دارو کامل در کادر.

### نکات Deploy

- **دسترسی شبکه:** پیش‌فرض `127.0.0.1` است و فقط خود همان کامپیوتر دسترسی دارد. برای شبکه `--host 0.0.0.0`.
- **احراز هویت ندارد.** قبل از انتشار: HTTPS، API key (مثلاً هدر `X-API-Key`) و rate limit اضافه شود. اگر کلید داخل اپ باشد قابل استخراج است؛ اگر اپ لاگین دارد توکن کاربر را چک کنید.
- **درخواست‌ها نوبتی‌اند** (قفل inference). هر کدام ۲ تا ۴ ثانیه؛ برای هم‌زمانی بالا به صف یا چند نمونه‌ی سرور نیاز است.
- **تأییدیه‌ها در RAM اند.** با ری‌استارت یا با چند worker، `request_id` ممکن است ۴۰۴ شود. اپ باید تحمل کند یا تأیید نهایی را سمت اپ نگه دارد.
- **سرور باید همیشه روشن بماند** (لود مدل ۳ تا ۱۲ ثانیه). پلتفرم‌های serverless که با هر درخواست بالا می‌آیند مناسب نیستند.
- **RAM:** هر worker یک کپی کامل از مدل دارد؛ با ۱ worker شروع کنید و مصرف را اندازه بگیرید.
- **اولین درخواست بعد از استارت** کمی کندتر است (warm-up در API وجود ندارد).
- **سیستم‌عامل:** همه‌چیز روی Windows و Python 3.13 تست شده؛ قبل از deploy روی لینوکس دوباره تست شود.

---

## 16. عیب‌یابی

| مشکل | علت و راه‌حل |
|---|---|
| `Annotation file not found` | `labels.csv` در `dataset/annotations/` نیست |
| `missing required column(s)` | هدر باید دقیقاً `image_name,medicine_label` باشد |
| هشدار `lang and ocr_version will be ignored` | بی‌خطر است (نام مدل‌ها صریح داده شده) |
| هشدار `OMP_NUM_THREADS set to 4` | عمدی است (`ocr_num_threads`) |
| کرش با `ConvertPirAttribute2RuntimeAttribute` | `ocr_use_mkldnn` را `False` نگه دار |
| لود اول خیلی طولانی | دانلود مدل‌ها؛ بار بعد از کش خوانده می‌شود |
| `/` و `/docs` ۴۰۴ می‌دهند | درست است؛ رابط وب عمداً وجود ندارد |
| `curl` در PowerShell کار نمی‌کند | `curl.exe` بنویس |
| پنجره‌ی انتخاب فایل در demo باز نمی‌شود | مسیر را دستی paste کن |
| نتیجه UNKNOWN برای دارویی که در لیست هست | با `ocr_check.py` ببین OCR متن را خوانده یا نه؛ اگر نخوانده مشکل از OCR/عکس است، اگر خوانده مشکل از تطبیق یا آستانه |
| اسم دارو در عکس گم می‌شود | اندازه‌ی عکس را کم نکن؛ عکس نزدیک‌تر و واضح‌تر بگیر |

---

## 17. محدودیت‌ها و نکات مسئولیتی

- فقط OCR **انگلیسی**؛ متن فارسی خوانده نمی‌شود.
- فقط متن مهم است؛ رنگ، شکل و ظاهر قرص استفاده نمی‌شود.
- نام‌ها فقط از `labels.csv` می‌آیند؛ پایگاه دارویی خارجی وجود ندارد. دارویی که در لیست نیست یا UNKNOWN می‌شود یا ممکن است با نام شبیه اشتباه تطبیق بخورد.
- امتیاز، **شباهت متنی** است نه احتمال درستی.
- عکس بد (تار، تاریک، بازتاب، کج) یا دیتاست کوچک ← UNKNOWN یا پیشنهاد اشتباه.
- نام‌های دستی ذخیره نمی‌شوند و دیتاست خودکار به‌روز نمی‌شود. به‌روزرسانی دیتاست نیاز به فرایند اعتبارسنجی جدا دارد.
- هیچ‌وقت نتیجه را بدون تأیید کاربر استفاده نکنید.
- این سیستم **توصیه‌ی پزشکی نمی‌دهد**؛ بر پایه‌ی خروجی آن نباید دوز، تداخل دارویی یا توصیه‌ی درمانی نمایش داده شود.
- زمان لود مدل عمداً از زمان هر عکس جدا گزارش می‌شود.

---

## 18. مسیر بهبود

| اولویت | مورد |
|---|---|
| بالا | اجرای ارزیابی روی ۱۹۲ ردیف، جمع‌آوری Negative ها و عکس‌های واقعی گوشی در شرایط مختلف |
| متوسط | تحلیل خطا (ذخیره‌ی `detected_texts` برای عکس‌های اشتباه و تفکیک مشکل OCR از تطبیق) |
| متوسط | منحنی آستانه روی tuning (آستانه‌های ۵۰ تا ۹۰) |
| متوسط | API key، HTTPS و rate limit |
| پایین | warm-up مدل در startup |
| پایین | ذخیره‌ی تأییدیه‌ها در SQLite یا Redis |
| پایین | مدل recognition دقیق‌تر (`PP-OCRv5_mobile_rec`) در صورت نیاز به دقت بیشتر |
| آینده | پشتیبانی از متن فارسی، افزایش دیتاست |
