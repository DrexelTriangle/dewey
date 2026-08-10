import argparse
import json
import multiprocessing as mp
import os
import re
import unicodedata
import time
from datetime import date as date_cls
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path, pdfinfo_from_path

DEFAULT_PDF = 'triangle_pdfs/2024_Triangles/feb-2-2024-final.pdf'
DEFAULT_DPI = 150
WORKER_RSS_GB = 2.5
IMAGE_DPI = 300
MIN_FIGURE_PX = 120

MONTHS = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}
ISO_DATE = re.compile(r'(?P<year>(?:18|19|20)\d{2})-(?P<month>\d{2})-(?P<day>\d{2})')
EDITION_DATE = re.compile(
    r'(?P<month>[a-z]{3})[a-z]*[-_ ](?P<day>\d{1,2})[-_ ](?P<year>\d{4})', re.I
)
BYLINE = re.compile(r'^(?:by|written by)\s+(.+)$', re.I)

_ENGINE = None
_OPTS = {}


def edition_date(pdf_path):
    stem = Path(pdf_path).stem

    iso = ISO_DATE.search(stem)
    if iso:
        try:
            return date_cls(
                int(iso.group('year')), int(iso.group('month')), int(iso.group('day'))
            ).isoformat()
        except ValueError:
            pass

    m = EDITION_DATE.search(stem)
    if not m:
        return None
    month = MONTHS.get(m.group('month').lower())
    if not month:
        return None
    try:
        return date_cls(int(m.group('year')), month, int(m.group('day'))).isoformat()
    except ValueError:
        return None


def build_engine(threads=4, models='fast', device='cpu'):
    from paddleocr import PPStructureV3

    tier = 'mobile' if models == 'fast' else 'server'
    return PPStructureV3(
        device=device,
        text_detection_model_name=f'PP-OCRv5_{tier}_det',
        text_recognition_model_name=f'PP-OCRv5_{tier}_rec',
        text_recognition_batch_size=2,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_table_recognition=False,
        use_formula_recognition=False,
        use_seal_recognition=False,
        use_chart_recognition=False,
        cpu_threads=threads,
    )


def _init_worker(dpi, threads, models, device, image_opts):
    global _ENGINE, _OPTS
    os.environ.setdefault('OMP_NUM_THREADS', str(threads))
    _OPTS = {'dpi': dpi, **image_opts}
    _ENGINE = build_engine(threads, models, device)


def _render(pdf, page_no, dpi):
    page = convert_from_path(pdf, dpi=dpi, first_page=page_no, last_page=page_no)[0]
    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    del page
    return img


def block_label(block):
    if isinstance(block, dict):
        value = block.get('block_label') or block.get('label') or ''
    else:
        value = getattr(block, 'label', '')
    return str(value).lower()


def block_text(block):
    if isinstance(block, dict):
        value = block.get('block_content') or block.get('content') or ''
    else:
        value = getattr(block, 'content', '')
    if isinstance(value, list):
        value = '\n'.join(str(v) for v in value)
    return clean_text(str(value))


def block_bbox(block):
    if isinstance(block, dict):
        value = block.get('block_bbox') or block.get('bbox') or block.get('coordinate')
    else:
        value = getattr(block, 'bbox', None)
    if value is None:
        return [0, 0, 0, 0]
    if hasattr(value, 'tolist'):
        value = value.tolist()
    return [int(v) for v in value[:4]]


def clean_text(value):
    value = unicodedata.normalize('NFKC', value)
    value = value.replace('\r', '\n')
    lines = [re.sub(r'\s+', ' ', line).strip() for line in value.splitlines()]
    return '\n'.join(line for line in lines if line)


def is_figure(block):
    label = block['label']
    return any(word in label for word in ('image', 'figure', 'chart'))


def is_heading(block):
    label = block['label']
    text = block['text']
    if not text or len(text) > 160:
        return False
    if any(word in label for word in ('title', 'headline', 'header')):
        return True
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.75


def parse_blocks(page_no, raw_blocks):
    blocks = []
    for raw in raw_blocks:
        block = {
            'page': page_no,
            'label': block_label(raw),
            'text': block_text(raw),
            'bbox': block_bbox(raw),
        }
        if block['text'] or is_figure(block):
            blocks.append(block)
    return blocks


def _crop_figures(pdf, page_no, blocks, page_img):
    figures = [block for block in blocks if is_figure(block)]
    if not figures or not _OPTS.get('image_dir'):
        return

    scale = _OPTS['image_dpi'] / _OPTS['dpi']
    hires = page_img if scale == 1 else _render(pdf, page_no, _OPTS['image_dpi'])
    height, width = hires.shape[:2]

    image_dir = Path(_OPTS['image_dir'])
    image_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf).stem

    for index, fig in enumerate(figures):
        x0, y0, x1, y1 = (int(v * scale) for v in fig['bbox'])
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if (x1 - x0) < _OPTS['min_figure'] or (y1 - y0) < _OPTS['min_figure']:
            continue

        name = f'{stem}_p{page_no:02d}_{index}.jpg'
        cv2.imwrite(
            str(image_dir / name), hires[y0:y1, x0:x1],
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        fig['src'] = f'{_OPTS["image_prefix"]}{name}'
        fig['width'] = x1 - x0
        fig['height'] = y1 - y0


def _process_page(task):
    pdf, page_no = task
    img = _render(pdf, page_no, _OPTS['dpi'])

    blocks = []
    for result in _ENGINE.predict(img):
        blocks.extend(parse_blocks(page_no, result['parsing_res_list']))

    _crop_figures(pdf, page_no, blocks, img)
    return pdf, page_no, blocks


def slugify(value):
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode()
    value = re.sub(r'[^a-zA-Z0-9]+', '-', value.lower()).strip('-')
    return value or 'article'


def excerpt(value, limit=180):
    value = re.sub(r'\s+', ' ', value).strip()
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(' ', 1)[0] + '...'


def split_byline(text):
    lines = text.splitlines()
    authors = []
    body = []
    for line in lines:
        match = BYLINE.match(line)
        if match:
            authors.extend(
                name.strip()
                for name in re.split(r'\s*(?:,| and |&)\s*', match.group(1))
                if name.strip()
            )
        else:
            body.append(line)
    return authors, '\n'.join(body).strip()


def make_article(raw, date, edition, index):
    body = '\n\n'.join(raw['body']).strip()
    authors, body = split_byline(body)
    figures = sorted(raw['figures'], key=lambda f: f.get('width', 0) * f.get('height', 0), reverse=True)
    title = raw['title'] or excerpt(body, 80) or f'{Path(edition).stem} article {index}'
    slug = f'{slugify(Path(edition).stem)}-{index:03d}-{slugify(title)[:80]}'
    return {
        'title': title,
        'slug': slug,
        'published_date': f'{date}T12:00:00',
        'authors': authors,
        'categories_list': [],
        'description': excerpt(body),
        'content': body,
        'featured_image': figures[0].get('src') if figures else None,
        'archive': {
            'edition': edition,
            'pages': sorted(raw['pages']),
            'figures': figures,
        },
    }


def build_articles(blocks, date, edition, min_chars):
    articles = []
    current = None
    pending_figures = []

    def finish():
        nonlocal current
        if not current:
            return
        text = ' '.join(current['body'])
        if len(text) >= min_chars:
            articles.append(make_article(current, date, edition, len(articles) + 1))
        current = None

    for block in blocks:
        if is_figure(block):
            if current:
                current['figures'].append(block)
                current['pages'].add(block['page'])
            else:
                pending_figures.append(block)
            continue

        text = block['text']
        if not text:
            continue

        if is_heading(block):
            finish()
            current = {
                'title': text.splitlines()[0],
                'body': [],
                'figures': pending_figures,
                'pages': {block['page']},
            }
            pending_figures = []
            extra = '\n'.join(text.splitlines()[1:]).strip()
            if extra:
                current['body'].append(extra)
            continue

        if current is None:
            current = {
                'title': '',
                'body': [],
                'figures': pending_figures,
                'pages': {block['page']},
            }
            pending_figures = []
        current['body'].append(text)
        current['pages'].add(block['page'])

    finish()
    return articles


def collect_pdfs(paths):
    found = []
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_dir():
            found.extend(sorted(path.rglob('*.pdf')))
        elif path.exists():
            found.append(path)
    return list(dict.fromkeys(found))


def resolve_device(choice):
    if choice == 'cpu':
        return 'cpu'
    try:
        import paddle

        usable = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        usable = False

    if usable:
        return 'gpu'
    if choice == 'gpu':
        raise SystemExit('--device gpu requested but no usable CUDA device found')
    return 'cpu'


def default_workers():
    try:
        with open('/proc/meminfo') as fh:
            available_gb = int(
                next(l for l in fh if l.startswith('MemAvailable')).split()[1]
            ) / 1e6
    except (OSError, StopIteration):
        available_gb = 8.0
    by_memory = max(1, int((available_gb - 2) / WORKER_RSS_GB))
    by_cpu = max(1, (os.cpu_count() or 4) // 4)
    return max(1, min(by_memory, by_cpu))


def main():
    parser = argparse.ArgumentParser(description='OCR a Triangle edition into Scalene article JSON.')
    parser.add_argument(
        'pdfs', nargs='*', default=[DEFAULT_PDF],
        help='PDF files, or directories to search recursively',
    )
    parser.add_argument('-o', '--out', help='JSON output path (single PDF only)')
    parser.add_argument(
        '--out-dir', help='write <stem>.json here instead of beside each PDF',
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='leave editions that already have JSON alone, so a run resumes',
    )
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI)
    parser.add_argument('--pages', help='page range to process, e.g. 1-4')
    parser.add_argument(
        '--workers', type=int, default=0,
        help='pages to parse in parallel (default: fit to free RAM and cores)',
    )
    parser.add_argument(
        '--threads', type=int, default=4,
        help='Paddle CPU threads per worker',
    )
    parser.add_argument(
        '--models', choices=('fast', 'accurate'), default='fast',
        help='fast: mobile OCR models, 3x throughput, occasional headline '
             'artifacts. accurate: server models.',
    )
    parser.add_argument(
        '--device', choices=('auto', 'gpu', 'cpu'), default='auto',
        help='auto uses the GPU when one is usable, else CPU',
    )
    parser.add_argument(
        '--no-images', action='store_true', help='skip figure extraction',
    )
    parser.add_argument(
        '--min-chars', type=int, default=200,
        help='drop extracted articles shorter than this (page furniture)',
    )
    args = parser.parse_args()

    pdfs = collect_pdfs(args.pdfs)
    if not pdfs:
        parser.error('no PDFs found')
    if args.out and len(pdfs) > 1:
        parser.error('--out takes a single PDF; use --out-dir for a batch')

    def out_path(pdf):
        if args.out:
            return Path(args.out)
        if args.out_dir:
            return Path(args.out_dir) / f'{pdf.stem}.json'
        return pdf.with_suffix('.json')

    if args.skip_existing:
        pdfs = [p for p in pdfs if not out_path(p).exists()]
        if not pdfs:
            print('nothing to do; all outputs exist')
            return

    tasks = []
    for pdf in pdfs:
        first, last = 1, pdfinfo_from_path(str(pdf))['Pages']
        if args.pages:
            bounds = args.pages.split('-')
            first, last = int(bounds[0]), int(bounds[-1])
        tasks.extend((str(pdf), p) for p in range(first, last + 1))

    json_root = Path(args.out).parent if args.out else (
        Path(args.out_dir) if args.out_dir else pdfs[0].parent
    )
    image_opts = {
        'image_dir': None if args.no_images else str(json_root / 'images'),
        'image_dpi': IMAGE_DPI,
        'image_prefix': 'images/',
        'min_figure': MIN_FIGURE_PX,
    }

    device = resolve_device(args.device)
    fallback = 1 if device == 'gpu' else default_workers()
    workers = min(args.workers or fallback, len(tasks))
    started = time.perf_counter()
    print(
        f'{len(pdfs)} edition(s), {len(tasks)} pages, {device}, '
        f'{workers} workers x {args.threads} threads',
        flush=True,
    )

    by_pdf = {str(p): {} for p in pdfs}
    done = 0
    pool = None
    if workers == 1:
        _init_worker(args.dpi, args.threads, args.models, device, image_opts)
        results = (_process_page(t) for t in tasks)
    else:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(
            workers, initializer=_init_worker,
            initargs=(args.dpi, args.threads, args.models, device, image_opts),
        )
        results = pool.imap_unordered(_process_page, tasks)

    try:
        for pdf_str, page_no, blocks in results:
            by_pdf[pdf_str][page_no] = blocks
            done += 1
            print(f'  {done}/{len(tasks)}  {Path(pdf_str).stem} p{page_no}', flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    total_articles = 0
    for pdf in pdfs:
        by_page = by_pdf[str(pdf)]
        blocks = [block for page_no in sorted(by_page) for block in by_page[page_no]]
        date = edition_date(pdf) or date_cls.today().isoformat()
        payload = build_articles(blocks, date, pdf.name, args.min_chars)
        total_articles += len(payload)

        destination = out_path(pdf)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f'{len(payload):4d} articles -> {destination}')

    elapsed = time.perf_counter() - started
    print(
        f'\n{total_articles} articles from {len(tasks)} pages in {elapsed:.1f}s '
        f'({elapsed / len(tasks):.2f}s/page)'
    )


if __name__ == '__main__':
    main()
