"""
convert_to_ipynb.py — Convert notebook source .py → .ipynb

Dùng marker chuẩn của jupytext (percent format):
    # %% [markdown]   → markdown cell
    # %%              → code cell

Usage:
    python convert_to_ipynb.py instantid_tpu_v5e.py
    python convert_to_ipynb.py instantid_tpu_v5e.py -o my_output.ipynb
    python convert_to_ipynb.py                        # convert tất cả *.py trong thư mục này
"""

import argparse
import pathlib
import re
import sys


def convert(src_path: pathlib.Path, dst_path: pathlib.Path | None = None) -> pathlib.Path:
    """Parse percent-format .py và tạo .ipynb."""
    try:
        import nbformat
    except ImportError:
        print("nbformat chưa được cài. Chạy:  pip install nbformat")
        sys.exit(1)

    src = src_path.read_text(encoding='utf-8')
    lines = src.splitlines()

    cells = []
    current_type: str = 'skip'   # 'skip' | 'code' | 'markdown'
    current_lines: list[str] = []

    def flush():
        nonlocal current_lines
        if not current_lines and current_type == 'skip':
            return
        # Bỏ trailing blank lines
        while current_lines and not current_lines[-1].strip():
            current_lines.pop()
        if not current_lines:
            current_lines = []
            return
        if current_type == 'markdown':
            # Bỏ dấu "# " đầu mỗi dòng
            content = '\n'.join(
                line[2:] if line.startswith('# ') else
                line[1:] if line.startswith('#') else line
                for line in current_lines
            )
            cells.append(nbformat.v4.new_markdown_cell(content))
        elif current_type == 'code':
            cells.append(nbformat.v4.new_code_cell('\n'.join(current_lines)))
        current_lines = []

    # Regex cho cell marker
    MD_MARKER  = re.compile(r'^# %%\s*\[markdown\]')
    CODE_MARKER = re.compile(r'^# %%(?!\s*\[)')

    for line in lines:
        if MD_MARKER.match(line):
            flush()
            current_type = 'markdown'
        elif CODE_MARKER.match(line):
            flush()
            current_type = 'code'
        else:
            if current_type != 'skip':
                current_lines.append(line)

    flush()  # flush cell cuối

    if not cells:
        print(f"⚠️  Không tìm thấy cell marker nào trong {src_path.name}")
        print("    Đảm bảo file dùng marker:  # %%  và  # %% [markdown]")
        sys.exit(1)

    nb = nbformat.v4.new_notebook(cells=cells)

    if dst_path is None:
        dst_path = src_path.with_suffix('.ipynb')

    nbformat.write(nb, str(dst_path))
    return dst_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert percent-format .py notebook source → .ipynb'
    )
    parser.add_argument(
        'source', nargs='?',
        help='File .py cần convert (mặc định: tất cả *.py trong thư mục hiện tại)')
    parser.add_argument(
        '-o', '--output', default=None,
        help='Tên file output .ipynb (chỉ dùng khi convert 1 file)')
    args = parser.parse_args()

    here = pathlib.Path(__file__).parent

    if args.source:
        src = pathlib.Path(args.source)
        if not src.is_absolute():
            src = here / src
        if not src.exists():
            print(f"❌ Không tìm thấy file: {src}")
            sys.exit(1)
        dst = pathlib.Path(args.output) if args.output else None
        out = convert(src, dst)
        print(f"✅  {src.name}  →  {out.name}")
    else:
        # Convert tất cả .py (trừ chính file này)
        py_files = [p for p in here.glob('*.py')
                    if p.name != pathlib.Path(__file__).name]
        if not py_files:
            print("Không có file .py nào để convert.")
            sys.exit(0)
        for src in sorted(py_files):
            try:
                out = convert(src)
                print(f"✅  {src.name}  →  {out.name}")
            except SystemExit:
                pass   # file không có marker → skip, tiếp tục


if __name__ == '__main__':
    main()