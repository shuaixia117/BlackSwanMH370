"""pytest 配置：把项目根目录加入导入路径。
pytest configuration: put the project root on the import path.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
