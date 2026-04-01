"""Vercel 서버리스 진입점 — Flask 앱을 Vercel이 실행할 수 있도록 연결."""

import sys
import os

# 프로젝트 루트를 import 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Vercel은 이 변수를 찾아서 WSGI 앱으로 실행함
app = app
