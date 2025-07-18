#!/usr/bin/env python3
"""
PKL to CSV Converter Script
이 스크립트는 BIPD/data 폴더의 pkl 파일들을 CSV 형식으로 변환합니다.
"""

import os
import pickle
import pandas as pd
from pathlib import Path
import argparse
import sys

def load_pkl_file(file_path):
    """
    pkl 파일을 로드하고 데이터 구조를 반환합니다.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_data_structure(data, file_name):
    """
    데이터 구조를 분석하고 출력합니다.
    """
    print(f"\n=== {file_name} 데이터 구조 분석 ===")
    print(f"Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Dictionary keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"  {key}: {type(value)} - {getattr(value, 'shape', 'N/A')}")
    elif isinstance(data, pd.DataFrame):
        print(f"DataFrame shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Index: {data.index}")
    elif isinstance(data, list):
        print(f"List length: {len(data)}")
        if len(data) > 0:
            print(f"First element type: {type(data[0])}")
    else:
        print(f"Data shape: {getattr(data, 'shape', 'N/A')}")

def convert_to_csv(data, output_path):
    """
    데이터를 CSV 파일로 변환합니다.
    """
    try:
        if isinstance(data, pd.DataFrame):
            # DataFrame인 경우 직접 CSV로 저장
            data.to_csv(output_path, index=True)
            print(f"✓ DataFrame을 CSV로 저장: {output_path}")
            
        elif isinstance(data, dict):
            # Dictionary인 경우 각 키에 대해 개별 CSV 파일 생성
            base_path = Path(output_path).parent
            base_name = Path(output_path).stem
            
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    csv_path = base_path / f"{base_name}_{key}.csv"
                    value.to_csv(csv_path, index=True)
                    print(f"✓ {key} 데이터를 CSV로 저장: {csv_path}")
                elif hasattr(value, 'to_csv'):
                    csv_path = base_path / f"{base_name}_{key}.csv"
                    value.to_csv(csv_path, index=True)
                    print(f"✓ {key} 데이터를 CSV로 저장: {csv_path}")
                else:
                    # 다른 형태의 데이터는 DataFrame으로 변환 시도
                    try:
                        df = pd.DataFrame(value)
                        csv_path = base_path / f"{base_name}_{key}.csv"
                        df.to_csv(csv_path, index=True)
                        print(f"✓ {key} 데이터를 DataFrame으로 변환 후 CSV로 저장: {csv_path}")
                    except Exception as e:
                        print(f"✗ {key} 데이터를 CSV로 변환할 수 없음: {e}")
                        
        elif isinstance(data, list):
            # List인 경우 DataFrame으로 변환 시도
            try:
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=True)
                print(f"✓ List를 DataFrame으로 변환 후 CSV로 저장: {output_path}")
            except Exception as e:
                print(f"✗ List를 CSV로 변환할 수 없음: {e}")
                
        else:
            # 다른 형태의 데이터는 DataFrame으로 변환 시도
            try:
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=True)
                print(f"✓ 데이터를 DataFrame으로 변환 후 CSV로 저장: {output_path}")
            except Exception as e:
                print(f"✗ 데이터를 CSV로 변환할 수 없음: {e}")
                
    except Exception as e:
        print(f"✗ CSV 변환 중 오류 발생: {e}")

def main():
    parser = argparse.ArgumentParser(description='PKL 파일을 CSV로 변환합니다.')
    parser.add_argument('--input-dir', default='.', help='PKL 파일이 있는 디렉토리 (기본: . - 현재 디렉토리)')
    parser.add_argument('--output-dir', default='./csv_output', help='CSV 파일을 저장할 디렉토리 (기본: ./csv_output)')
    parser.add_argument('--analyze-only', action='store_true', help='데이터 구조만 분석하고 변환하지 않음')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ 입력 디렉토리가 존재하지 않습니다: {input_dir}")
        sys.exit(1)
    
    # 출력 디렉토리 생성
    if not args.analyze_only:
        output_dir.mkdir(exist_ok=True)
        print(f"📁 출력 디렉토리: {output_dir}")
    
    # PKL 파일 찾기
    pkl_files = list(input_dir.glob("*.pkl"))
    
    if not pkl_files:
        print(f"❌ {input_dir}에서 PKL 파일을 찾을 수 없습니다.")
        sys.exit(1)
    
    print(f"🔍 발견된 PKL 파일: {len(pkl_files)}개")
    
    for pkl_file in pkl_files:
        print(f"\n📊 처리 중: {pkl_file.name}")
        
        # PKL 파일 로드
        data = load_pkl_file(pkl_file)
        if data is None:
            continue
            
        # 데이터 구조 분석
        analyze_data_structure(data, pkl_file.name)
        
        # CSV로 변환 (분석 전용 모드가 아닌 경우)
        if not args.analyze_only:
            output_path = output_dir / f"{pkl_file.stem}.csv"
            convert_to_csv(data, output_path)
    
    print(f"\n✅ 작업 완료!")
    if not args.analyze_only:
        print(f"📁 CSV 파일들이 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()
