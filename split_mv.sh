#!/bin/bash

# 1. 파일 목록이 들어있는 텍스트 파일 이름
LIST_FILE=$1

# 2. 목적지 디렉토리 (사용자 입력 또는 기본값 설정)
# 스크립트 실행 시 인자로 받을 수 있게 구성했습니다. (예: ./move_files.sh /path/to/dest)
DEST_DIR=$2

# 목적지 디렉토리가 입력되지 않았을 경우 안내 후 종료
if [ -z "$DEST_DIR" ]; then
    echo "사용법: $0 <목적지_디렉토리>"
    exit 1
fi

# 목적지 디렉토리가 없으면 생성
if [ ! -d "$DEST_DIR" ]; then
    echo "디렉토리가 존재하지 않아 생성합니다: $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

# 3. 파일 목록을 읽어서 이동 처리
if [ -f "$LIST_FILE" ]; then
    echo "파일 이동을 시작합니다..."
    
    # line-by-line으로 읽기 (-r 옵션은 백슬래시 해석 방지)
    while IFS= read -r filename || [ -n "$filename" ]; do
        # 공백 및 개행 제거
        filename=$(echo "$filename" | xargs)
        
        # 빈 줄은 건너뜀
        if [ -z "$filename" ]; then
            continue
        fi

        # 파일이 실제로 존재하는지 확인 후 이동
        if [ -f "$filename" ]; then
            mv "$filename" "$DEST_DIR/"
            echo "[성공] 이동 완료: $filename -> $DEST_DIR/"
        else
            echo "[실패] 파일을 찾을 수 없음: $filename"
        fi
    done < "$LIST_FILE"
    
    echo "모든 작업이 완료되었습니다."
else
    echo "오류: $LIST_FILE 파일을 찾을 수 없습니다."
    exit 1
fi