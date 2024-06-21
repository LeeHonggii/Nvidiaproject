# Cross Editor Ai
## NVIDIA-2차-프로젝트-교차편집ai


## 프로젝트 설명

## 요구사항
- Node.js (버전 16)
- Python (버전 3.11)
- cuda 12.2
- npm 또는 yarn

## 설치 방법

### 1. 레포지토리 클론
```bash
git clone https://github.com/LeeHonggii/Nvidiaproject.git
cd Nvidiaproject
```

### 2. 의존성 설치
프로젝트의 루트 디렉토리에서 다음 명령어를 실행합니다.
```bash
cd react-sever
npm install
```

### 3. 백엔드 의존성 설치
`src/server` 디렉토리로 이동하여 의존성을 설치합니다.
```bash
cd src/server
npm install
```

### 4. 서버 실행
프로젝트의 루트 디렉토리에서 다음 명령어를 실행합니다.
```bash
npm start
```

백엔드 서버를 실행합니다.
```bash
cd src/server
node index.js
```

### 5. Python 스크립트 실행
Python 으로만 실행을 원할경우, 다음 명령어로 Python 스크립트를 실행할 수 있습니다.
data 폴더에 영상을 넣고
```bash
python3 main.py
```

## 프로젝트 구조
```plaintext
Nvidiaproject/
│
├── data /
│   └──영상파일들
│
├── pose / 
│
├── react-sever/
│   ├── src/
│   │   ├── server/
│   │   │   └── index.js
│   │   ├── components/
│   │   ├── App.js
│   │   └── ... (기타 프론트엔드 파일)
│   ├── public/
│   ├── package.json
│   └── ... (기타 프론트엔드 파일)
│
├── .gitignore
├── README.md
├── main.py
└── ... (기타 루트 파일)
```

## 사용 방법
- `localhost:3000`에서 프론트엔드 애플리케이션에 접속할 수 있습니다.
- 서버가 실행 중이면 `localhost:5000`에서 백엔드 엔드포인트에 접속할 수 있습니다.
