const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const { exec } = require('child_process');
const fs = require('fs');
const WebSocket = require('ws');

const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(__dirname, '../../../data'));
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  },
});

const upload = multer({ storage: storage });

app.post('/upload', upload.single('file'), (req, res) => {
  res.json({ message: 'File uploaded successfully', filename: req.file.originalname });
});

app.post('/analyze', (req, res) => {
  const mainPyPath = path.join(__dirname, '../../../main.py');
  exec('which python', (error, pythonPath, stderr) => {
    if (error) {
      console.error(`which error: ${error}`);
      return res.status(500).json({ error: 'Error finding Python executable' });
    }

    pythonPath = pythonPath.trim();
    const pythonProcess = exec(`${pythonPath} ${mainPyPath}`, { maxBuffer: 1024 * 1024 * 10 });

    pythonProcess.stdout.on('data', (data) => {
      console.log(`stdout: ${data}`); // 모든 stdout 로그 출력
      const lines = data.split('\n');
      lines.forEach(line => {
        const koreanRegex = /[가-힣]/;
        if (koreanRegex.test(line)) {
          const cleanLine = line.split('%')[0]; // % 이후의 내용 제거
          if (ws) {
            ws.send(JSON.stringify({ type: 'stdout', data: cleanLine }));
          }
        }
      });
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`); // 모든 stderr 로그 출력
      const lines = data.split('\n');
      lines.forEach(line => {
        const koreanRegex = /[가-힣]/;
        if (koreanRegex.test(line)) {
          const cleanLine = line.split('%')[0]; // % 이후의 내용 제거
          if (ws) {
            ws.send(JSON.stringify({ type: 'stderr', data: cleanLine }));
          }
        }
      });
    });

    pythonProcess.on('close', (code) => {
      res.json({ message: 'Video analysis complete', output: 'combined_video.mp4' });
    });
  });
});

app.delete('/delete', (req, res) => {
  const { filename } = req.body;

  if (!filename) {
    return res.status(400).json({ error: 'Filename is required' });
  }

  const filePath = path.join(__dirname, '../../../data', filename);

  fs.unlink(filePath, (err) => {
    if (err) {
      return res.status(500).json({ error: 'Error deleting file' });
    }
    res.json({ message: 'File deleted successfully' });
  });
});

app.get('/final-video', (req, res) => {
  const videoPath = path.join(__dirname, './combined_video.mp4');
  if (fs.existsSync(videoPath)) {
    res.sendFile(videoPath);
  } else {
    res.status(404).json({ error: 'Final video not found' });
  }
});

// 특정 디렉토리의 모든 비디오 파일 삭제 엔드포인트 추가
app.delete('/delete-videos', (req, res) => {
    const videoDirectory = path.join(__dirname, '../../../data');
  
    fs.readdir(videoDirectory, (err, files) => {
      if (err) {
        return res.status(500).json({ error: 'Error reading directory' });
      }
  
      const videoFiles = files.filter(file => file.endsWith('.mp4'));
      videoFiles.forEach(file => {
        fs.unlink(path.join(videoDirectory, file), err => {
          if (err) {
            console.error(`Error deleting file: ${file}`, err);
          }
        });
      });
  
      res.json({ message: 'All video files deleted successfully' });
    });
  });
  
  
const server = app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

const wss = new WebSocket.Server({ server });

let ws;
wss.on('connection', (socket) => {
  ws = socket;
  console.log('WebSocket connection established');
  socket.on('close', () => {
    console.log('WebSocket connection closed');
    ws = null;
  });
});
