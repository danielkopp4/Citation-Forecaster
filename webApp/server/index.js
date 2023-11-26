const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process'); 

const cors = require("cors");
const app = express();
const port = 8000;
app.use(cors());


app.use(bodyParser.json());

app.get("/message", (req, res) => {
  res.json({ message: "Hello from server!" });
}); 

app.post('/submit-text', (req, res) => {
  console.log('Received text:', req.body.text);

  const pythonProcess = spawn('python3', ['word_count.py']);
  let scriptOutput = '';

  pythonProcess.stdin.write(req.body.text);
  pythonProcess.stdin.end();

  pythonProcess.stdout.on('data', (data) => {
    scriptOutput += data.toString();
  });

  pythonProcess.stdout.on('end', () => {
    try {
      const parsedOutput = JSON.parse(scriptOutput);
      res.json({ message: 'Received', wordCount: parsedOutput.word_count });
    } catch (error) {
      console.error('Error parsing Python script output:', error);
      // Send an error response only if no response has been sent
      if (!res.headersSent) {
        res.status(500).send('Internal Server Error');
      }
    }
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
    // Send an error response only if no response has been sent
    if (!res.headersSent) {
      res.status(500).send('Error in Python script');
    }
  });
});


app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});