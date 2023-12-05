// Import necessary modules
const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process'); // Allows you to spawn child processes
const cors = require("cors");

// Create an Express application
const app = express();
const port = 8000;

// Enable CORS (Cross-Origin Resource Sharing) to allow requests from different origins
app.use(cors());

// Use the body-parser middleware to parse incoming JSON requests
app.use(bodyParser.json());

// Define a route for handling GET requests to "/message"
app.get("/message", (req, res) => {
  res.json({ message: "Hello from server!" });
}); 

// Define a route for handling POST requests to "/submit-text"
app.post('/submit-text', (req, res) => {
  const { text, title } = req.body;

  console.log('Received text:', { text, title });

  // Spawn a child process to run a Python script named 'word_count.py'
  const pythonProcess = spawn('python3', ['word_count.py']);
  let scriptOutput = '';

  // Write the 'text' data to the stdin of the Python process
  pythonProcess.stdin.write(text);
  pythonProcess.stdin.end();

  // Capture and process the stdout of the Python process
  pythonProcess.stdout.on('data', (data) => {
    scriptOutput += data.toString();
  });

  // Handle the end event of the Python process
  pythonProcess.stdout.on('end', () => {
    try {
      // Parse the JSON output from the Python script
      const parsedOutput = JSON.parse(scriptOutput);
      res.json({ message: 'Received', wordCount: parsedOutput.word_count, wordCountTitle: (title.trim().split(" ").length )});
    } catch (error) {
      console.error('Error parsing Python script output:', error);

      // Send an error response with a 500 Internal Server Error status code
      if (!res.headersSent) {
        res.status(500).send('Internal Server Error');
      }
    }
  });

  // Handle the stderr of the Python process
  pythonProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);

    // Send an error response with a 500 Internal Server Error status code
    if (!res.headersSent) {
      res.status(500).send('Error in Python script');
    }
  });
});

// Start the Express server and listen on the specified port
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
