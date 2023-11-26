const express = require('express');
const bodyParser = require('body-parser');

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
  res.json({ message: 'Received' });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});