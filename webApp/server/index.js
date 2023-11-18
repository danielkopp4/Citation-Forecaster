const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const port = 5000;

app.use(cors());
app.use(bodyParser.json());

app.post('/submit-text', (req, res) => {
  console.log('Received text:', req.body.text);
  res.json({ message: 'Received' });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});