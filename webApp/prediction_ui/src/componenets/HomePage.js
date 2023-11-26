// src/HomePage.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';


function HomePage() {
  const [text, setText] = useState('');
  const [notification, setNotification] = useState('');
  const [message, setMessage] = useState('');

  const handleTextChange = (event) => {
    const textarea = event.target;
    setText(textarea.value);
  };
  const [responseMessage, setResponseMessage] = useState('');
  const [wordCount, setWordCount] = useState('');
  const handleSubmit = async () => {
    try {
      const response = await axios.post('http://localhost:8000/submit-text', { text });
      setResponseMessage(response.data.message);
      setWordCount(response.data.wordCount);
    } catch (error) {
      console.error('Error submitting text:', error);
      setResponseMessage('Error submitting text');
    }
  };

  useEffect(() => {
    fetch("http://localhost:8000/message")
      .then((res) => res.json())
      .then((data) => setMessage(data.message));
  }, []);

  return (
    <div>
      <div className="navbar">
        <button onClick={() => console.log("Button 1 clicked!")}>HomePage</button>
        <button onClick={() => console.log("Button 2 clicked!")}>History</button>
        <button onClick={() => console.log("Button 3 clicked!")}>About</button>
        <button onClick={() => console.log("Button 4 clicked!")}>Credits</button>
      </div>
      <div className="content">
        Welcome to the homepage!
        <div className="textbox-container">
          <textarea
            placeholder="Type your paragraph here..."
            value={text}
            onChange={handleTextChange}
            className="cool-textbox"
            rows={3}
          />
          <button onClick={handleSubmit} className="submit-button">Submit</button>
          {responseMessage && <div className="notification">Server: {responseMessage}</div> && wordCount !== null && <div>Word Count: {wordCount}</div>}
        </div>
      </div>

      <div className="Message">
        <h1>{message}</h1>
      </div>
    </div>
  );
}

export default HomePage;
