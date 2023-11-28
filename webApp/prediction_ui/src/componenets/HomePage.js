// src/HomePage.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';


function HomePage() {
  const [text, setText] = useState('');
  const [title, setTitle] = useState('');
  const [notification, setNotification] = useState('');
  const [message, setMessage] = useState('');

  const handleTextChange = (event) => {
    const textarea = event.target;
    setText(textarea.value);
  };
  const [responseMessage, setResponseMessage] = useState('');
  const [wordCount, setWordCount] = useState('');
  const [titleWordCount, setTitleWordCount] = useState('');
  const handleSubmit = async () => {
    try {
      const response = await axios.post('http://localhost:8000/submit-text', { text, title });
      const { message, wordCount, wordCountTitle } = response.data;
      setResponseMessage(message);
      setWordCount(wordCount);
      setTitleWordCount(wordCountTitle);
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
      
      <div className="content">
        <div className="textbox-container">
          <textarea
            placeholder="Type your Title here..."
            value={title}
            onChange={(t) => {
              setTitle(t.target.value );
            }}
            className="other-cool-textbox"
            rows={3}
          />

          <textarea
            placeholder="Type your Abstract here..."
            value={text}
            onChange={handleTextChange}
            className="cool-textbox"
            rows={3}
          />
          <button onClick={handleSubmit} className="submit-button">Submit</button>
          {responseMessage && <div className="notification">Server: {responseMessage}</div> && wordCount !== null && <div>Title Word count: {titleWordCount}, Absrtact Word Count: {wordCount}</div>}
        </div>
        
      </div>

      <div className="Message">
      </div>
    </div>
  );
}

export default HomePage;
