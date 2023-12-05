// src/HomePage.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';

/**
 * HomePage component for submitting text and title to a backend server,
 * and displaying the response including word counts.
 */
function HomePage() {
  // State hooks for managing component state
  const [text, setText] = useState(''); // State for the abstract text
  const [title, setTitle] = useState(''); // State for the title text
  const [responseMessage, setResponseMessage] = useState(''); // State for the server response message
  const [wordCount, setWordCount] = useState(''); // State for the word count of the abstract
  const [titleWordCount, setTitleWordCount] = useState(''); // State for the word count of the title

  // Handler for abstract text area change
  const handleTextChange = (event) => {
    setText(event.target.value);
  };

  // Handler for form submission
  const handleSubmit = async () => {
    try {
      // Post request to backend with text and title
      const response = await axios.post('http://localhost:8000/submit-text', { text, title });
      // Destructuring response data
      const { message, wordCount, wordCountTitle } = response.data;
      // Updating states with response data
      setResponseMessage(message);
      setWordCount(wordCount);
      setTitleWordCount(wordCountTitle);
    } catch (error) {
      // Handling and logging submission errors
      console.error('Error submitting text:', error);
      setResponseMessage('Error submitting text');
    }
  };

  // useEffect hook to fetch initial message from server on component mount
  useEffect(() => {
    fetch("http://localhost:8000/message")
      .then((res) => res.json())
      .then((data) => setResponseMessage(data.message));
  }, []);

  // Render method returning JSX
  return (
    <div>
      <div className="content">
        <div className="textbox-container">
          {/* Title input area */}
          <textarea
            placeholder="Type your Title here..."
            value={title}
            onChange={(t) => setTitle(t.target.value)}
            className="other-cool-textbox"
            rows={3}
          />

          {/* Abstract input area */}
          <textarea
            placeholder="Type your Abstract here..."
            value={text}
            onChange={handleTextChange}
            className="cool-textbox"
            rows={3}
          />

          {/* Submit button */}
          <button onClick={handleSubmit} className="submit-button">Submit</button>

          {/* Displaying server response and word counts */}
          {responseMessage && (
            <div className="notification">
              Server: {responseMessage}
              {wordCount !== null && (
                <div>Title Word count: {titleWordCount}, Abstract Word Count: {wordCount}</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Exporting HomePage component
export default HomePage;
