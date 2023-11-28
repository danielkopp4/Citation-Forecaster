import logo from './logo.svg';
import './App.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './componenets/HomePage';
import AboutPage from './componenets/AboutPage';
import TimelinePage from './componenets/TimelinePage';
import NavBar from './componenets/NavBar';

function App() {
  return (
    <Router>
      
      <NavBar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/timeline" element={<TimelinePage />} />
      </Routes>
    </Router>
  );
}

export default App;
