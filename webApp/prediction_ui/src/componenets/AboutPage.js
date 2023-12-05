import React from 'react';
import './AboutPage.css';

/**
 * AboutPage component that provides information about the project.
 * It includes details on the project's purpose, dataset sources, and instructions on how to download the project code.
 */
function AboutPage() {
  return (
    <div>
      <div className="about-container">
        {/* Heading for the about page */}
        <h1>About Our Project</h1>

        {/* Brief description of the project */}
        <p>"ReadMe"</p>

        {/* Detailed information about the project */}
        <ul>
          This project analyzes publications to predict the number of citations of a publication.
          This work introduces a dataset that contains the publications from the 
          <a href="https://www.kaggle.com/datasets/Cornell-University/arxiv">arxiv dataset</a> and combines it with the 
          citation count from the <a href="https://www.crossref.org/services/cited-by/">cross ref API</a>. 
          The download script merges these two sources to get a novel dataset that can be used to determine relationships in academic citations. 

          The citation count has already been downloaded and included in the data folder as it requires multiple days to download the full count for all 1.7 million publications. 
          The download was run April 2023. To update the citation count simply delete the `complete.data` file and see the downloading section to see how to redownload the data.

          {/* Instructions for downloading the project code */}
          <p>To download the code run `git clone git@github.com:danielkopp4/citation_prediction.git` 
          or `git clone https://github.com/danielkopp4/citation_prediction.git` for HTTP.</p>
        </ul>
      </div>
    </div>
  );
}

// Exporting AboutPage component
export default AboutPage;
