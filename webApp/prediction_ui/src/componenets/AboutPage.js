import React from 'react';
import './AboutPage.css'; // Assuming you have a separate CSS file for styling

function AboutPage() {
  return (
    <div className="about-container">
      <h1>About Our Project</h1>
      <p>This project is a web application designed to ... [add your project description here]</p>
      <p>Key Features:</p>
      <ul>
      This project analyzes publications to predict the number of citations of a publication. This work introduces a dataset that contains the publications from the [arxiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) and combines it with the citation count from the [cross ref API](https://www.crossref.org/services/cited-by/). The download script merges these two sources to get a novel dataset that can be used to determine relationships in acedemic citations. 

The citation count has alreay been downloaded and included in the data folder as it requires muliple days to download the full count for all 1.7 million publications. The download was run April 2023. To update the citation count simple delete the `complete.data` file and see the downloading section to see how to redownload the data.

To download the code run `git clone git@github.com:danielkopp4/citation_prediction.git` or `git clone https://github.com/danielkopp4/citation_prediction.git` for HTTP.
      </ul>
    </div>
  );
}

export default AboutPage;