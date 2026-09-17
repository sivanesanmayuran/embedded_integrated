import os
import re
import requests

# Your markdown content
md_content = """
The following is a list of free and/or open source books on machine learning, statistics, data mining, etc. 

## Machine Learning / Data Mining 
* [Distributed Machine Learning Patterns](https://github.com/terrytangyuan/distributed-ml-patterns) - Book (free to read online) + Code 
* [The Hundred-Page Machine Learning Book](http://themlbook.com/wiki/doku.php) 
* [Real World Machine Learning](https://www.manning.com/books/real-world-machine-learning) [Free Chapters] 
* [An Introduction To Statistical Learning With Applications In R](https://drive.usercontent.google.com/download?id=106d-rN7cXpyAkgrUqjcPONNCyO-rX7MQ&export=download) - Book + R Code 
* [An Introduction To Statistical Learning With Applications In Python](https://drive.usercontent.google.com/download?id=1ajFkHO6zjrdGNqhqW1jKBZdiNGh_8YQ1&export=download) - Book + Python Code 
* [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) - Book 
* [Computer Age Statistical Inference (CASI)](https://web.stanford.edu/~hastie/CASI_files/PDF/casi.pdf) ([Permalink as of October 2017](https://perma.cc/J8JG-ZVFW)) - Book 
* [Probabilistic Programming & Bayesian Methods for Hackers](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/) - Book + IPython Notebooks 
* [Think Bayes](https://greenteapress.com/wp/think-bayes/) - Book + Python Code 
* [Information Theory, Inference, and Learning Algorithms](http://www.inference.phy.cam.ac.uk/mackay/itila/book.html) 
* [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/) 
* [Data Intensive Text Processing w/ MapReduce](https://lintool.github.io/MapReduceAlgorithms/) 
* [Reinforcement Learning: - An Introduction](http://incompleteideas.net/book/the-book-2nd.html) ([Permalink to Nov 2017 Draft](https://perma.cc/83ER-64M3)) 
* [Mining Massive Datasets](http://infolab.stanford.edu/~ullman/mmds/book.pdf)
"""

# Create a folder to save the PDFs
output_folder = "downloaded_books"
os.makedirs(output_folder, exist_ok=True)

# Regex pattern to find markdown links: [Link Text](URL)
link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# Extract all links
matches = link_pattern.findall(md_content)

print("Starting downloads...")

for title, url in matches:
    # Filter for URLs that point to PDFs or direct Google Drive download files
    if url.endswith(".pdf") or "export=download" in url:
        # Clean title to create a safe filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).strip()
        filename = f"{safe_title}.pdf"
        filepath = os.path.join(output_folder, filename)
        
        print(f"Downloading: {title}...")
        try:
            # Set a user-agent header to avoid being blocked by servers
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Save the file in chunks to handle large documents efficiently
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f" Saved as: {filepath}")
            
        except Exception as e:
            print(f" Failed to download {title}. Error: {e}")

print("\nProcess finished!")
