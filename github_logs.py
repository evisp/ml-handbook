import requests
import pandas as pd
from datetime import datetime

# List of student usernames
students = [
    "arberzylyftariholberton", "Megi96", "erdi6sh", "astheone", "sindi16",
    "saralozi", "kloekam", "krisaldamihali", "pjepri", "frenk1j",
    "donaldgjegji1-maker", "Sara2-3", "egigj-dev"
]

repo_name = "holbertonschool-machine_learning"
folder_path = "math/linear_algebra"  # folder to filter commits

all_commits = []

for username in students:
    print(f"Fetching commits for {username}...")
    page = 1
    while True:
        url = f"https://api.github.com/repos/{username}/{repo_name}/commits"
        params = {
            'path': folder_path,  # filter by folder
            'per_page': 100,
            'page': page
        }
        response = requests.get(url, params=params)
        data = response.json()

        if not data or 'message' in data:  # no more commits or repo doesn't exist
            break

        for c in data:
            author = username  # since each repo belongs to the user
            commit_date = datetime.fromisoformat(c['commit']['author']['date'].replace('Z',''))
            all_commits.append({
                'author': author,
                'datetime': commit_date,
                'day': commit_date.strftime("%A"),
                'hour': commit_date.hour
            })
        page += 1

# Create pandas DataFrame
df = pd.DataFrame(all_commits)

# Save to CSV
df.to_csv("student_commits.csv", index=False)
print("Saved all commits to student_commits.csv")
