# Practice with Python: Functions, Loops, and Data Structures

## Why This Matters

Real-world machine learning (and software work) almost always involves reading, processing, and summarizing structured data. Becoming comfortable manipulating lists and dictionaries in Python will enable you to prepare datasets, analyze results, and build robust ML code.

For this hands-on practice, you'll step through a mini-project working with a common data format (JSON), using real-world-style user data. You'll build small functions and use loops to transform, filter, and analyze the data, exactly the skillset needed for both ML work and technical interviews.

![Data Structures Illustration](https://i.imgur.com/pfNbPkj.png)

You'll use a set of sample user data provided by [JSONPlaceholder](https://jsonplaceholder.typicode.com/users), which mimics the structure of typical web APIs. Each user includes fields like name, email, address, company, etc.

You have two options for how to get this data:

- **Option 1:** Copy the result from [https://jsonplaceholder.typicode.com/users](https://jsonplaceholder.typicode.com/users) directly into your Python file as a list, assigning it to a variable named, say `data`.
- **Option 2 (If curious):** Use the `requests` module to fetch the data at runtime.

Example setup (with `requests`):
```python
import requests

data = requests.get("https://jsonplaceholder.typicode.com/users").json()

# If you get an error, run this in the terminal first:
# pip install requests
```

## Step-by-Step Practice Tasks

Let’s start simple and build up to more complex processing. **Create a Python script called `user_processing.py`. Each task can be a separate function in that file.**

### Task 1: Print All User Names
Write a function that loops through the `data` and prints the `name` of each user.

### Task 2: Collect Emails with a Specific Domain
Write a function that returns a list of all users' emails ending with `.biz`.

### Task 3: Find Users in a Given City
Write a function that takes a city name as an argument and prints the names and emails of all users living in that city (`data[i]["address"]["city"]`).

### Task 4: Count Companies
Write a function that builds and prints a dictionary: keys are company names (from `data[i]["company"]["name"]`), values are counts of how many users work at each company.

### Task 5: List Unique Zipcodes
Write a function that extracts all unique zip codes from the address section of each user, and prints the sorted list of zip codes.

### Task 6: Summarize Users by Username Initial
Write a function that creates a dictionary mapping first letters of usernames (from `data[i]["username"]`) to a list of names of users starting with that letter (e.g., `'B': ['Bret', 'Brandon']`).

### Task 7: Nicely Print a User's Full Address
Write a function that takes a username and prints their full address in this format:
`"Leanne Graham: Apt. 556, Kulas Light, Gwenborough, 92998-3874"`

## Submission Guidelines
- **Organize your code:** Place each task in its own function, and call the functions from a main block (`if __name__ == "__main__":`).
- **Comment your code:** Briefly explain what each function does.
- **Test each function:** Call them with sample values to demonstrate the output.

## Summary & Next Steps

**Completing these tasks will help you:**
- Practice for-loops, list/dictionary processing, and function writing
- Work with real-world nested data
- Get ready for ML work, interviews, and project tasks

Next, you'll expand this skillset to analyze larger datasets and write code for typical ML preprocessing steps.
