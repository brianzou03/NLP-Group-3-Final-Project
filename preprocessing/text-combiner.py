import os
import pandas as pd

# Define the directory containing the text files
directory = "congress"

# Initialize a list to hold file data
data = []

# Iterate through all files in the directory
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".txt"):  # Process only .txt files
        file_path = os.path.join(directory, filename)
        
        # Read the file content
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # Append the filename and content to the data list
        data.append({"filename": filename, "content": content})

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
output_csv = "articles.csv"
df.to_csv(output_csv, index=False)

print(f"Successfully combined files into {output_csv}")
