import csv
import random

# Step 1: Read the CSV file and store its contents in a list
input_file = "outputrain.csv"
output_file = "traindata.csv"

with open(input_file, 'r') as file:
    reader = csv.reader(file)
    csv_data = list(reader)

# Separate the header row from the data rows
header_row = csv_data[0]
data_rows = csv_data[1:]

# Step 2: Shuffle the data rows (excluding the header row)
random.shuffle(data_rows)

# Combine the header row with the shuffled data rows
shuffled_csv_data = [header_row] + data_rows

# Step 3: Write the shuffled list back to the CSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(shuffled_csv_data)

print("CSV rows shuffled (excluding header) and written to output.csv.")
