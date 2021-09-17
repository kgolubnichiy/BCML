import sys
import csv

input = list(csv.reader(open("test.csv", "r")))
output = open('test_filtered.csv', 'w', newline='')
writer = csv.writer(output, delimiter=',')
writer.writerow(input[0])

for i in range(1, len(input)):
  option = input[i][0]
  for j in range(i+1, len(input)):
    if option == input[j][0]:
      writer.writerow(input[i])
      break
      

