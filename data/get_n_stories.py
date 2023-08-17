import re
import os

num_stories = 1000
count = 0
training_filename = 'TinyStories-' + str(num_stories) + '.txt'

def transmute_stories(num_stories=num_stories, count=count):
    if os.path.exists(training_filename):
        os.remove(training_filename)
    with open('TinyStories-train.txt') as source, open(training_filename, 'a') as train_data:
        line_accumulator = ""
        for line in source:
            if num_stories <= count:
                return
            line = re.sub(r"\n", " ", line)
            if re.search("<|endoftext|>", line) is None:
                line_accumulator += line    
            else:
                line_accumulator = line_accumulator.strip()
                if count == num_stories - 1:
                    print(line_accumulator, file=train_data, end='')
                else:
                    print(line_accumulator, file=train_data, end='\n')
                line_accumulator = ""
                count += 1

transmute_stories(num_stories, count)

train_data_check = 0
with open(training_filename) as train_data:
    for line in train_data:
        train_data_check += 1
print("Stories written: ", train_data_check)       