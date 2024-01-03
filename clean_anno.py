input_path = "/home/jovyan/sonnt373/data/ALPR_OCR/251123_train_plate_gen_250k/train.txt"
output_path = "/home/jovyan/sonnt373/data/ALPR_OCR/251123_train_plate_gen_250k/train_refactor.txt"  # Change this to your desired output file path

dict_false = ['I', 'J', 'O', 'Q', 'R', 'W']
list_length = []
with open(input_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Create a new file for writing lines with text_length < 64
with open(output_path, 'w', encoding='utf-8') as output_file:
    for line in lines:
        text = line.split('\t')[1]
        text_length = len(text.strip())  # Using strip to remove any leading/trailing whitespace
        # if text_length < 60:
        #     output_file.write(line)  # Write the line to the new file
        # else:
        #     list_length.append(text_length)
        check = True
        for char in dict_false:
            if char in text.strip():
                check = False
                break
        if check == False:
            print(text.strip())
            list_length.append(text_length)
        else:
            output_file.write(line)  # Write the line to the new file

print(f'=====> length = {len(list_length)}')