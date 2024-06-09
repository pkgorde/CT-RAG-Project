def convert_newlines_in_file(file_path):
    # Read the original content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Replace the '\n' string with an actual newline character
    modified_content = content.replace('\\n', '\n')

    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)

# Replace 'example.txt' with your file name
convert_newlines_in_file('llama2_response.txt')
