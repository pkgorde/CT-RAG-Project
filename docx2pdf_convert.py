# import docx
import docx2pdf
import sys

file_path = sys.argv[1]

# # Open a docx file
# doc = docx.Document(file_path)

# # # Save the document as a pdf
# doc.save(''.join(file_path.split('.')[:-1])+'.pdf')

docx2pdf.convert(file_path, ''.join(file_path.split('.')[:-1])+'.pdf')