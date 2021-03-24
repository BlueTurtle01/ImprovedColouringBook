from PIL import Image
import matplotlib.pyplot as plt
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as imag
from reportlab.platypus import SimpleDocTemplate, PageBreak
from Clustering import file_path as original_image






def pdf_creator():
    colour_palette = "Name.png"

    elements = []
    styles = getSampleStyleSheet()

    # Create a canvas to instantiate the PDF file
    canvas = SimpleDocTemplate('ColouringBook.pdf')

    elements.append(imag(original_image, width=400, height= 500))
    elements.append(PageBreak())
    elements.append(imag("OutputImages/clustered.jpg", width=400, height = 500))
    elements.append(PageBreak())
    elements.append(imag(colour_palette, width=400, height=50))
    elements.append(imag("OutputImages/FinalI.jpg", width=400, height = 500))
    elements.append(PageBreak())

    canvas.build(elements)

pdf_creator()