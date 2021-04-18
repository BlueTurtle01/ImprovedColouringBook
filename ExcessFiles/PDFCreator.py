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


def book_creator(original_image):
    from reportlab.platypus import Image as imag
    from reportlab.platypus import PageBreak, NextPageTemplate, BaseDocTemplate, Frame, PageTemplate
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import cm

    elements = []
    # Create a canvas to instantiate the PDF file
    canvas = BaseDocTemplate((str(output_path) + 'ColouringBook.pdf'), pagesize=A4, rightMargin=25, leftMargin=25,
                             topMargin=25, bottomMargin=25)

    portrait_frame = Frame(canvas.leftMargin, canvas.bottomMargin, canvas.width, canvas.height, id='portrait_frame')
    landscape_frame = Frame(canvas.leftMargin, canvas.bottomMargin, canvas.height, canvas.width, id='landscape_frame')

    width = 21
    # Insert the original image
    elements.append(NextPageTemplate('landscape'))
    elements.append(get_image(str("InputImages/" + file_name + ".jpg"), width=width*cm))
    elements.append(PageBreak())

    # Insert the Compressed/clustered image
    elements.append(get_image(("OutputImages/" + str(file_name) + "/Compressed.jpg"), width=width*cm))
    elements.append(PageBreak())

    # Insert the palette of colours required to replicate the clustered output
    #elements.append(imag((str(output_path) + "Palette.png"), width=400, height=50))
    # Insert the outline image
    elements.append(get_image((str(output_path) + "Threshold Gray.jpg"), width=width*cm))
    elements.append(PageBreak())

    # Insert the palette of colours required to replicate the clustered output
    #elements.append(imag((str(output_path) + "Palette.png"), width=400, height=50))
    # Insert the blended image
    elements.append(get_image((str(output_path) + "/Blended.jpg"), width=width*cm))

    canvas.addPageTemplates([PageTemplate(id='portrait', frames=portrait_frame),
                             PageTemplate(id='landscape', frames=landscape_frame, pagesize=landscape(A4))])

    canvas.build(elements)


def get_image(path, width):
    from reportlab.platypus import Image
    from reportlab.lib import utils
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    aspect = ih / float(iw)
    return Image(path, width=width, height=(width * aspect))

def pdf_creator(original_image):
    from reportlab.platypus import Image as imag
    from reportlab.platypus import PageBreak, NextPageTemplate, BaseDocTemplate, Frame, PageTemplate
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import cm

    elements = []
    # Create a canvas to instantiate the PDF file
    canvas = BaseDocTemplate((str(output_path) + 'ColouringBook.pdf'), pagesize=A4, rightMargin=25, leftMargin=25,
                             topMargin=25, bottomMargin=25)

    portrait_frame = Frame(canvas.leftMargin, canvas.bottomMargin, canvas.width, canvas.height, id='portrait_frame')
    landscape_frame = Frame(canvas.leftMargin, canvas.bottomMargin, canvas.height, canvas.width, id='landscape_frame')

    width = 21
    # Insert the original image
    elements.append(NextPageTemplate('landscape'))
    elements.append(get_image(str("InputImages/" + file_name + ".jpg"), width=width*cm))
    elements.append(PageBreak())

    # Insert the Compressed/clustered image
    elements.append(get_image(("OutputImages/" + str(file_name) + "/Compressed.jpg"), width=width*cm))
    elements.append(PageBreak())

    # Insert the palette of colours required to replicate the clustered output
    #elements.append(imag((str(output_path) + "Palette.png"), width=400, height=50))
    # Insert the outline image
    elements.append(get_image((str(output_path) + "Threshold Gray.jpg"), width=width*cm))
    elements.append(PageBreak())

    # Insert the palette of colours required to replicate the clustered output
    #elements.append(imag((str(output_path) + "Palette.png"), width=400, height=50))
    # Insert the blended image
    elements.append(get_image((str(output_path) + "/Blended.jpg"), width=width*cm))

    canvas.addPageTemplates([PageTemplate(id='portrait', frames=portrait_frame),
                             PageTemplate(id='landscape', frames=landscape_frame, pagesize=landscape(A4))])

    canvas.build(elements)

#book_creator(original_image)
