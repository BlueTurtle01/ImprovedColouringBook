from matplotlib.pyplot import imsave
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
from flask import Flask, render_template, send_file
from flask_wtf import FlaskForm
from wtforms import FileField
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from flask_uploads import configure_uploads, IMAGES, UploadSet
import os
import glob
import io


# Instantiate the Flask class and assign it to the variable "app"
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = '/static/img/'
app.config['SECRET_KEY'] = 'thisisasecret'
app.config['UPLOADS_DEFAULT_DEST'] = '/static/img/'
images = UploadSet('images', IMAGES)
configure_uploads(app, images)


class Canny:
    def __init__(self, name, user_image, clusters, scalar):
        self.file_name = name
        self.imag = user_image
        self.gray = cv2.cvtColor(np.array(self.imag), cv2.COLOR_BGR2GRAY)
        thresh, self.bw = cv2.threshold(self.gray, 20, 255, cv2.THRESH_BINARY)
        self.clusters = clusters
        self.height, self.width = self.imag.size
        self.scalar = scalar

    def calculate_filter_size(self):
        # We calculate the number of unique colours to determine the kernel_size. An image with a lot of unique
        # colours requires more blurring, which is achieved with a bigger kernel_size.
        unique_colors = set()
        for i in range(self.imag.size[0]):
            for j in range(self.imag.size[1]):
                pixel = self.imag.getpixel((i, j))
                unique_colors.add(pixel)

        filter_size = int(str(round(len(unique_colors), -3))[0])

        # Filter size needs to be odd. Sometimes I forget and put an even filter size. This will catch this error and
        # reduce it by one to make it odd again.
        if filter_size % 2 == 0:
            filter_size = max(5, filter_size + 1)

        else:
            filter_size = max(5, filter_size)

        return filter_size

    def k_means(self):
        # Resize the image in the hopes that kmeans and contours can find the edges easier.
        w, h = self.imag.size
        if w > 1000:
            h = int(h * 1000. / w)
            w = 1000
        imag = self.imag.resize((w, h), Image.NEAREST)

        # Dimension of the original image
        cols, rows = imag.size

        # Flatten the image with the new dimensions.
        imag = np.array(imag).reshape(rows * cols, 3)

        # Implement k-means clustering to form k clusters
        kmeans = MiniBatchKMeans(n_clusters=self.clusters)
        kmeans.fit(imag)

        # Replace each pixel value with its nearby centroid
        compressed_image = kmeans.cluster_centers_[kmeans.labels_]
        compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

        # Reshape the image to original dimension
        self.compressed_image = compressed_image.reshape(rows, cols, 3)


    def median(self):
        self.compressed_image = cv2.resize(self.compressed_image,
                                           dsize=(self.height * self.scalar, self.width * self.scalar),
                                           interpolation=cv2.INTER_NEAREST)

        filter_size = self.calculate_filter_size()
        self.median = cv2.medianBlur(self.compressed_image, filter_size)

        pil_compressed = Image.fromarray(self.median)
        pil_compressed.save(f'{directory_path}static/img/images/{self.file_name}Guide.pdf', "PDF", resolution=100.0)

    def auto_canny(self, sigma=0.33):
        self.canny = cv2.cvtColor(self.median, cv2.COLOR_RGB2GRAY)

        # compute the median of the single channel pixel intensities
        v = np.median(self.canny)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (0.3 - sigma) * v))
        upper = int(min(255, (0.8 - sigma) * v))

        self.edged = cv2.Canny(self.canny, lower, upper)
        edged = cv2.dilate(self.edged, (20, 20), iterations=1)
        self.edged = cv2.erode(edged, (20, 20), iterations=1)

    def draw_contours(self):
        """
        Draw the contours over a blank array. The function cv2.DrawContours overlays the contours on top of the bitwise array.
        Which is not ideal if the bitwise array contains some small, noisy contours. Therefore, I created an empty array first and then used this as the base
        for drawing the contours onto.
        :param edged: Output of the Canny Edge Detection algorithm after applying erode and dilation.
        :return:
        """
        contours, hierarchy = cv2.findContours(self.edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

        long_contours = []
        for contour in contours:
            if contour.shape[0] > 20:
                long_contours.append(contour)

        #edged = cv2.bitwise_not(self.edged)
        #rgb = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
        #temp_array = np.ones([rgb.shape[0], rgb.shape[1], rgb.shape[2]])

        self.bw = np.repeat(self.bw[:, :, np.newaxis], 3, axis=2)
        contours_ = cv2.drawContours(self.bw, long_contours, -1, (0, 0, 0), thickness=1)
        imsave(f'static/img/{self.file_name}', contours_, cmap='gray')

    def create_svg(self, contours):
        # For some reason height and width got swapped. As a dirty hack I will just reassign them and reassess if it becomes a bigger problem.
        width = self.height
        height = self.width

        self.bw = np.repeat(self.bw[:, :, np.newaxis], 3, axis=2)

        with open(f'{directory_path}static/img/images/{self.file_name}.svg', "w+") as f:
            f.write(f'<svg width="{width}px" height="{height}px" xmlns="http://www.w3.org/2000/svg">')

            for c in contours:
                f.write('<path d="M')
                for i in range(len(c)):
                    x, y = c[i][0]
                    f.write(f"{x} {y} ")
                f.write('" style="stroke:black;fill:none"/>')
            f.write("</svg>")

    @staticmethod
    def calculate_long_contours(contours):
        long_contours = []
        for contour in contours:
            if contour.shape[0] > 10:
                long_contours.append(contour)

        return long_contours

    def create_pdf(self):
        drawing = svg2rlg(f'{directory_path}static/img/images/{self.file_name}.svg')
        renderPDF.drawToFile(drawing, f'{directory_path}static/img/images/{self.file_name}.pdf')

        from zipfile import ZipFile
        # Create a ZipFile Object
        with ZipFile(f'{directory_path}static/img/images/{self.file_name}.zip', 'w') as zipObj2:
            # Add multiple files to the zip
            zipObj2.write(f'{directory_path}static/img/images/{self.file_name}.pdf')
            zipObj2.write(f'{directory_path}static/img/images/{self.file_name}Guide.pdf')


    def collate_svg(self):
        contours, hierarchy = cv2.findContours(self.edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        contours = self.calculate_long_contours(contours)

        self.create_svg(contours)
        self.create_pdf()

    def clean_up(self):
        for file in glob.glob(f'{directory_path}static/img/images/{self.file_name}'):
            os.remove(file)



class MyForm(FlaskForm):
    image = FileField('image')


### Config Settings ###
clusters = 11
scale_factor = 1

# directory_path is need as the path required for my local vs the PythonAnywhere server is different.
directory_path = f'/'  # end with a slash


@app.route('/', methods=['GET', 'POST'])
def index():
    form = MyForm()

    if form.validate_on_submit():
        filename = images.save(form.image.data)
        if filename.endswith(".png"):

            image = Image.open(f'{directory_path}static/img/images/{filename}')

            # If the file is a PNG we need to remove the extension to resave it as a jpg
            filename_png, extension = filename.split(".")
            image.convert('RGB').save(f'{directory_path}static/img/images/{filename_png}.jpg', "JPEG")

            # Open the new jpg
            image = Image.open(f'{directory_path}static/img/images/{filename_png}.jpg')

            # We send the filename to the Class for it to overwrite the original image with the output to save server space.
            # "filename" includes the .png extension though that we don't want. Therefore, we need to add the .jpg extension to our now shortened filename_png
            filename = f'{filename_png}'
            pic = Canny(user_image=image, name=filename, clusters=clusters, scalar=scale_factor)
            pic.k_means()
            pic.median()
            pic.auto_canny(sigma=0.33)
            pic.collate_svg()
            #pic.clean_up()
            svg_path = filename.split(".")[0]

            #return send_file(f'{directory_path}static/img/images/{svg_path}Guide.pdf', as_attachment=True)
            return send_file(f'{directory_path}static/img/images/{svg_path}.zip', as_attachment=True)

        else:
            filename_split = filename.split(".")
            extension = filename.split(".")[-1]
            filename = ".".join(filename_split[0:-1])
            if extension == "jpg":
                image = Image.open(f'{directory_path}static/img/images/{filename}.jpg')
            else:
                image = Image.open(f'{directory_path}static/img/images/{filename}.jpeg')
            pic = Canny(user_image=image, name=filename, clusters=clusters, scalar=scale_factor)
            pic.k_means()
            pic.median()
            pic.auto_canny(sigma=0.33)
            pic.collate_svg()
            #pic.clean_up()

            #return send_file(f'{directory_path}static/img/images/{filename}Guide.pdf', as_attachment=True)
            return send_file(f'{directory_path}static/img/images/{filename}.zip', as_attachment=True)

    return render_template('upload2.html', form=form)


@app.route('/privacy')
def privacy_policy():
    return render_template('privacy.html')



### Blog ###
import sys
from flask import Flask, render_template, url_for
from flask_flatpages import FlatPages
from flask_frozen import Freezer

DEBUG = True
FLATPAGES_AUTO_RELOAD = DEBUG
FLATPAGES_EXTENSION = '.md'

app.config.from_object(__name__)
pages = FlatPages(app)
freezer = Freezer(app)


@app.route('/blog')
def blog():
    return render_template('bloghome.html', pages=pages)


@app.route('/<path:path>.html')
def page(path):
    print("View function activated!")
    page = pages.get_or_404(path)
    return render_template('page.html', page=page)


@freezer.register_generator
def pagelist():
    for page in pages:
        yield url_for('page', path=page.path)



if __name__ == '__main__':
    app.run()
