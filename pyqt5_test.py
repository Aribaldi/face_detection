import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from functools import partial
import io
from PIL import Image
from matplotlib import pyplot as plt
from main import template_matching, viola_jones, face_symmetry
from PIL.ImageQt import ImageQt


class PhotoLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Image \n\n')
        #self.pil_img = Image.new('RGB', (32,32), color='white')
        self.img_path = ''
        self.setStyleSheet('''
        QLabel {
            border-radius: 40px;
            border: 2px solid green;;
        }''')


    def setPixmap(self, *args, **kwargs):
        super().setPixmap(*args, **kwargs)
        self.setStyleSheet('''
        QLabel {
            border: none;
        }''')


    # def convert_to_pil(self, *args):
    #     img = QImage(*args)
    #     buffer = QBuffer()
    #     buffer.open(QBuffer.ReadWrite)
    #     img.save(buffer, "PNG")
    #     self.pil_img = Image.open(io.BytesIO(buffer.data()))


class Template(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.template_matchingTabUI(), "Template matching")
        self.tabs.addTab(self.VLTabUI(), "Viola-Jones")
        self.tabs.addTab(self.FSTabUI(), "Face symmetry axes")
        top_layout = QGridLayout(self)
        top_layout.addWidget(self.tabs)
        self.setWindowTitle('Face detection')
        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.tabs)
        self.setCentralWidget(self.scroll)

    def template_matchingTabUI(self):
        tm_tab = QWidget()
        original_photo = PhotoLabel()
        template = PhotoLabel()
        result = PhotoLabel()
        original_header = QLabel()
        original_header.setText('Choose input image')
        original_header.adjustSize()
        template_header = QLabel()
        template_header.setText('Choose template image')
        template_header.adjustSize()
        result_header = QLabel()
        result_header.setText('Result:')
        result_header.adjustSize()
        btn_original = QPushButton('Browse...')
        btn_original.setStyleSheet('''QPushButton {background-color: green;;}''')
        btn_original.setObjectName('orig')
        btn_original.clicked.connect(partial(self.open_image, btn_original, original_photo, template))
        btn_template = QPushButton('Browse...')
        btn_template.setStyleSheet('''QPushButton {background-color: green;;}''')
        btn_template.setObjectName('temp')
        btn_template.clicked.connect(partial(self.open_image, btn_template, original_photo, template))
        btn_result = QPushButton('Get result')
        btn_result.setStyleSheet('''QPushButton {background-color: green;;}''')
        btn_result.setObjectName('res')
        btn_result.clicked.connect(partial(self.get_tm_result, original_photo, result, template))

        tm_tab.layout_outer = QGridLayout()
        tm_tab.layout_original = QVBoxLayout()
        tm_tab.layout_template = QVBoxLayout()
        tm_tab.layout_result = QVBoxLayout()
        tm_tab.layout_original.addWidget(original_header, 0)
        tm_tab.layout_template.addWidget(template_header, 0)
        tm_tab.layout_original.addWidget(btn_original, 1)
        tm_tab.layout_template.addWidget(btn_template, 1)
        tm_tab.layout_original.addWidget(original_photo, 2)
        tm_tab.layout_template.addWidget(template, 2)
        tm_tab.layout_result.addWidget(result_header, 0)
        tm_tab.layout_result.addWidget(btn_result, 1)
        tm_tab.layout_result.addWidget(result, 2)
        tm_tab.layout_outer.addLayout(tm_tab.layout_original, 1, 0)
        tm_tab.layout_outer.addLayout(tm_tab.layout_template, 2, 0)
        tm_tab.layout_outer.addLayout(tm_tab.layout_result, 3, 0)
        tm_tab.setLayout(tm_tab.layout_outer)

        return tm_tab

    def VLTabUI(self):
        fs_tab = QWidget()
        original_photo = PhotoLabel()
        result = PhotoLabel()
        original_header = QLabel()
        original_header.setText('Choose input image')
        original_header.adjustSize()
        result_header = QLabel()
        result_header.setText('Result:')
        result_header.adjustSize()
        btn_original = QPushButton('Browse...')
        btn_original.setStyleSheet('''QPushButton {background-color: green;;}''')
        btn_original.setObjectName('orig')
        btn_original.clicked.connect(partial(self.open_image, btn_original, original_photo, result))
        btn_result = QPushButton('Get result')
        btn_result.setStyleSheet('''QPushButton {background-color: green;;}''')
        btn_result.setObjectName('res')
        btn_result.clicked.connect(partial(self.get_vl_result, original_photo, result))

        fs_tab.layout_outer = QGridLayout()
        fs_tab.layout_original = QVBoxLayout()
        fs_tab.layout_result = QVBoxLayout()
        fs_tab.layout_original.addWidget(original_header, 0)
        fs_tab.layout_original.addWidget(btn_original, 1)
        fs_tab.layout_original.addWidget(original_photo, 2)
        fs_tab.layout_result.addWidget(result_header, 0)
        fs_tab.layout_result.addWidget(btn_result, 1)
        fs_tab.layout_result.addWidget(result, 2)
        fs_tab.layout_outer.addLayout(fs_tab.layout_original, 1, 0)
        fs_tab.layout_outer.addLayout(fs_tab.layout_result, 3, 0)
        fs_tab.setLayout(fs_tab.layout_outer)
        return fs_tab

    def FSTabUI(self):
        fs_tab = QWidget()
        original_photo = PhotoLabel()
        result = PhotoLabel()
        original_header = QLabel()
        original_header.setText('Choose input image')
        original_header.adjustSize()
        result_header = QLabel()
        result_header.setText('Result:')
        result_header.adjustSize()
        btn_original = QPushButton('Browse...')
        btn_original.setStyleSheet('''QPushButton {background-color: green;;}''')
        btn_original.setObjectName('orig')
        btn_original.clicked.connect(partial(self.open_image, btn_original, original_photo, result))
        btn_result = QPushButton('Get result')
        btn_result.setStyleSheet('''QPushButton {background-color: green;;}''')
        btn_result.setObjectName('res')
        btn_result.clicked.connect(partial(self.get_symmetry, original_photo, result))

        fs_tab.layout_outer = QGridLayout()
        fs_tab.layout_original = QVBoxLayout()
        fs_tab.layout_result = QVBoxLayout()
        fs_tab.layout_original.addWidget(original_header, 0)
        fs_tab.layout_original.addWidget(btn_original, 1)
        fs_tab.layout_original.addWidget(original_photo, 2)
        fs_tab.layout_result.addWidget(result_header, 0)
        fs_tab.layout_result.addWidget(btn_result, 1)
        fs_tab.layout_result.addWidget(result, 2)
        fs_tab.layout_outer.addLayout(fs_tab.layout_original, 1, 0)
        fs_tab.layout_outer.addLayout(fs_tab.layout_result, 3, 0)
        fs_tab.setLayout(fs_tab.layout_outer)
        return fs_tab


    def open_image(self, btn, original_photo, template, filename=None):
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(self, 'Select Photo', QDir.currentPath(), 'Images (*.png *.jpg)')
            if not filename:
                return
            if btn.objectName() == 'orig':
                original_photo.setPixmap(QPixmap(filename))
                original_photo.img_path = filename
                #original_photo.convert_to_pil(QPixmap(filename))
            if btn.objectName() == 'temp':
                template.setPixmap(QPixmap(filename))
                template.img_path = filename
                #template.convert_to_pil(QPixmap(filename))

    def get_tm_result(self, original_photo, result, template=None):
        result_pil_image = template_matching(original_photo.img_path, template.img_path)
        result_qPixmap = QPixmap.fromImage(ImageQt(result_pil_image))
        result.setPixmap(result_qPixmap)

    def get_vl_result(self, original_photo, result):
        result_pil_image = viola_jones(original_photo.img_path)
        result_qPixmap = QPixmap.fromImage(ImageQt(result_pil_image))
        result.setPixmap(result_qPixmap)

    def get_symmetry(self, original_photo, result):
        result_pil_image = face_symmetry(original_photo.img_path)
        result_qPixmap = QPixmap.fromImage(ImageQt(result_pil_image))
        result.setPixmap(result_qPixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = Template()
    gui.show()
    sys.exit(app.exec_())