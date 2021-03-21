import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from functools import partial
import io
from PIL import Image
from matplotlib import pyplot as plt
from main import template_matching, viola_jones
from PIL.ImageQt import ImageQt


class PhotoLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Image \n\n')
        self.pil_img = Image.new('RGB', (32,32), color='white')
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


    def convert_to_pil(self, *args):
        img = QImage(*args)
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        img.save(buffer, "PNG")
        self.pil_img = Image.open(io.BytesIO(buffer.data()))


class Template(QWidget):
    def __init__(self):
        super().__init__()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.template_matchingTabUI(), "Template matching")
        self.tabs.addTab(self.VLTabUI(), "Viola-Jones")
        top_layout = QGridLayout(self)
        top_layout.addWidget(self.tabs)
        self.setWindowTitle('Face detection')

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
        btn_original.setObjectName('orig')
        btn_original.clicked.connect(partial(self.open_image, btn_original, original_photo, template))
        btn_template = QPushButton('Browse...')
        btn_template.setObjectName('temp')
        btn_template.clicked.connect(partial(self.open_image, btn_template, original_photo, template))
        btn_result = QPushButton('Get result')
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
        vl_tab = QWidget()
        original_photo = PhotoLabel()
        result = PhotoLabel()
        original_header = QLabel()
        original_header.setText('Choose input image')
        original_header.adjustSize()
        result_header = QLabel()
        result_header.setText('Result:')
        result_header.adjustSize()
        btn_original = QPushButton('Browse...')
        btn_original.setObjectName('orig')
        btn_original.clicked.connect(partial(self.open_image, btn_original, original_photo, result))
        btn_result = QPushButton('Get result')
        btn_result.setObjectName('res')
        btn_result.clicked.connect(partial(self.get_vl_result, original_photo, result))

        vl_tab.layout_outer = QGridLayout()
        vl_tab.layout_original = QVBoxLayout()
        vl_tab.layout_result = QVBoxLayout()
        vl_tab.layout_original.addWidget(original_header, 0)
        vl_tab.layout_original.addWidget(btn_original, 1)
        vl_tab.layout_original.addWidget(original_photo, 2)
        vl_tab.layout_result.addWidget(result_header, 0)
        vl_tab.layout_result.addWidget(btn_result, 1)
        vl_tab.layout_result.addWidget(result, 2)
        vl_tab.layout_outer.addLayout(vl_tab.layout_original, 1, 0)
        vl_tab.layout_outer.addLayout(vl_tab.layout_result, 3, 0)
        vl_tab.setLayout(vl_tab.layout_outer)
        return vl_tab

    def open_image(self, btn, original_photo, template, filename=None):
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(self, 'Select Photo', QDir.currentPath(), 'Images (*.png *.jpg)')
            if not filename:
                return
            if btn.objectName() == 'orig':
                original_photo.setPixmap(QPixmap(filename))
                original_photo.convert_to_pil(QPixmap(filename))
            if btn.objectName() == 'temp':
                template.setPixmap(QPixmap(filename))
                template.convert_to_pil(QPixmap(filename))

    def get_tm_result(self, original_photo, result, template=None):
        result_pil_image = Image.fromarray(template_matching(original_photo.pil_img, template.pil_img))
        result_qPixmap = QPixmap.fromImage(ImageQt(result_pil_image))
        result.setPixmap(result_qPixmap)

    def get_vl_result(self, original_photo, result):
        result_pil_image = Image.fromarray(viola_jones(original_photo.pil_img))
        result_qPixmap = QPixmap.fromImage(ImageQt(result_pil_image))
        result.setPixmap(result_qPixmap)






if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = Template()
    gui.show()
    sys.exit(app.exec_())