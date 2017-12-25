import sys
from PyQt4 import QtGui
from PyQt4.QtGui import QFileDialog
import CarPlateOCR as cpo

#ocr = cpo.OCR('licence_plate')

def remove_extension(path):
	return ('.').join(str(path).split('.')[:-1])

def get_file_name(path):
	return (str(path)).split('/')[-1]

def selectAndRenderFile():
    gui.fileName = QFileDialog.getOpenFileName()
    print(gui.fileName)
    gui.pic.setPixmap(QtGui.QPixmap(gui.fileName))
    gui.pic.move(10, 10)
    gui.pic.resize(1000, 250)
    gui.pic.show()
    gui.ocr = cpo.OCR(get_file_name(remove_extension(gui.fileName)))

def calculate():
	chars = gui.ocr.process_image(str(gui.fileName))
	new = remove_extension(get_file_name(gui.fileName)) + '_out.png'
	print(new)
	gui.setPic(new)
	gui.setLabelText(str(chars))

class GUI(QtGui.QWidget):

    def __init__(self):
        super(GUI, self).__init__()
        
        self.title = 'Car plate character recognition'
        self.filename = ''
        self.ocr = ''
        self.initUI()


    def setPic(self, file):
        self.pic.setPixmap(QtGui.QPixmap(file))


    def setLabelText(self, text):
        self.label.setText('Licence plate text is: ' + text)
        self.label.adjustSize()
    
        
    def initUI(self):
        self.btn_calc = QtGui.QPushButton('Calculate', self)
        self.btn_calc.resize(self.btn_calc.sizeHint())
        self.btn_calc.move(1100, 50)  
        self.btn_calc.clicked.connect(calculate)

        self.btn_file = QtGui.QPushButton('Select image', self)
        self.btn_file.resize(self.btn_file.sizeHint())
        self.btn_file.move(1100, 20)
        self.btn_file.clicked.connect(selectAndRenderFile)

        self.label = QtGui.QLabel(self)
        self.label.move(1025, 100)

        self.pic = QtGui.QLabel(self)

        self.setGeometry(100, 300, 1200, 300)
        self.setWindowTitle(self.title)    
    
        self.show()




def execute():      
    app = QtGui.QApplication(sys.argv)
    global gui
    gui = GUI()
    sys.exit(app.exec_())