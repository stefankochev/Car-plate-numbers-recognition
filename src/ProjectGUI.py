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
    gui.ocr = cpo.OCR(get_file_name(remove_extension(gui.fileName)), gui)

def displayNext():
    try:
        gui.setPic(gui.q[0])
        gui.q = gui.q[1:]
    except:
        pass

def calculate():
    gui.q = []
    chars = gui.ocr.process_image(str(gui.fileName))

    if(not gui.check_box.checkState()):
        new = 'output/' + remove_extension(get_file_name(gui.fileName)) + '_out.png'
        gui.setPic(new)
        gui.setLabelText(str(chars))

class GUI(QtGui.QWidget):

    def __init__(self):
        super(GUI, self).__init__()
        
        self.title = 'Car plate character recognition'
        self.filename = ''
        self.ocr = ''
        self.q = []
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

        self.btn_next = QtGui.QPushButton('Next', self)
        self.btn_next.resize(self.btn_next.sizeHint())
        self.btn_next.move(1025, 160)
        self.btn_next.clicked.connect(displayNext)

        self.check_box = QtGui.QCheckBox('Step by step', self)
        self.check_box.move(1025, 140)

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