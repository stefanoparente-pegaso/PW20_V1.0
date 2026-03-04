import os.path
import pathlib

from src import DatasetGenerator
import xml.etree.ElementTree as ET

tree = ET.parse('config.xml')
xmlRoot = tree.getroot()

def initConfig():
    rootPath = pathlib.Path(__file__).parent.resolve()
    xmlRoot.find("datasetPath").text = str(rootPath / "data" / "dataset.csv")
    xmlRoot.find("datasetBckPath").text = str(rootPath / "data" / "bck")

def createds():
    initConfig()
    a=xmlRoot.find("datasetPath").text
    B=xmlRoot.find("datasetBckPath").text
    DatasetGenerator.generateDataset(xmlRoot.find("datasetPath").text ,xmlRoot.find("datasetBckPath").text)

createds()