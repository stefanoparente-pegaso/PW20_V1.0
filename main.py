from src import DatasetGenerator
import xml.etree.ElementTree as ET

tree = ET.parse('config.xml')
root = tree.getroot()
print()


def createds():
    DatasetGenerator.generateDataset(root.find('datasetPath').text, root.find('datasetBckPath').text)
