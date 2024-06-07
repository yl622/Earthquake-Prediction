import xml.etree.ElementTree as ET
from datetime import datetime
import glob
import os

earthquake_taiwan = list()

# change it to your own path
directory_path = r'C:\Users\ylee7\Downloads\E-A0073-002' 

target_file = os.path.join(directory_path, 'CWA-EQ-Catalog-*.xml')
xml_files = glob.glob(target_file)

namespace = {'cwa': 'urn:cwa:gov:tw:cwacommon:0.1'}

def parse_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
                
    for eq_info in root.findall('.//cwa:EarthquakeInfo', namespace):
        origin_time = eq_info.find('cwa:OriginTime', namespace).text

        date_time = datetime.strptime(origin_time, '%Y-%m-%dT%H:%M:%S%z')
        date = date_time.strftime('%m/%d/%Y')
        time = date_time.strftime('%H:%M:%S')


        longitude = float(eq_info.find('cwa:EpicenterLongitude', namespace).text)
        latitude = float(eq_info.find('cwa:EpicenterLatitude', namespace).text)
        depth = float(eq_info.find('cwa:FocalDepth', namespace).text)
        magnitude = float(eq_info.find('cwa:LocalMagnitude', namespace).text)
        
        earthquake_taiwan.append([date, time, latitude, longitude, depth, magnitude])

def taiwan_earthquake_data():
    """
    read .xml file like "CWA-EQ-Catalog-*.xml" in dataset
    """
    """
        Input:

            None

        Return:

            list: date           string in format 'mm/dd/yyyy'
                  time           string in format 'hh:mm:ss'
                  latitude       float
                  longitude      float
                  depth          float
                  magnitude      float

    """
    for xml_file in xml_files:
        parse_xml_file(xml_file)
    
    return earthquake_taiwan

if __name__ == "__main__":
    data = taiwan_earthquake_data()
    for line in data:
        print(line)
