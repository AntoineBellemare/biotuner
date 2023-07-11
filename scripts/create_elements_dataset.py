'''This script creates a dataset of elements and their emission spectra from the NIST website.
Note that the wavelengths are expressed in Angstrom.
'''

from bs4 import BeautifulSoup
import requests
import pandas as pd
import re

NIST_elements_URL = "https://physics.nist.gov/PhysRefData/Handbook/element_name_a.htm"
table_template = "https://physics.nist.gov/PhysRefData/Handbook/Tables/{}table2.htm"
response = requests.get(NIST_elements_URL)

soup = BeautifulSoup(response.content, 'html.parser')
elements = soup.find_all('a')

# From GPT
elements_types = {
    'Alkali Metals': ['Lithium', 'Sodium', 'Potassium', 'Rubidium', 'Cesium', 'Francium'],
    'Alkaline Earth Metals': ['Beryllium', 'Magnesium', 'Calcium', 'Strontium', 'Barium', 'Radium'],
    'Transition Metals': ['Scandium', 'Titanium', 'Vanadium', 'Chromium', 'Manganese', 'Iron', 'Cobalt',
                          'Nickel', 'Copper', 'Zinc', 'Yttrium', 'Zirconium', 'Niobium', 'Molybdenum',
                          'Technetium', 'Ruthenium', 'Rhodium', 'Palladium', 'Silver', 'Cadmium',
                          'Hafnium', 'Tantalum', 'Tungsten', 'Rhenium', 'Osmium', 'Iridium', 'Platinum',
                          'Gold', 'Mercury', 'Rutherfordium', 'Dubnium', 'Seaborgium', 'Bohrium',
                          'Hassium', 'Meitnerium', 'Darmstadtium', 'Roentgenium', 'Copernicium',
                          'Nihonium', 'Flerovium', 'Moscovium', 'Livermorium', 'Tennessine', 'Oganesson'],
    'Post-transition Metals': ['Aluminum', 'Gallium', 'Indium', 'Tin', 'Thallium', 'Lead', 'Bismuth',
                               'Polonium', 'Livermorium'],
    'Metalloids': ['Boron', 'Silicon', 'Germanium', 'Arsenic', 'Antimony', 'Tellurium', 'Polonium'],
    'Nonmetals': ['Hydrogen', 'Carbon', 'Nitrogen', 'Oxygen', 'Phosphorus', 'Sulfur', 'Selenium'],
    'Noble Gases': ['Helium', 'Neon', 'Argon', 'Krypton', 'Xenon', 'Radon', 'Oganesson'],
    'Lanthanides': ['Lanthanum', 'Cerium', 'Praseodymium', 'Neodymium', 'Promethium', 'Samarium',
                    'Europium', 'Gadolinium', 'Terbium', 'Dysprosium', 'Holmium', 'Erbium', 'Thulium',
                    'Ytterbium', 'Lutetium'],
    'Actinides': ['Actinium', 'Thorium', 'Protactinium', 'Uranium', 'Neptunium', 'Plutonium', 'Americium',
                  'Curium', 'Berkelium', 'Californium', 'Einsteinium', 'Fermium', 'Mendelevium', 'Nobelium',
                  'Lawrencium', 'Rutherfordium', 'Dubnium', 'Seaborgium', 'Bohrium', 'Hassium',
                  'Meitnerium', 'Darmstadtium', 'Roentgenium', 'Copernicium', 'Nihonium', 'Flerovium',
                  'Moscovium', 'Livermorium', 'Tennessine', 'Oganesson']
}
def find_type(string, elements_dict):
    for key, element_list in elements_dict.items():
        if string in element_list:
            return key
    return None


for medium in ["Air", "Vacuum"]:
    intensity_list = []
    persistence_list = []
    wavelength_list = []
    element_list = []
    type_list = []
    for element in elements:
        if element.text != '':
            print(element.text)
            table_URL = table_template.format(element.text.lower())
            print(table_URL)
            response = requests.get(table_URL)
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')
            for table in tables:
                if medium in table.text:
                    for row in table.find_all('tr'):
                        if len(row.text.strip()) != 0:
                            if row.text.strip()[0].isdigit():
                                split_text = row.text.strip().split('\xa0')
                                print(split_text)
                                # There are two different ways lines are formatted, manage them separately
                                if len(split_text) == 1:
                                    split_text = row.text.strip().split(' ')
                                    intensity = split_text[0]
                                    wavelength = re.sub(r'[^0-9.]', '', split_text[1].split(' ')[0])
                                    persistense = 0 
                                    if "P" in split_text[-2]:
                                        persistence = 1
                                else: 
                                    intensity = split_text[0].split(' ')[0]
                                    wavelength = split_text[1].split(' ')[0][:-2]
                                    if "P" in split_text[0].split(' ')[-1]:
                                        persistence = 1
                                    else:
                                        persistence = 0 

                                intensity_list.append(intensity)
                                persistence_list.append(persistence)
                                wavelength_list.append(wavelength)
                                element_list.append(element.text)
                                type_list.append(find_type(element.text, elements_types))

    elements_df = pd.DataFrame({
        'element': element_list,
        'wavelength': wavelength_list,
        'intensity': intensity_list,
        'persistence': persistence_list,
        'type': type_list

    })
    elements_df.to_csv(f'./data/{medium.lower()}_elements.csv', index=False)

