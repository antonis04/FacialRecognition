import numpy as np

FEATURE_NAMES = [
    'rozmiar_glowy', 'obwod_twarzy', 'indeks_twarzy', 'kat_twarzowy',
    'wysokosc_twarzy', 'calkowita_dlugosc_twarzy', 'szerokosc_bizygomatyczna',
    'wysokosc_czola', 'nachylenie_czola', 'kat_nosowo_czolowy',
    'szerokosc_miedzy_kacikami_wew', 'odleglosc_miedzy_zrenicami',
    'odleglosc_miedzy_zew_kacikami', 'glebokosc_oczodolu', 'szerokosc_dwuskroniowa',
    'indeks_nosowy', 'dlugosc_nosa', 'kat_nosowo_wargowy', 'szerokosc_zuchwy',
    'kat_zuchwy', 'dlugosc_szczeki', 'pochylenie_wargi_górnej', 'kat_NA_FH',
    'kat_wargowo_brodkowy', 'wysokosc_wargi_górnej', 'wysokosc_wargi_dolnej'
]

female_values = {
    'rozmiar_glowy': 78.5, 'obwod_twarzy': 544, 'indeks_twarzy': 83,
    'kat_twarzowy': 166.5, 'wysokosc_twarzy': 112, 'calkowita_dlugosc_twarzy': 172,
    'szerokosc_bizygomatyczna': 132.5, 'wysokosc_czola': 50, 'nachylenie_czola': 6,
    'kat_nosowo_czolowy': 136.5, 'szerokosc_miedzy_kacikami_wew': 32,
    'odleglosc_miedzy_zrenicami': 65, 'odleglosc_miedzy_zew_kacikami': 88,
    'glebokosc_oczodolu': 9, 'szerokosc_dwuskroniowa': 110, 'indeks_nosowy': 64,
    'dlugosc_nosa': 42.5, 'kat_nosowo_wargowy': 105, 'szerokosc_zuchwy': 90,
    'kat_zuchwy': 122.5, 'dlugosc_szczeki': 50, 'pochylenie_wargi_górnej': 14,
    'kat_NA_FH': 88, 'kat_wargowo_brodkowy': 125, 'wysokosc_wargi_górnej': 7,
    'wysokosc_wargi_dolnej': 11
}

male_values = {
    'rozmiar_glowy': 79.5, 'obwod_twarzy': 569, 'indeks_twarzy': 86,
    'kat_twarzowy': 162.5, 'wysokosc_twarzy': 120, 'calkowita_dlugosc_twarzy': 190,
    'szerokosc_bizygomatyczna': 142.5, 'wysokosc_czola': 60, 'nachylenie_czola': 10,
    'kat_nosowo_czolowy': 126.5, 'szerokosc_miedzy_kacikami_wew': 33,
    'odleglosc_miedzy_zrenicami': 65, 'odleglosc_miedzy_zew_kacikami': 91,
    'glebokosc_oczodolu': 11, 'szerokosc_dwuskroniowa': 115, 'indeks_nosowy': 66,
    'dlugosc_nosa': 51.5, 'kat_nosowo_wargowy': 92.5, 'szerokosc_zuchwy': 105,
    'kat_zuchwy': 128.5, 'dlugosc_szczeki': 54, 'pochylenie_wargi_górnej': 8,
    'kat_NA_FH': 85, 'kat_wargowo_brodkowy': 130, 'wysokosc_wargi_górnej': 8,
    'wysokosc_wargi_dolnej': 12
}

def get_feature_vector(gender):
    if gender == 'female':
        values = female_values
    else:
        values = male_values
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)

def get_default_feature_vector():
    f_female = get_feature_vector('female')
    f_male = get_feature_vector('male')
    return (f_female + f_male) / 2.0