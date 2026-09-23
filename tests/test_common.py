import pandas as pd

from bpp.common import (former_names, fy_of_date, is_private_company, normalize_company_name,
                        parse_fy_label, to_timestamp)


def test_normalize_company_name():
    assert normalize_company_name("The Jaypee Infratech Ltd.") == "JAYPEE INFRATECH"
    assert normalize_company_name("ABC Steel & Power Limited (In Liquidation)") == "ABC STEEL AND POWER"
    assert normalize_company_name("XYZ India Limited") == "XYZ INDIA"          # INDIA is kept on purpose


def test_private_company_detection():
    assert is_private_company("Brisk Logistics Pvt. Ltd.")
    assert is_private_company("Halcyon Agro Foods Private Limited")
    assert not is_private_company("Rudrasen Power Limited")


def test_former_names():
    assert former_names("Kalpavik Textiles Limited (formerly known as Kalpavik Spinners Limited)") == \
        ["Kalpavik Spinners Limited"]


def test_fiscal_years():
    assert parse_fy_label("2022-23") == 2023
    assert parse_fy_label("FY23") == 2023
    assert parse_fy_label("FY2019.pdf") == 2019
    assert parse_fy_label(2021) == 2021
    assert fy_of_date(pd.Timestamp("2022-05-10")) == 2023
    assert fy_of_date(pd.Timestamp("2023-03-31")) == 2023


def test_dates_iso_and_indian():
    assert to_timestamp("2019-09-04") == pd.Timestamp("2019-09-04")     # ISO is never day-first
    assert to_timestamp("04-09-2019") == pd.Timestamp("2019-09-04")     # Indian dd-mm-yyyy
    assert to_timestamp("") is None
