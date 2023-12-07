import pytest
from idp.main import process_model_output
from idp.annotations.annotation_utils import CLASS_TO_LABEL_STR_MAP, Classes


class TestProcessModelOutput:
    def test_empty_input(self):
        with pytest.raises(ValueError):
            process_model_output({})

    def test_invalid_charge_text_format(self):
        input = {
            CLASS_TO_LABEL_STR_MAP[Classes.WATER_CONSUMPTION]: "Water consumption",
            CLASS_TO_LABEL_STR_MAP[Classes.WASTEWATER_CONSUMPTION]: "2026",
        }

        with pytest.raises(ValueError):
            process_model_output(input)

    def test_invalid_consumption_text_format(self):
        input = {
            CLASS_TO_LABEL_STR_MAP[Classes.WATER_CONSUMPTION_DETAILS]: "Water",
            CLASS_TO_LABEL_STR_MAP[Classes.WASTEWATER_CONSUMPTION_DETAILS]: "$51.22",
        }

        with pytest.raises(ValueError):
            process_model_output(input)

    def test_irrelevant_keys(self):
        input = {"Wastewater_days": "33 days"}

        with pytest.raises(ValueError):
            process_model_output(input)

    def test_valid_input(self):
        input = {
            CLASS_TO_LABEL_STR_MAP[Classes.BALANCE_STILL_OWING]: "$0.00",
            CLASS_TO_LABEL_STR_MAP[Classes.WASTEWATER_CONSUMPTION]: "$ 37.53",
            CLASS_TO_LABEL_STR_MAP[Classes.WATER_CONSUMPTION_DETAILS]: "22.00 kL",
            CLASS_TO_LABEL_STR_MAP[Classes.WASTEWATER_CONSUMPTION_DETAILS]: "17.27kL",
            CLASS_TO_LABEL_STR_MAP[Classes.WASTEWATER_FIXED_DETAILS]: "32 days",
            CLASS_TO_LABEL_STR_MAP[
                Classes.THIS_READING
            ]: "This reading 24-May-22 423 Actual",
            CLASS_TO_LABEL_STR_MAP[
                Classes.LAST_READING
            ]: "Last reading 22-Apr-22 401 Estimate",
        }

        output = {
            CLASS_TO_LABEL_STR_MAP[Classes.BALANCE_STILL_OWING]: 0.00,
            CLASS_TO_LABEL_STR_MAP[Classes.WASTEWATER_CONSUMPTION]: 37.53,
            CLASS_TO_LABEL_STR_MAP[Classes.WATER_CONSUMPTION_DETAILS]: 22.00,
            CLASS_TO_LABEL_STR_MAP[Classes.WASTEWATER_CONSUMPTION_DETAILS]: 17.27,
            CLASS_TO_LABEL_STR_MAP[Classes.WASTEWATER_FIXED_DETAILS]: 32,
            CLASS_TO_LABEL_STR_MAP[Classes.THIS_READING]: "24-May-22",
            CLASS_TO_LABEL_STR_MAP[Classes.LAST_READING]: "22-Apr-22",
        }

        assert process_model_output(input) == output
