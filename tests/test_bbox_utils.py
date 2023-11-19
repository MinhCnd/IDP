import pytest
from idp.annotations.bbox_utils import is_box_a_within_box_b, merge_box_extremes


class TestMergeBoxExtremes:
    def test_with_multiple_boxes(self):
        assert merge_box_extremes([[1, 1, 3, 3], [2, 2, 4, 4]]) == [1, 1, 4, 4]

    def test_with_single_box(self):
        assert merge_box_extremes([[2, 2, 5, 5]]) == [2, 2, 5, 5]

    def test_with_overlapping_boxes(self):
        assert merge_box_extremes([[1, 1, 4, 4], [2, 2, 3, 3]]) == [1, 1, 4, 4]

    def test_with_non_overlapping_boxes(self):
        assert merge_box_extremes([[1, 1, 2, 2], [3, 3, 4, 4]]) == [1, 1, 4, 4]

    def test_with_negative_coordinates(self):
        assert merge_box_extremes([[-3, -3, -1, -1], [1, 1, 3, 3]]) == [-3, -3, 3, 3]

    def test_with_empty_box_list(self):
        assert merge_box_extremes([]) == []

    def test_with_invalid_box_format(self):
        with pytest.raises(ValueError):
            merge_box_extremes([[1, 2], [3, 4, 5, 6]])


class TestBoxAWithinB:
    def test_box_a_completely_inside_box_b(self):
        assert is_box_a_within_box_b([2, 2, 4, 4], [1, 1, 5, 5])

    def test_box_a_partially_inside_box_b(self):
        assert not is_box_a_within_box_b([0, 2, 4, 4], [1, 1, 5, 5])

    def test_box_a_completely_outside_box_b(self):
        assert not is_box_a_within_box_b([6, 6, 8, 8], [1, 1, 5, 5])

    def test_box_a_equal_to_box_b(self):
        assert is_box_a_within_box_b([1, 1, 5, 5], [1, 1, 5, 5])

    def test_negative_coordinates(self):
        assert is_box_a_within_box_b([-3, -3, -1, -1], [-4, -4, 0, 0])

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            is_box_a_within_box_b([1, 2, 3], [1, 2, 3, 4])
