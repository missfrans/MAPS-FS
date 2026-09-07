from mapsfs.config import stable_hash


def test_stable_hash_order_independent_for_dicts():
    assert stable_hash({"a":1,"b":2}) == stable_hash({"b":2,"a":1})
