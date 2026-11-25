from lgor.gtlines.wdvv import ProductTFT


def test_wdvv_44_consistent_and_unique():
    T = ProductTFT([4, 4])
    assert T.check(4) == 0 and T.check(5) == 0        # product_corr input satisfies WDVV
    r = T.extend(6)
    assert r["status"] == "unique" and r["unknowns"] == 46
