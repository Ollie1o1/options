"""Tests for src/insider — Form 4 parsing + cluster-buy scoring (offline)."""
from __future__ import annotations

import unittest

from src.insider.parse import parse_form4
from src.insider.signal import cluster_score


FORM4_BUY = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerCik>0000320193</issuerCik><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector><isOfficer>0</isOfficer>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-06-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>100.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

FORM4_SELL = FORM4_BUY.replace(">P<", ">S<").replace(">A<", ">D<").replace(
    "DOE JANE", "ROE RICHARD")

FORM4_OFFICER_BUY = FORM4_BUY.replace("DOE JANE", "SMITH ALEX").replace(
    "<isDirector>1</isDirector><isOfficer>0</isOfficer>",
    "<isDirector>0</isDirector><isOfficer>1</isOfficer>")


class ParseForm4Test(unittest.TestCase):
    def test_parses_open_market_buy(self):
        txs = parse_form4(FORM4_BUY)
        self.assertEqual(len(txs), 1)
        t = txs[0]
        self.assertEqual(t["owner"], "DOE JANE")
        self.assertTrue(t["is_director"])
        self.assertFalse(t["is_officer"])
        self.assertEqual(t["code"], "P")
        self.assertEqual(t["shares"], 5000.0)
        self.assertEqual(t["price"], 100.50)
        self.assertEqual(t["value"], 5000.0 * 100.50)
        self.assertEqual(t["date"], "2026-06-01")

    def test_parses_sell(self):
        t = parse_form4(FORM4_SELL)[0]
        self.assertEqual(t["code"], "S")

    def test_garbage_returns_empty(self):
        self.assertEqual(parse_form4("<not-xml"), [])
        self.assertEqual(parse_form4(""), [])


def _tx(owner="DOE JANE", code="P", value=600_000.0, date="2026-06-01",
        officer=False, director=True):
    return {"owner": owner, "is_officer": officer, "is_director": director,
            "code": code, "shares": 1000, "price": value / 1000,
            "value": value, "date": date}


class ClusterScoreTest(unittest.TestCase):
    def test_cluster_buy_two_insiders(self):
        s = cluster_score([_tx("DOE JANE"), _tx("SMITH ALEX", officer=True)],
                          today="2026-06-11")
        self.assertGreaterEqual(s["score"], 0.8)
        self.assertEqual(s["label"], "CLUSTER BUY")
        self.assertEqual(s["n_buyers"], 2)

    def test_single_notable_officer_buy(self):
        s = cluster_score([_tx("SMITH ALEX", officer=True, value=250_000)],
                          today="2026-06-11")
        self.assertEqual(s["label"], "NOTABLE BUY")
        self.assertGreaterEqual(s["score"], 0.5)
        self.assertLess(s["score"], 0.8)

    def test_small_single_buy_weak(self):
        s = cluster_score([_tx(value=20_000)], today="2026-06-11")
        self.assertEqual(s["label"], "WEAK BUY")

    def test_sells_do_not_score_but_are_counted(self):
        s = cluster_score([_tx(code="S"), _tx("ROE RICHARD", code="S")],
                          today="2026-06-11")
        self.assertEqual(s["score"], 0.0)
        self.assertEqual(s["label"], "NONE")
        self.assertGreater(s["sell_value"], 0)

    def test_old_buys_outside_window_ignored(self):
        s = cluster_score([_tx(date="2025-12-01")], today="2026-06-11",
                          window_days=90)
        self.assertEqual(s["label"], "NONE")

    def test_same_owner_two_buys_is_not_a_cluster(self):
        s = cluster_score([_tx(date="2026-06-01"), _tx(date="2026-06-05")],
                          today="2026-06-11")
        self.assertEqual(s["n_buyers"], 1)
        self.assertNotEqual(s["label"], "CLUSTER BUY")


if __name__ == "__main__":
    unittest.main()
