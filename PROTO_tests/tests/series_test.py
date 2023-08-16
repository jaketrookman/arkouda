import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import arkouda as ak


class TestSeries:
    def test_series_creation(self):
        idx = ak.arange(3)
        for val in ak.arange(3), ak.array(["A", "B", "C"]):
            ans = ak.Series(data=val, index=idx).to_list()
            for series in (
                ak.Series(data=val, index=idx),
                ak.Series(data=val),
                ak.Series(val, idx),
                ak.Series(val),
                ak.Series((idx, val)),
            ):
                assert isinstance(series, ak.Series)
                assert isinstance(series.index, ak.Index)
                assert series.to_list() == ans

        with pytest.raises(TypeError):
            ak.Series(index=idx)

        with pytest.raises(TypeError):
            ak.Series((ak.arange(3),))

        with pytest.raises(TypeError):
            ak.Series()

        with pytest.raises(ValueError):
            ak.Series(data=ak.arange(3), index=ak.arange(6))

    def test_lookup(self):
        for val in ak.arange(3), ak.array(["A", "B", "C"]):
            s = ak.Series(data=val, index=ak.arange(3))

            for key in 1, [0, 2], ak.Index([1]), ak.Index([0, 2]):
                lk = s.locate(key)
                assert isinstance(lk, ak.Series)
                key = ak.array(key) if not isinstance(key, int) else key
                assert (lk.index == s.index[key]).all()
                assert (lk.values == s.values[key]).all()

            # testing multi-index lookup
            mi = ak.MultiIndex([ak.arange(3), ak.array([2, 1, 0])])
            s = ak.Series(data=val, index=mi)
            lk = s.locate(mi[0])
            assert isinstance(lk, ak.Series)
            assert lk.index.index == mi[0].index
            assert lk.values[0] == val[0]

            # ensure error with scalar and multi-index
            with pytest.raises(TypeError):
                s.locate(0)

            with pytest.raises(TypeError):
                s.locate([0, 2])

    def test_add(self):
        added = ak.Series(ak.arange(3)).add(ak.Series(data=ak.arange(3, 6), index=ak.arange(3, 6)))
        assert added.index.to_list() == list(range(6))
        assert added.values.to_list() == list(range(6))

    def test_topn(self):
        top = ak.Series(ak.arange(30)).topn(10)
        assert top.values.to_list() == list(range(29, 19, -1))
        assert top.index.to_list() == list(range(29, 19, -1))

    def test_sort(self):
        ordered = ak.arange(5)
        perm = ak.array([3, 1, 4, 0, 2])

        idx_sort = ak.Series(data=ordered, index=perm).sort_index()
        assert idx_sort.index.to_list() == ordered.to_list()
        assert (idx_sort.values == perm).all()

        val_sort = ak.Series(data=perm, index=ordered).sort_values()
        assert val_sort.index.to_pandas().tolist() == perm.to_list()
        assert (val_sort.values == ordered).all()

    def test_head_tail(self):
        n = 10
        s = ak.Series(ak.arange(n))

        for i in range(n):
            head = s.head(i)
            assert head.index.to_list() == list(range(i))
            assert head.values.to_list() == list(range(i))

            tail = s.tail(i)
            assert tail.index.to_list() == ak.arange(n)[-i:n].to_list()
            assert tail.values.to_list() == ak.arange(n)[-i:n].to_list()

    def test_value_counts(self):
        s = ak.Series(ak.array([1, 2, 0, 2, 0]))

        c = s.value_counts()
        assert c.index.to_list() == [0, 2, 1]
        assert c.values.to_list() == [2, 2, 1]

        c = s.value_counts(sort=False)
        assert c.index.to_list() == list(range(3))
        assert c.values.to_list() == [2, 1, 2]

    def test_concat(self):
        s = ak.Series(ak.arange(5))
        s2 = ak.Series(ak.arange(5, 11), ak.arange(5, 11))
        s3 = ak.Series(ak.arange(5, 10), ak.arange(5, 10))

        df = ak.Series.concat([s, s2], axis=1)
        assert isinstance(df, ak.DataFrame)

        ref_df = pd.DataFrame(
            {
                "idx": list(range(11)),
                "val_0": [0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
                "val_1": [0, 0, 0, 0, 0, 5, 6, 7, 8, 9, 10],
            }
        )
        assert_frame_equal(ref_df, df.to_pandas())

        def list_helper(arr):
            return arr.to_list() if isinstance(arr, (ak.pdarray, ak.Index)) else arr.tolist()

        for fname in "concat", "pdconcat":
            func = getattr(ak.Series, fname)
            c = func([s, s2])
            assert list_helper(c.index) == list(range(11))
            assert list_helper(c.values) == list(range(11))

            df = func([s, s3], axis=1)
            if fname == "concat":
                ref_df = pd.DataFrame(
                    {"idx": [0, 1, 2, 3, 4], "val_0": [0, 1, 2, 3, 4], "val_1": [5, 6, 7, 8, 9]}
                )
                assert isinstance(df, ak.DataFrame)
                assert_frame_equal(ref_df, df.to_pandas())
            else:
                ref_df = pd.DataFrame({0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]})
                assert isinstance(df, pd.DataFrame)
                assert_frame_equal(ref_df, df)

    def test_index_as_index_compat(self):
        # added to validate functionality for issue #1506
        df = ak.DataFrame({"a": ak.arange(10), "b": ak.arange(10), "c": ak.arange(10)})
        g = df.groupby(["a", "b"])
        g.broadcast(g.sum("c"))
