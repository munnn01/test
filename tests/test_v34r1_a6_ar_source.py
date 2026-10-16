import unittest
from adaptive_vcm.v34r1_a6.ar_source import provider_window,historical_parity


class ProviderWindow(unittest.TestCase):
    def test_matches_original_center_stride(self):
        names=[f'img_{i:05d}.jpg' for i in range(1,123)]
        window,selected=provider_window(names,122)
        self.assertEqual(window,[f'img_{i:05d}.jpg' for i in range(46,78)])
        self.assertEqual(selected,[f'img_{i:05d}.jpg' for i in range(46,78,2)])

    def test_incomplete_directory_no_substitution(self):
        for names,count in [(['img_00001.jpg']*32,32),([f'img_{i:05d}.jpg' for i in range(1,32)],31),
            ([f'img_{i:05d}.jpg' for i in range(2,34)],32),([f'img_{i:05d}.jpg' for i in range(1,33)]+['note.txt'],32)]:
            with self.assertRaises(ValueError):provider_window(names,count)

    def test_historical_16_or_32_hashes_preserved_without_inventing_old_hashes(self):
        window=[f'img_{i:05d}.jpg' for i in range(1,33)];selected=['video/'+n for n in window[::2]]
        hashes={n:'sha-'+n for n in window}
        for declared in (window,window[::2]):
            old=dict(provider_window_files=declared,provider_selected_paths=selected,file_sha256={n:hashes[n] for n in declared})
            self.assertEqual(historical_parity(hashes,selected,old),len(declared))
            bad=dict(hashes);bad[declared[0]]='different'
            with self.assertRaises(ValueError):historical_parity(bad,selected,old)
        with self.assertRaises(ValueError):historical_parity(hashes,list(reversed(selected)),old)


if __name__=='__main__':unittest.main()
