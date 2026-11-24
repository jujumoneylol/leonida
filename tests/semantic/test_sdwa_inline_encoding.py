"""Regression tests for LLVM 19.1.7 printing 8-byte SDWA inline-constant instructions as 12 bytes.

Covers both unsigned (v_cvt_f16_u16_sdwa, as in ps_sdwa_f16_const) and signed (v_cvt_f16_i16_sdwa) conversions
where the stored RDNA2 blob holds an 8-byte SDWA encoding with an inline constant src0 (0x80..0xf8, s0=1) but
LLVM 19 prints src0=0xff with a trailing 4-byte literal word. Verifies check_printed_encodings recovers the native
operand, shifts subsequent instruction PCs back by -4 bytes, and rejects unrelated encoding mismatches.
"""
import os
import tempfile
import unittest

import support

rdna2spirv = support.rdna2spirv


class SdwaInlineEncodingTests(unittest.TestCase):
    def _run_check(self, raw_blob, listing_lines):
        with tempfile.TemporaryDirectory() as tmp:
            fssm_dir = os.path.join(tmp, "fssm")
            disasm_dir = os.path.join(tmp, "disasm")
            os.makedirs(fssm_dir)
            os.makedirs(disasm_dir)
            name = "ps_sdwa_inline_test"
            code_off = 0x180
            blob = bytes(code_off) + raw_blob
            with open(os.path.join(fssm_dir, name + ".fssm"), "wb") as f:
                f.write(blob)
            s_path = os.path.join(disasm_dir, name + ".s")
            header = f"; {name}  code 0x{code_off:x}..0x{len(blob):x} (synthetic test)\n"
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(header + "".join(listing_lines))
            info = rdna2spirv.parse_file(s_path)
            rdna2spirv.check_printed_encodings(s_path, info)
            return info

    def test_v_cvt_f16_u16_and_i16_sdwa_12byte_llvm19_print(self):
        # Exact 8-byte encodings from ps_sdwa_f16_const (u16 opc=0x50 -> 0xa0, i16 opc=0x51 -> 0xa2) + s_endpgm.
        raw_u16 = bytes([0xF9, 0xA0, 0x0A, 0x7E, 0xF2, 0x14, 0x85, 0x00])  # v_cvt_f16_u16_sdwa v5, 1.0 ...
        raw_i16 = bytes([0xF9, 0xA2, 0x0C, 0x7E, 0xF4, 0x14, 0x85, 0x00])  # v_cvt_f16_i16_sdwa v6, 2.0 ...
        raw_end = bytes([0x00, 0x00, 0x81, 0xBF])                           # s_endpgm

        # Simulated LLVM 19.1.7 12-byte print: byte 4 changed to 0xff + 4-byte literal word appended.
        prt_u16 = bytes([0xF9, 0xA0, 0x0A, 0x7E, 0xFF, 0x14, 0x85, 0x00, 0x00, 0x00, 0x80, 0x3F])
        prt_i16 = bytes([0xF9, 0xA2, 0x0C, 0x7E, 0xFF, 0x14, 0x85, 0x00, 0x00, 0x00, 0x00, 0x40])

        def enc_comment(b):
            return "; encoding: [" + ",".join(f"0x{x:02x}" for x in b) + "]"

        listing = [
            f"\tv_cvt_f16_u16_sdwa v5, 0x3f800000 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 {enc_comment(prt_u16)}\n",
            f"\tv_cvt_f16_i16_sdwa v6, 0x40000000 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 {enc_comment(prt_i16)}\n",
            f"\ts_endpgm {enc_comment(raw_end)}\n",
        ]
        info = self._run_check(raw_u16 + raw_i16 + raw_end, listing)
        insns = info["insns"]
        self.assertEqual(len(insns), 3)
        # Recovered native inline float constants and 8-byte encodings.
        self.assertEqual(insns[0]["args"][1], {"k": "fimm", "v": 1.0})
        self.assertEqual(insns[0]["enc"], raw_u16.hex())
        self.assertEqual(insns[0]["pc"], 0x180)
        self.assertEqual(insns[1]["args"][1], {"k": "fimm", "v": 2.0})
        self.assertEqual(insns[1]["enc"], raw_i16.hex())
        self.assertEqual(insns[1]["pc"], 0x188)
        # s_endpgm PC must be shifted back by -8 total bytes (two 12->8 recoveries) to 0x190.
        self.assertEqual(insns[2]["pc"], 0x190)

    def test_sdwa_mismatch_in_other_fields_still_rejected(self):
        raw_u16 = bytes([0xF9, 0xA0, 0x0A, 0x7E, 0xF2, 0x14, 0x85, 0x00])
        raw_end = bytes([0x00, 0x00, 0x81, 0xBF])
        # Corrupt src0_sel byte (0x85 -> 0x84): must NOT be silently accepted!
        bad_prt = bytes([0xF9, 0xA0, 0x0A, 0x7E, 0xFF, 0x14, 0x84, 0x00, 0x00, 0x00, 0x80, 0x3F])
        enc = "; encoding: [" + ",".join(f"0x{x:02x}" for x in bad_prt) + "]"
        listing = [
            f"\tv_cvt_f16_u16_sdwa v5, 0x3f800000 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 {enc}\n",
            "\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            self._run_check(raw_u16 + raw_end, listing)


if __name__ == "__main__":
    unittest.main()
